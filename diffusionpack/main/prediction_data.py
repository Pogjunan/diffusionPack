from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class ForecastDataset(Dataset):
    def __init__(self, csv_path, *, history_steps=5, map_channels=1, resolution=32,
                 cadence_minutes=30, lead_minutes=30, value_min=None, value_max=None,
                 require_target=True, origin_time=None, history_lag_minutes=0, future_steps=1,
                 target_csv=None):
        if any(type(n) is not int or n < 1 for n in (history_steps, future_steps, map_channels, resolution)):
            raise ValueError("History, future, channels and resolution must be positive integers")
        if cadence_minutes < 1 or lead_minutes < 1:
            raise ValueError("Cadence and lead must be strictly positive")
        if type(history_lag_minutes) is not int or history_lag_minutes < 0:
            raise ValueError("history_lag_minutes must be a nonnegative integer")
        if value_min is None or value_max is None or not np.isfinite([value_min, value_max]).all() or value_max <= value_min:
            raise ValueError("Set finite physical value_min and value_max with max > min")
        csv_path = Path(csv_path).resolve()
        df = pd.read_csv(csv_path)
        columns = ["timestamp", "path", "available_at"]
        if df.empty or not all(column in df.columns for column in columns):
            raise ValueError("CSV needs timestamp, path and available_at columns with data")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
        df["available_at"] = pd.to_datetime(df["available_at"], utc=True, format="ISO8601")
        if df[columns].isna().any().any():
            raise ValueError("Missing timestamp, path or available_at")
        if df["timestamp"].duplicated().any():
            raise ValueError("Duplicate timestamps")
        if (df["available_at"] < df["timestamp"]).any():
            raise ValueError("Availability predates observation")

        self.records = {}
        for time, path, available in zip(df["timestamp"], df["path"], df["available_at"]):
            path = csv_path.parent / path
            if path.suffix.lower() != ".npy" or (require_target and not path.is_file()):
                raise ValueError(f"Expected a .npy grid: {path}")
            self.records[time] = (path, available)
        self.history_steps = history_steps
        self.future_steps = future_steps
        self.map_channels = map_channels
        self.resolution = resolution
        self.cadence = timedelta(minutes=cadence_minutes)
        self.lead = timedelta(minutes=lead_minutes)
        self.value_min = float(value_min)
        self.value_max = float(value_max)
        self.require_target = require_target
        self.windows = []
        self.rejections = {"missing_history": 0, "late_history": 0, "missing_target": 0}
        allowed_targets = None
        if target_csv is not None:
            target_rows = pd.read_csv(target_csv)
            if target_rows.empty or "timestamp" not in target_rows:
                raise ValueError("Target CSV needs timestamp rows")
            target_times = pd.to_datetime(target_rows["timestamp"], utc=True, format="ISO8601")
            if target_times.isna().any() or target_times.duplicated().any():
                raise ValueError("Target times must be present and unique")
            allowed_targets = set(target_times)
            self.rejections["target_outside_split"] = 0
        lag = timedelta(minutes=history_lag_minutes)
        history_ends = sorted(self.records) if origin_time is None else [pd.to_datetime(origin_time, utc=True, format="ISO8601") - lag]
        history_times = self.records.keys()
        if not require_target:
            history_times = {time for time, (path, _) in self.records.items() if path.is_file()}
        for history_end in history_ends:
            origin = history_end + lag
            history = tuple(history_end - self.cadence * k for k in reversed(range(history_steps)))
            if any(moment not in history_times for moment in history):
                self.rejections["missing_history"] += 1
                continue
            if any(self.records[moment][1] > origin for moment in history):
                self.rejections["late_history"] += 1
                continue
            targets = tuple(origin + self.lead + self.cadence * k for k in range(future_steps))
            if require_target and any(moment not in self.records for moment in targets):
                self.rejections["missing_target"] += 1
                continue
            if allowed_targets is not None and any(moment not in allowed_targets for moment in targets):
                self.rejections["target_outside_split"] += 1
                continue
            self.windows.append((origin, history, targets))
        if not self.windows:
            raise ValueError(f"No valid prediction windows; exclusions={self.rejections}")

    def __len__(self):
        return len(self.windows)

    def load_grid(self, moment):
        path = self.records[moment][0]
        values = np.load(path, allow_pickle=False)
        if values.ndim == 2:
            values = values[None]
        shape = (self.map_channels, self.resolution, self.resolution)
        if values.shape != shape:
            raise ValueError(f"Expected {shape}, received {values.shape} at {path}")
        if not (np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.floating)):
            raise ValueError(f"Expected a real-valued grid at {path}")
        values = values.astype(np.float64, copy=False)
        scaled = 2 * ((values - self.value_min) / (self.value_max - self.value_min)) - 1
        scaled = scaled.astype(np.float32)
        if not np.isfinite(scaled).all():
            raise ValueError(f"Non-finite scaled grid at {path}")
        return torch.from_numpy(scaled)

    def __getitem__(self, index):
        origin, history, targets = self.windows[index]
        condition = torch.cat([self.load_grid(moment) for moment in history], dim=0)
        if self.require_target:
            future = torch.cat([self.load_grid(moment) for moment in targets], dim=0)
        else:
            future = torch.empty(0, dtype=condition.dtype)
        return condition, future, origin.isoformat(), tuple(moment.isoformat() for moment in targets)


def inverse_scale(image, value_min, value_max):
    if value_max <= value_min:
        raise ValueError("Physical scale must be increasing")
    return (image + 1) * ((value_max - value_min) / 2) + value_min
