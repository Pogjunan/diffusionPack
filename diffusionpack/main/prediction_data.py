import csv
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def parse_time(value):
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Manifest timestamps must include UTC offset or Z")
    return parsed.astimezone(timezone.utc)


def read_manifest(path, *, check_files=True):
    path = Path(path).resolve()
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not {"timestamp", "path", "available_at"}.issubset(rows.fieldnames or ()):
            raise ValueError("Manifest needs timestamp,path,available_at columns")
        for row in rows:
            valid = parse_time(row["timestamp"])
            available = parse_time(row["available_at"])
            source = path.parent / row["path"]
            if valid in records:
                raise ValueError(f"Duplicate valid timestamp: {valid.isoformat()}")
            if source.suffix.lower() != ".npy" or (check_files and not source.is_file()):
                raise ValueError(f"Expected a .npy grid: {source}")
            if available < valid:
                raise ValueError(f"Availability predates observation: {valid.isoformat()}")
            records[valid] = (source, available)
    if not records:
        raise ValueError("Empty time-series manifest")
    return records


def manifest_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ForecastDataset(Dataset):
    def __init__(self, manifest, *, history_steps=5, map_channels=1, resolution=32,
                 cadence_minutes=30, lead_minutes=30, value_min=None, value_max=None,
                 require_target=True, origin_time=None, history_lag_minutes=0, future_steps=1):
        if any(type(n) is not int or n < 1 for n in (history_steps, future_steps, map_channels, resolution)):
            raise ValueError("History, future, channels and resolution must be positive integers")
        if cadence_minutes < 1 or lead_minutes < 1:
            raise ValueError("Cadence and lead must be strictly positive")
        if type(history_lag_minutes) is not int or history_lag_minutes < 0:
            raise ValueError("history_lag_minutes must be a nonnegative integer")
        if value_min is None or value_max is None or not np.isfinite([value_min, value_max]).all() or value_max <= value_min:
            raise ValueError("Set finite physical value_min and value_max with max > min")
        self.records = read_manifest(manifest, check_files=require_target)
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
        lag = timedelta(minutes=history_lag_minutes)
        history_ends = sorted(self.records) if origin_time is None else [parse_time(origin_time) - lag]
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
        # History: (H*C, height, width); future: (F*C, height, width).
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
