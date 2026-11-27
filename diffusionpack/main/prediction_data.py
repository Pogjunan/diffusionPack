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


def read_manifest(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    records = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = csv.DictReader(stream)
        if not {"timestamp", "path", "available_at"}.issubset(rows.fieldnames or ()):
            raise ValueError("Manifest needs timestamp,path,available_at columns")
        for row in rows:
            valid = parse_time(row["timestamp"])
            available = parse_time(row["available_at"])
            relative = Path(row["path"])
            source = relative if relative.is_absolute() else path.parent / relative
            if valid in records:
                raise ValueError(f"Duplicate valid timestamp: {valid.isoformat()}")
            if not source.is_file() or source.suffix.lower() != ".npy":
                raise ValueError(f"Missing .npy grid: {source}")
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
                 require_target=True, origin_time=None, history_lag_minutes=0):
        if history_steps < 1 or map_channels < 1 or resolution < 1:
            raise ValueError("History, channels and resolution must be positive")
        if cadence_minutes < 1 or lead_minutes < 1:
            raise ValueError("Cadence and lead must be strictly positive")
        if type(history_lag_minutes) is not int or history_lag_minutes < 0:
            raise ValueError("history_lag_minutes must be a nonnegative integer")
        if value_min is None or value_max is None or not np.isfinite([value_min, value_max]).all() or value_max <= value_min:
            raise ValueError("Set finite physical value_min and value_max with max > min")
        self.records = read_manifest(manifest)
        self.history_steps = history_steps
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
        for history_end in history_ends:
            origin = history_end + lag
            history = tuple(history_end - self.cadence * k for k in reversed(range(history_steps)))
            if any(moment not in self.records for moment in history):
                self.rejections["missing_history"] += 1
                continue
            if any(self.records[moment][1] > origin for moment in history):
                self.rejections["late_history"] += 1
                continue
            target = origin + self.lead
            if require_target and target not in self.records:
                self.rejections["missing_target"] += 1
                continue
            self.windows.append((origin, history, target))
        if not self.windows:
            raise ValueError(f"No valid prediction windows; exclusions={self.rejections}")

    def __len__(self):
        return len(self.windows)

    def load_grid(self, moment):
        path = self.records[moment][0]
        with path.open("rb") as stream:
            values = np.load(stream, allow_pickle=False)
        if values.ndim == 2:
            values = values[None]
        shape = (self.map_channels, self.resolution, self.resolution)
        if values.shape != shape:
            raise ValueError(f"Expected {shape}, received {values.shape} at {path}")
        if not np.issubdtype(values.dtype, np.number):
            raise ValueError(f"Non-numeric data at {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite grid values at {path}")
        if np.any(values < self.value_min) or np.any(values > self.value_max):
            raise ValueError(f"Out-of-range physical values at {path}")
        values = np.asarray(values, dtype=np.float32)
        return (torch.from_numpy(values) - self.value_min) * (2 / (self.value_max - self.value_min)) - 1

    def __getitem__(self, index):
        origin, history, target = self.windows[index]
        condition = torch.cat([self.load_grid(moment) for moment in history], dim=0)
        if self.require_target:
            future = self.load_grid(target)
        else:
            future = torch.empty(0, dtype=condition.dtype)
        return condition, future, origin.isoformat(), target.isoformat()


def inverse_scale(image, value_min, value_max):
    if value_max <= value_min:
        raise ValueError("Physical scale must be increasing")
    return (image + 1) * ((value_max - value_min) / 2) + value_min
