import argparse
from pathlib import Path

import pandas as pd


def prepare(args):
    if args.available_delay_minutes < 0:
        raise ValueError("Availability delay must be nonnegative")
    files = sorted(Path(args.data_dir).resolve().rglob("*.npy"))
    if not files:
        raise ValueError("No .npy files found")
    rows = [dict(name=path.stem, path=str(path), split="frames") for path in files]
    for split in ("train", "valid", "test"):
        names = pd.read_csv(getattr(args, f"{split}_txt"), header=None, names=["name"],
                            dtype=str, comment="#")["name"].str.strip()
        if names.isna().any():
            raise ValueError(f"Missing filename in {split} list")
        rows.extend(dict(name=Path(name).stem, path="", split=split) for name in names)
    df = pd.DataFrame(rows)
    local_times = pd.to_datetime(df["name"], format=args.filename_format)
    df["timestamp"] = pd.to_datetime(local_times.dt.strftime("%Y-%m-%dT%H:%M:%S") + args.time_offset, utc=True)
    frames = df[df["split"] == "frames"].copy()
    if frames["timestamp"].duplicated().any():
        raise ValueError("Multiple .npy files have the same observation time")
    frames["available_at"] = frames["timestamp"] + pd.to_timedelta(args.available_delay_minutes, unit="min")
    targets = {split: df.loc[df["split"] == split, ["timestamp"]].copy() for split in ("train", "valid", "test")}
    for split, table in targets.items():
        if table.empty or table["timestamp"].duplicated().any():
            raise ValueError(f"Empty or duplicate {split} targets")
        missing = set(table["timestamp"]) - set(frames["timestamp"])
        if missing:
            raise ValueError(f"Missing {split} target files, first timestamp: {min(missing)}")
    for left, right in (("train", "valid"), ("train", "test"), ("valid", "test")):
        if set(targets[left]["timestamp"]) & set(targets[right]["timestamp"]):
            raise ValueError(f"{left} and {right} targets overlap")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    frames[["timestamp", "path", "available_at"]].to_csv(output / "frames.csv", index=False)
    for split, table in targets.items():
        table.to_csv(output / f"{split}_targets.csv", index=False)
    pd.DataFrame([dict(time_offset=args.time_offset, available_delay_minutes=args.available_delay_minutes,
                       filename_format=args.filename_format)]).to_csv(output / "time_settings.csv", index=False)
    print(f"Saved {len(frames)} frame records. Target counts: "
          + ", ".join(f"{split}={len(table)}" for split, table in targets.items()))
    print("Use frames.csv for all three *_csv settings and the separate *_targets.csv files.")


def main():
    parser = argparse.ArgumentParser(description="Prepare forecast CSV files")
    for name in ("data-dir", "train-txt", "valid-txt", "test-txt", "output", "time-offset"):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument("--available-delay-minutes", type=int, required=True)
    parser.add_argument("--filename-format", default="%Y-%m-%d_%H%M")
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
