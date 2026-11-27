"""Check the legacy-list bridge using temporary synthetic files only."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from prepare_scrd_data import prepare


@pytest.fixture
def bridge_args(tmp_path):
    data = tmp_path / "maps"
    data.mkdir()
    names = ["2026-11-26_0930", "2026-11-26_1000", "2026-11-26_1030", "2026-11-26_1100"]
    for index, name in enumerate(names):
        np.save(data / f"{name}.npy", np.full((2, 2), index, dtype=np.float32))
    paths = {}
    for split, selected in (("train", names[:2]), ("valid", names[2:3]), ("test", names[3:])):
        path = tmp_path / f"{split}.txt"
        path.write_text("# Existing target membership\n" + "\n".join(f"{name}.npy" for name in selected) + "\n")
        paths[f"{split}_txt"] = str(path)
    return SimpleNamespace(data_dir=str(data), output=str(tmp_path / "csv"), time_offset="+09:00",
                           available_delay_minutes=5, filename_format="%Y-%m-%d_%H%M", **paths)


def test_timezone_delay_and_split_membership_are_preserved(bridge_args):
    data = Path(bridge_args.data_dir)
    before = {path: path.read_bytes() for path in data.glob("*.npy")}
    prepare(bridge_args)
    output = Path(bridge_args.output)
    frames = pd.read_csv(output / "frames.csv")
    timestamps = pd.to_datetime(frames["timestamp"], utc=True)
    available = pd.to_datetime(frames["available_at"], utc=True)
    expected = pd.date_range("2026-11-26T00:30:00Z", periods=4, freq="30min")
    assert list(timestamps) == list(expected)
    assert ((available - timestamps) == pd.Timedelta(minutes=5)).all()
    assert all(Path(path).is_absolute() and Path(path).is_file() for path in frames["path"])
    for split, indices in (("train", [0, 1]), ("valid", [2]), ("test", [3])):
        targets = pd.read_csv(output / f"{split}_targets.csv")
        assert list(pd.to_datetime(targets["timestamp"], utc=True)) == list(expected[indices])
    settings = pd.read_csv(output / "time_settings.csv")
    assert settings.loc[0, "time_offset"] == "+09:00"
    assert settings.loc[0, "available_delay_minutes"] == 5
    assert {path: path.read_bytes() for path in data.glob("*.npy")} == before


def test_duplicate_targets_observations_and_split_overlap_are_rejected(bridge_args):
    train_list, valid_list = Path(bridge_args.train_txt), Path(bridge_args.valid_txt)
    original_train, original_valid = train_list.read_text(), valid_list.read_text()
    train_list.write_text("2026-11-26_0930.npy\n2026-11-26_0930.npy\n")
    with pytest.raises(ValueError, match="duplicate train targets"):
        prepare(bridge_args)
    assert not Path(bridge_args.output).exists()
    train_list.write_text(original_train)
    valid_list.write_text("2026-11-26_1000.npy\n")
    with pytest.raises(ValueError, match="train and valid targets overlap"):
        prepare(bridge_args)
    assert not Path(bridge_args.output).exists()
    valid_list.write_text(original_valid)
    duplicate_dir = Path(bridge_args.data_dir) / "another_copy"
    duplicate_dir.mkdir()
    np.save(duplicate_dir / "2026-11-26_0930.npy", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="same observation time"):
        prepare(bridge_args)
    assert not Path(bridge_args.output).exists()


def test_existing_output_is_never_overwritten(bridge_args):
    prepare(bridge_args)
    output = Path(bridge_args.output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare(bridge_args)
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before
