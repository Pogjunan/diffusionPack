import numpy as np
import pandas as pd
import pytest
import torch

from diffusionpack.scrd import runner
from diffusionpack.scrd import ForecastDiffusion
from diffusionpack.scrd.config import SCRDConfig


def write_maps(folder, start, count=8):
    folder.mkdir(parents=True)
    start = pd.Timestamp(start)
    pattern = np.linspace(0, 0.1, 256, dtype=np.float32).reshape(16, 16)
    rows = []
    for index in range(count):
        moment = start + pd.Timedelta(minutes=30 * index)
        path = folder / f"map_{index:02d}.npy"
        np.save(path, pattern + np.float32(0.1 + 0.05 * index))
        rows.append(dict(timestamp=moment.isoformat(), path=path.name, available_at=moment.isoformat()))
    csv_path = folder / "maps.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return str(csv_path)


def settings(train_csv, output, **changes):
    options = dict(train_csv=train_csv, valid_csv=None, test_csv=None,
                   history_steps=2, future_steps=2, map_channels=1, resolution=16,
                   cadence_minutes=30, lead_minutes=30, history_lag_minutes=0,
                   value_min=0., value_max=1., units="synthetic units",
                   widths=(8, 16), norm_groups=4, output_norm_groups=4,
                   time_dim=16, window=4, tcu_depth=1,
                   steps=1, batch_size=2, workers=0, cpu_threads=1, device="cpu",
                   save_every=1, log_every=1, validation_every=0, validation_origins=1,
                   progress=False, output=str(output), seed=11)
    options.update(changes)
    return SCRDConfig(**options)


@pytest.fixture
def small_run(tmp_path):
    csv_path = write_maps(tmp_path / "train", "2026-01-01T00:00:00Z")
    config = settings(csv_path, tmp_path / "run")
    path = runner.train(config)
    return config, path


def assert_same_state(first, second):
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, atol=0, rtol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_same_state(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_same_state(a, b)
    else:
        assert first == second


def test_resume_exactly_matches_three_uninterrupted_updates(tmp_path):
    csv_path = write_maps(tmp_path / "maps", "2026-01-01T00:00:00Z")
    full_path = runner.train(settings(csv_path, tmp_path / "full", steps=3))
    split_path = runner.train(settings(csv_path, tmp_path / "split", steps=2))
    split_state = torch.load(split_path, weights_only=True)
    assert split_state["cursor"] == 4 and split_state["epoch"] == 0
    resumed_path = runner.train(settings(csv_path, tmp_path / "split", steps=3, resume=str(split_path)))
    full = torch.load(full_path, weights_only=True)
    resumed = torch.load(resumed_path, weights_only=True)
    for key in ("model", "optimizer", "rng", "step", "epoch", "cursor", "best", "train_origins"):
        assert_same_state(full[key], resumed[key])
    assert full["step"] == 3 and full["epoch"] == 1 and full["cursor"] == 0
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "full/loss.csv"),
                                  pd.read_csv(tmp_path / "split/loss.csv"), check_exact=True)
    assert full["config"]["diffusion_steps"] == 1000


@pytest.mark.parametrize("changed", [{"units": "different units"}, {"value_max": 2.},
                                      {"target": "future"}, {"diffusion_steps": 999}])
def test_resume_rejects_changed_science_settings(small_run, tmp_path, changed):
    config, checkpoint = small_run
    options = dict(vars(config), steps=2, resume=str(checkpoint))
    options.update(changed)
    with pytest.raises(ValueError, match="settings differ"):
        runner.train(SCRDConfig(**options))


def test_inference_needs_history_only_and_keeps_checkpoint_units(small_run, tmp_path, monkeypatch):
    trained, checkpoint = small_run
    csv_path = write_maps(tmp_path / "test", "2026-03-01T00:00:00Z", count=2)
    calls = []

    def fixed_forecast(self, condition, **kwargs):
        calls.append(condition.shape)
        return self.last_frame(condition) + 0.2

    monkeypatch.setattr(ForecastDiffusion, "sample", fixed_forecast)
    output = tmp_path / "predictions"
    config = SCRDConfig(checkpoint=str(checkpoint), test_csv=csv_path, output=str(output),
                        origin_time="2026-03-01T00:30:00Z", samples=2, max_origins=1,
                        device="cpu", cpu_threads=1, progress=False)
    scores = runner.infer(config)
    assert scores.empty and len(calls) == 2
    records = pd.read_csv(output / "forecasts.csv")
    assert len(records) == 4
    assert set(records["units"]) == {"synthetic units"}
    assert list(records["member"]) == [1, 1, 2, 2]
    assert records["timestamp"].nunique() == 2
    for name in records["path"]:
        grid = np.load(output / name)
        assert grid.shape == (1, 16, 16) and np.isfinite(grid).all()
    assert not (output / "errors.csv").exists()
    assert not (output / "metrics.csv").exists()
    with pytest.raises(ValueError, match="No valid prediction windows"):
        runner.infer(SCRDConfig(**dict(vars(config), output=str(tmp_path / "requires_truth"))), evaluate=True)


def test_persistence_evaluation_matches_known_errors_without_checkpoint(tmp_path):
    csv_path = write_maps(tmp_path / "test", "2026-03-01T00:00:00Z", count=4)
    output = tmp_path / "evaluation"
    config = settings(None, output, test_csv=csv_path, checkpoint="does_not_exist.pt",
                      samples=2, max_origins=1)
    scores = runner.infer(config, evaluate=True, persistence=True).sort_values("lead_minutes")
    np.testing.assert_allclose(scores["mse"], [0.05 ** 2, 0.1 ** 2], atol=1e-8, rtol=1e-5)
    np.testing.assert_allclose(scores["mae"], [0.05, 0.1], atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(scores["bias"], [-0.05, -0.1], atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(scores["centered_mse"], 0, atol=1e-12)
    np.testing.assert_allclose(scores["sample_mse"], scores["mse"], atol=1e-10)
    assert len(pd.read_csv(output / "metrics.csv")) == 2
    assert len(pd.read_csv(output / "forecasts.csv")) == 2
    with pytest.raises(FileExistsError):
        runner.infer(config, evaluate=True, persistence=True)


def test_ensemble_mean_and_individual_sample_errors_are_distinct(small_run, tmp_path, monkeypatch):
    _, checkpoint = small_run
    csv_path = write_maps(tmp_path / "test", "2026-03-01T00:00:00Z", count=4)
    offsets = iter([0., 0.4])

    def two_members(self, condition, **kwargs):
        return self.last_frame(condition) + next(offsets)

    monkeypatch.setattr(ForecastDiffusion, "sample", two_members)
    config = SCRDConfig(checkpoint=str(checkpoint), test_csv=csv_path, output=str(tmp_path / "ensemble"),
                        samples=2, max_origins=1, device="cpu", cpu_threads=1, progress=False)
    scores = runner.infer(config, evaluate=True).sort_values("lead_minutes")
    np.testing.assert_allclose(scores["mse"], [0.0025, 0.], atol=1e-8, rtol=1e-5)
    np.testing.assert_allclose(scores["sample_mse"], [0.0125, 0.01], atol=1e-8, rtol=1e-5)


def test_evaluation_rejects_targets_seen_in_training(small_run, tmp_path):
    trained, checkpoint = small_run
    config = SCRDConfig(checkpoint=str(checkpoint), test_csv=trained.train_csv,
                        output=str(tmp_path / "leaking_test"), samples=1, max_origins=1,
                        device="cpu", cpu_threads=1, progress=False)
    with pytest.raises(ValueError, match="Test targets overlap"):
        runner.infer(config, evaluate=True)
    assert not (tmp_path / "leaking_test").exists()


def test_diagnosis_uses_noise_levels_and_correct_error_scaling(small_run, tmp_path):
    _, checkpoint = small_run
    csv_path = write_maps(tmp_path / "valid", "2026-02-01T00:00:00Z", count=4)
    output = tmp_path / "diagnosis"
    config = SCRDConfig(checkpoint=str(checkpoint), valid_csv=csv_path, output=str(output),
                        max_origins=1, device="cpu", cpu_threads=1, progress=False)
    result = runner.diagnose(config)
    assert list(result["t"]) == [0, 50, 200, 500, 799, 999]
    factor = (1 - result["alpha_bar"]) / result["alpha_bar"]
    np.testing.assert_allclose(result["clean_mse"], result["epsilon_mse"] * factor, rtol=5e-4, atol=1e-7)
    assert np.isfinite(result.select_dtypes("number").to_numpy()).all()
    assert len(pd.read_csv(output / "noise_levels.csv")) == 6


def test_check_data_blocks_overlapping_targets_and_invalid_scale(tmp_path):
    csv_path = write_maps(tmp_path / "maps", "2026-01-01T00:00:00Z")
    config = settings(csv_path, tmp_path / "unused", valid_csv=csv_path)
    with pytest.raises(ValueError, match="targets overlap"):
        runner.train(config, check_data=True)
    with pytest.raises(ValueError, match="value_min and value_max"):
        runner.train(settings(csv_path, tmp_path / "unused", value_min=1., value_max=1.), check_data=True)
    with pytest.raises(ValueError, match="physical units"):
        runner.train(settings(csv_path, tmp_path / "unused", units=""), check_data=True)
    assert not (tmp_path / "unused").exists()


def test_target_selection_applies_to_all_joint_horizons(tmp_path):
    csv_path = write_maps(tmp_path / "maps", "2026-01-01T00:00:00Z")
    targets = tmp_path / "targets.csv"
    pd.DataFrame({"timestamp": ["2026-01-01T01:00:00Z", "2026-01-01T01:30:00Z"]}).to_csv(targets, index=False)
    source = runner.dataset(settings(csv_path, tmp_path / "unused", train_targets=str(targets)), "train")
    assert len(source) == 1
    condition, future, origin, times = source[0]
    assert condition.shape == future.shape == (2, 16, 16)
    assert origin == "2026-01-01T00:30:00+00:00"
    assert times == ("2026-01-01T01:00:00+00:00", "2026-01-01T01:30:00+00:00")
