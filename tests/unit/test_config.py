import yaml

from src import config
from src.training.train import candidates


def test_yaml_values_feed_python_constants():
    raw = yaml.safe_load(config.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert config.RANDOM_STATE == raw["data"]["random_state"]
    assert config.TEST_SIZE == raw["data"]["test_size"]
    assert config.FALSE_NEGATIVE_COST == raw["decision_policy"]["false_negative_cost"]
    assert config.RISK_LOW_UPPER < config.RISK_HIGH_LOWER


def test_alternative_config_file_via_env(tmp_path, monkeypatch):
    raw = yaml.safe_load(config.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    raw["data"]["test_size"] = 0.2
    custom = tmp_path / "experiment.yaml"
    custom.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("CHURNOPS_CONFIG", str(custom))
    assert config.load_config()["data"]["test_size"] == 0.2


def test_candidates_are_built_from_config():
    cfg = config.TRAINING_CONFIG
    full = candidates()
    grid = cfg["lightgbm_grid"]
    assert len(full) == 2 + 2 * len(grid)  # logreg, rf, then each lgbm grid point +/- calibration
    by_name = {c.name: c for c in full}
    assert by_name["logreg"].params == cfg["logreg"]
    assert by_name["lightgbm_v1"].params == grid[0]
    assert by_name["lightgbm_v1_calibrated"].params["calibration_cv"] == cfg["calibration"]["cv"]


def test_quick_mode_and_overrides():
    cfg = {**config.TRAINING_CONFIG, "lightgbm_grid": [{"n_estimators": 5, "num_leaves": 3}]}
    quick = candidates(quick=True, cfg=cfg)
    assert [c.name for c in quick] == ["logreg", "random_forest", "lightgbm_v1", "lightgbm_v1_calibrated"]
    rf = next(c for c in quick if c.name == "random_forest")
    assert rf.params["n_estimators"] == cfg["quick"]["random_forest_n_estimators"]
    lgbm = next(c for c in quick if c.name == "lightgbm_v1").make()
    assert lgbm.named_steps["model"].get_params()["num_leaves"] == 3
