"""DW05 joins chunks with crossfade (the chunk takes effect on arrival) and holds the gripper for 4 slots; other families keep splice."""
from pathlib import Path
from types import SimpleNamespace

import pytest

import managed_rollout

YAM = Path(__file__).resolve().parents[1]


def test_only_dw05_fuses_by_default_and_committed_actions_always_splice():
    assert managed_rollout.CHUNK_FUSION_DEFAULTS == {"dw05": "crossfade"}
    assert managed_rollout.default_chunk_fusion("dw05", committed=False) == "crossfade"
    assert managed_rollout.default_chunk_fusion("dw05", committed=True) == "splice"
    for family in ("rho", "dreamzero", "fastwam", "aw0", "gr00t", "pi05", "longwam", "praxis1"):
        assert managed_rollout.default_chunk_fusion(family, committed=False) == "splice"


def test_only_dw05_holds_the_gripper():
    assert managed_rollout.HOLD_GRIPPER_ROWS == {"dw05": 4}


@pytest.mark.parametrize("family, requested, expected", [
    ("dw05", None, "crossfade"),
    ("dw05", "rail", "rail"),       # an explicit flag always wins
    ("dw05", "splice", "splice"),
    ("fastwam", None, "splice"),
])
def test_the_rollout_resolves_the_chunk_fusion_default(family, requested, expected, tmp_path, monkeypatch):
    class SettingsSaved(Exception):
        pass

    saved = {}

    def save_settings(path, settings):
        saved.update(settings)
        raise SettingsSaved

    monkeypatch.chdir(YAM.parents[1])
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(managed_rollout, "record_settings", save_settings)
    args = SimpleNamespace(policy_family=family, control_hz=None, prefetch_lead_steps=None, plan_rows=None,
                           chunk_fusion=requested, fusion_rows=4, shadow=False,
                           instruction="go", rrd=tmp_path / "run.rrd", deployment="d", seconds=30, noise_seed=None)
    identity = dict(state_dim=7, action_dim=7, action_horizon=32, camera_mapping={"top": "a", "left": "b"})
    with pytest.raises(SettingsSaved):
        managed_rollout.rollout_joint_policy(None, identity, args)
    assert saved["chunk_fusion"] == expected
