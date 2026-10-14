"""Gymnasium API, registration and scoring."""

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import gym_taskboard  # noqa: F401  (registers the environments)
from gym_taskboard import scoring


def test_registered_ids():
    ids = [i for i in gym.registry if i.startswith("TaskBoard1")]
    assert "TaskBoard1-Assembly-v0" in ids and "TaskBoard1-Nuts-Disassembly-v0" in ids
    assert len(ids) == 10


@pytest.mark.parametrize("obs_type", ["state", "image"])
def test_check_env(obs_type):
    env = gym.make("TaskBoard1-Pegs-Assembly-v0", obs_type=obs_type, image_size=(64, 64))
    check_env(env.unwrapped, skip_render_check=True)
    obs, info = env.reset(seed=0)
    assert obs["joints"].shape == (18,)
    if obs_type == "image":
        assert {"hand", "left", "right"} <= obs.keys() and obs["hand"].shape == (64, 64, 3)
    else:
        assert obs["objects"].shape == (7 * (8 + 2),)
    assert info["max_score"] == 8 * 6
    env.close()


def test_disassembly_start_and_step():
    env = gym.make("TaskBoard1-Disassembly-v0")
    _, info = env.reset(seed=0)
    assert info["score"] == 0 and info["max_score"] == 20 * 3
    _, r, term, trunc, _ = env.step(env.action_space.sample())
    assert r == 0 and not term
    env.close()


def test_single_part_selection():
    env = gym.make("TaskBoard1-Assembly-v0", parts=["nut_m8", "usb"])
    _, info = env.reset(seed=0)
    assert set(info["parts"]) == {"nut_m8", "usb"}
    env.close()


def test_success_probability():
    # 30 / 30 successes at 95 % confidence -> P_S ~ 0.905 (NIST test-method example)
    assert scoring.success_probability(30, 30) == pytest.approx(0.905, abs=1e-3)
    assert scoring.success_probability(0, 30) == 0.0
