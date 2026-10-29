"""Integration test against real AEA data. Skips cleanly if the dataset
isn't present on this machine -- the dataset lives outside the repo and
isn't guaranteed on every machine this runs on (e.g. a future second
training box), unlike the synthetic unit-test harness."""

import numpy as np
import pytest

from locus.config import DEFAULT_CONFIG_PATH, load_config

config = load_config(DEFAULT_CONFIG_PATH)
_dataset_dir = config["paths"]["dataset_dir"]

pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(_dataset_dir).exists(),
    reason=f"AEA dataset not present at {_dataset_dir} on this machine",
)


def test_loads_real_sequence_with_pinhole_frames_and_moving_pose():
    from locus.data.aea_loader import load_sequence

    sequence_id = config["aea_sequences"][0]
    seq = load_sequence(sequence_id, config=config)

    assert seq.intrinsics.width == config["image"]["width"]
    assert seq.intrinsics.height == config["image"]["height"]

    frames = list(seq.iter_frames(stride=20, max_frames=10))
    assert len(frames) == 10

    positions = []
    for frame in frames:
        assert frame.rgb.shape == (config["image"]["height"], config["image"]["width"], 3)
        assert frame.rgb.dtype == np.uint8
        assert frame.depth is None, "AEA depth isn't wired up yet (see aea_loader module docstring)"
        assert frame.pose is not None
        positions.append(frame.pose.t_world_cam.copy())

    # The wearer should actually be moving across these frames, not frozen --
    # otherwise pose parsing/lookup silently degenerated to one fixed row.
    total_motion = sum(
        float(np.linalg.norm(positions[i + 1] - positions[i])) for i in range(len(positions) - 1)
    )
    assert total_motion > 0.01
