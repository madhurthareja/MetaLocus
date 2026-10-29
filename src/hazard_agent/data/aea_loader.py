"""Loads one AEA sequence's downloaded assets.

Expects the layout produced by scripts' AEA downloader:
<dataset_dir>/<sequence_name>/{video_main_rgb, mps_slam_trajectories,
mps_slam_calibration, mps_slam_points, mps_eye_gaze, annotations}.

Pose is read from the open-loop trajectory, not the post-processed
closed-loop one -- see docs/PROJECT_PLAN.md section 3.
"""

from pathlib import Path


class AeaSequence:
    def __init__(self, sequence_dir: str):
        self.sequence_dir = Path(sequence_dir)

    def rgb_frames(self):
        raise NotImplementedError

    def open_loop_trajectory(self):
        raise NotImplementedError

    def point_cloud(self):
        raise NotImplementedError

    def calibration(self):
        raise NotImplementedError
