"""Fisheye undistortion and pinhole intrinsics via projectaria_tools.

AEA's main RGB stream is fisheye; this module wraps projectaria_tools'
calibration data (from each sequence's mps_slam_calibration.zip) to
undistort frames and expose the pinhole intrinsics K used for 3D lifting
in geometry/depth_lifting.py.
"""


class AriaCalibration:
    def __init__(self, calibration_path: str):
        self.calibration_path = calibration_path

    def undistort(self, frame):
        raise NotImplementedError

    def intrinsics(self):
        """Return pinhole intrinsics K derived from the fisheye calibration."""
        raise NotImplementedError
