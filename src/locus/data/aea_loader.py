"""Real AEA (Aria Everyday Activities) sequences -> locus.types.Frame.

This is the actual benchmark data source (locus.data.synthetic is a unit-
test harness, not a benchmark substitute). What a sequence gives us for
free, and what it doesn't:

- Pose: real 6-DoF device trajectory from MPS (open-loop, not closed-loop
  -- open-loop is the raw odometry estimate without the final global
  bundle adjustment pass, which is what a live/online system would
  actually have access to; closed-loop is a retrospective-only product).
- RGB: the preview_rgb.mp4 is raw fisheye (FisheyeRadTanThinPrism, aka
  Aria's FISHEYE624 model), not the pinhole image locus.geometry assumes
  everywhere else. This loader undistorts every frame to a virtual
  pinhole camera via projectaria_tools' own calibration math -- that's
  what makes an AEA frame's Pose/Intrinsics compatible with
  geometry.frames/lifting without changing either of them.
- Depth: AEA ships a semi-dense SLAM point cloud (mps_slam_points.zip),
  not a dense per-pixel depth map. Projecting it into a sparse depth
  image per frame is real future work, not done here -- Frame.depth is
  None for every AEA frame right now. That means lift()/lift_from_box()
  can't run on real AEA frames yet; this loader gets pose+RGB+intrinsics
  right first rather than quietly fabricating depth.

Calibration notes (verified against an actual frame, not assumed): the
camera-rgb calibration in online_calibration.jsonl is parameterized at a
2880x2880 native sensor resolution (the json carries no image size field
-- 2880 is Aria Gen1's documented RGB sensor resolution, and rescaling
from it with DeviceVersion.Gen1 round-trips correctly; DeviceVersion.Gen2
raises immediately, which further confirms this is a Gen1 recording, as
the whole AEA dataset is). Online calibration drifts slightly over a
session (thermal effects); this loader uses the first record in the
jsonl as a fixed-for-the-sequence approximation rather than the
time-varying series, which is accurate enough for a short clip.
"""

from __future__ import annotations

import json
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from locus.config import DEFAULT_CONFIG_PATH, load_config
from locus.geometry.intrinsics import Intrinsics
from locus.types import Frame, Pose

NATIVE_RGB_RESOLUTION = 2880  # Aria Gen1 RGB sensor; see module docstring


def load_manifest(manifest_path: str | Path) -> dict:
    with open(manifest_path) as f:
        return json.load(f)["sequences"]


def _manifest_file_path(dataset_dir: Path, sequence_id: str, manifest: dict, key: str) -> Path:
    filename = manifest[sequence_id][key]["filename"]
    return dataset_dir / sequence_id / filename


def _ensure_extracted(zip_path: Path, member: str) -> Path:
    """Extracts `member` from `zip_path` next to it (idempotent) and
    returns its path. AEA's MPS outputs ship zipped; the video does not."""
    extract_dir = zip_path.parent / "_extracted" / zip_path.stem
    out_path = extract_dir / member
    if not out_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extract(member, extract_dir)
    return out_path


def _load_rgb_calibration_native(sequence_dir: Path, manifest: dict, sequence_id: str):
    from projectaria_tools.core import calibration
    from projectaria_tools.core.sophus import SE3

    calib_zip = _manifest_file_path(sequence_dir.parent, sequence_id, manifest, "mps_slam_calibration")
    jsonl_path = _ensure_extracted(calib_zip, "online_calibration.jsonl")

    with open(jsonl_path) as f:
        first_record = json.loads(f.readline())

    rgb_entry = next(c for c in first_record["CameraCalibrations"] if c["Label"] == "camera-rgb")
    projection = rgb_entry["Projection"]
    quat = rgb_entry["T_Device_Camera"]["UnitQuaternion"]
    T_device_camera = SE3.from_quat_and_translation(
        quat[0], quat[1], rgb_entry["T_Device_Camera"]["Translation"]
    )

    return calibration.CameraCalibration(
        "camera-rgb",
        calibration.CameraModelType.FISHEYE624,
        np.array(projection["Params"]),
        T_device_camera,
        NATIVE_RGB_RESOLUTION,
        NATIVE_RGB_RESOLUTION,
        None,
        1.5,
        rgb_entry["SerialNumber"],
    )


def _build_undistortion_calibrations(
    sequence_dir: Path,
    manifest: dict,
    sequence_id: str,
    video_resolution: int,
    dst_width: int,
    dst_height: int,
    dst_focal_length: float,
):
    from projectaria_tools.core import calibration

    native_calib = _load_rgb_calibration_native(sequence_dir, manifest, sequence_id)
    src_calib = calibration.rescale_camera_calibration(
        native_calib, np.array([video_resolution, video_resolution]), calibration.DeviceVersion.Gen1
    )
    dst_calib = calibration.get_linear_camera_calibration(
        dst_width, dst_height, dst_focal_length, "camera-rgb-pinhole", src_calib.get_transform_device_camera()
    )
    return src_calib, dst_calib


@dataclass
class AeaSequence:
    sequence_id: str
    video_path: Path
    fps: float
    n_video_frames: int
    intrinsics: Intrinsics
    _poses: list  # mps.OpenLoopTrajectoryPose, sorted by timestamp
    _pose_timestamps_s: np.ndarray
    _T_device_camera: np.ndarray  # (4, 4)
    _src_calib: object  # rescaled fisheye calibration, for distort_by_calibration
    _dst_calib: object  # virtual pinhole calibration matching self.intrinsics

    def _nearest_pose(self, t: float):
        idx = int(np.searchsorted(self._pose_timestamps_s, t))
        idx = min(max(idx, 0), len(self._poses) - 1)
        return self._poses[idx]

    def iter_frames(self, stride: int = 1, max_frames: int | None = None) -> Iterator[Frame]:
        """Yields undistorted-RGB, real-pose, depth=None Frames in order."""
        from projectaria_tools.core import calibration as aria_calibration

        cap = cv2.VideoCapture(str(self.video_path))
        try:
            frame_index = 0
            yielded = 0
            while True:
                ok, bgr = cap.read()
                if not ok:
                    break
                if frame_index % stride == 0:
                    t = frame_index / self.fps
                    pose_row = self._nearest_pose(t)
                    T_world_device = pose_row.transform_odometry_device.to_matrix()
                    T_world_camera = T_world_device @ self._T_device_camera

                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    undistorted = aria_calibration.distort_by_calibration(rgb, self._dst_calib, self._src_calib)

                    pose = Pose(
                        timestamp=t,
                        R_world_cam=T_world_camera[:3, :3],
                        t_world_cam=T_world_camera[:3, 3],
                    )
                    yield Frame(timestamp=t, rgb=undistorted, depth=None, pose=pose)
                    yielded += 1
                    if max_frames is not None and yielded >= max_frames:
                        return
                frame_index += 1
        finally:
            cap.release()


def load_sequence(
    sequence_id: str,
    config: dict | None = None,
    video_resolution: int = 1408,
    dst_focal_length: float = 280.0,
) -> AeaSequence:
    from projectaria_tools.core import mps

    config = config or load_config(DEFAULT_CONFIG_PATH)
    dataset_dir = Path(config["paths"]["dataset_dir"])
    manifest = load_manifest(config["paths"]["dataset_manifest"])
    sequence_dir = dataset_dir / sequence_id

    video_path = _manifest_file_path(dataset_dir, sequence_id, manifest, "video_main_rgb")
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    traj_zip = _manifest_file_path(dataset_dir, sequence_id, manifest, "mps_slam_trajectories")
    traj_csv = _ensure_extracted(traj_zip, "open_loop_trajectory.csv")
    poses = mps.read_open_loop_trajectory(str(traj_csv))
    pose_timestamps_s = np.array([p.utc_timestamp.total_seconds() for p in poses])

    width, height = config["image"]["width"], config["image"]["height"]
    src_calib, dst_calib = _build_undistortion_calibrations(
        sequence_dir, manifest, sequence_id, video_resolution, width, height, dst_focal_length
    )
    intrinsics = Intrinsics(
        fx=float(dst_calib.get_focal_lengths()[0]),
        fy=float(dst_calib.get_focal_lengths()[1]),
        cx=float(dst_calib.get_principal_point()[0]),
        cy=float(dst_calib.get_principal_point()[1]),
        width=width,
        height=height,
    )

    return AeaSequence(
        sequence_id=sequence_id,
        video_path=video_path,
        fps=fps,
        n_video_frames=n_frames,
        intrinsics=intrinsics,
        _poses=poses,
        _pose_timestamps_s=pose_timestamps_s,
        _T_device_camera=src_calib.get_transform_device_camera().to_matrix(),
        _src_calib=src_calib,
        _dst_calib=dst_calib,
    )
