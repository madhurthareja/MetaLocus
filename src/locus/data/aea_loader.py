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
  not a dense per-pixel depth map. `load_point_cloud` + `project_sparse_depth`
  turn it into a per-frame sparse depth image: the static world-frame point
  cloud is projected into each frame's camera via its real pose, keeping
  the nearest point per pixel (a z-buffer, same occlusion handling as
  locus.data.synthetic's farthest-first painting). This is naturally sparse
  (a few thousand valid pixels out of ~300K, not a dense map) -- which is
  exactly the shape lift()/lift_from_box() were written for already (they
  already gate on a minimum count of valid depth pixels in a region).
  The point cloud's own observations (semidense_observations.csv.gz) are
  recorded against the SLAM cameras, not the RGB camera, so they can't be
  used to occlusion-test against the RGB view directly; the per-pixel
  z-buffer is the practical substitute.

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


def load_point_cloud(
    sequence_id: str, config: dict | None = None, max_dist_std: float = 0.15
) -> np.ndarray:
    """World-frame (px, py, pz) for every semi-dense SLAM point passing a
    distance-uncertainty gate. `max_dist_std` follows MPS's own documented
    filtering convention (points with poorly-constrained distance aren't
    useful depth samples)."""
    from projectaria_tools.core import mps

    config = config or load_config(DEFAULT_CONFIG_PATH)
    dataset_dir = Path(config["paths"]["dataset_dir"])
    manifest = load_manifest(config["paths"]["dataset_manifest"])

    points_zip = _manifest_file_path(dataset_dir, sequence_id, manifest, "mps_slam_points")
    points_csv = _ensure_extracted(points_zip, "semidense_points.csv.gz")
    points = mps.read_global_point_cloud(str(points_csv))

    positions = np.array([p.position_world for p in points if p.distance_std <= max_dist_std])
    return positions


def project_sparse_depth(
    intrinsics: Intrinsics, pose: Pose, points_world: np.ndarray, max_range_m: float = 15.0
) -> np.ndarray:
    """Projects a static world-frame point cloud into one camera pose,
    keeping the nearest point per pixel (a z-buffer -- the vectorized
    equivalent of locus.data.synthetic's farthest-first painting)."""
    p_cam = (pose.R_world_cam.T @ (points_world - pose.t_world_cam).T).T
    depth_vals = p_cam[:, 2]
    in_front = (depth_vals > 0) & (depth_vals <= max_range_m)
    p_cam, depth_vals = p_cam[in_front], depth_vals[in_front]

    u = intrinsics.fx * p_cam[:, 0] / depth_vals + intrinsics.cx
    v = intrinsics.fy * p_cam[:, 1] / depth_vals + intrinsics.cy
    in_bounds = (u >= 0) & (u < intrinsics.width) & (v >= 0) & (v < intrinsics.height)
    u, v, depth_vals = u[in_bounds], v[in_bounds], depth_vals[in_bounds]

    depth = np.zeros((intrinsics.height, intrinsics.width), dtype=float)
    order = np.argsort(-depth_vals)  # farthest first, so the nearest overwrites last
    u_px = u[order].astype(int)
    v_px = v[order].astype(int)
    depth[v_px, u_px] = depth_vals[order]
    return depth


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

    def iter_frames(
        self,
        stride: int = 1,
        max_frames: int | None = None,
        point_cloud_world: np.ndarray | None = None,
        depth_max_range_m: float = 15.0,
        start_frame: int = 0,
    ) -> Iterator[Frame]:
        """Yields undistorted-RGB, real-pose Frames in order, starting at
        `start_frame` (e.g. to reach a window with real wearer motion rather
        than always the clip's opening seconds). Frame.depth is None unless
        `point_cloud_world` (from `load_point_cloud`) is given, in which case
        it's the sparse point-cloud projection for that pose."""
        from projectaria_tools.core import calibration as aria_calibration

        cap = cv2.VideoCapture(str(self.video_path))
        try:
            if start_frame > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_index = start_frame
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
                    depth = None
                    if point_cloud_world is not None:
                        depth = project_sparse_depth(
                            self.intrinsics, pose, point_cloud_world, max_range_m=depth_max_range_m
                        )
                    yield Frame(timestamp=t, rgb=undistorted, depth=depth, pose=pose)
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
