"""Real SANPO sequences -> locus.types.Frame.

Unlike AEA's semi-dense point-cloud depth (~18% of a frame), SANPO ships
a dense CREStereo depth map alongside every frame (~96% valid-pixel
coverage) and already-pinhole intrinsics -- no fisheye correction needed.

Access: the dataset is a public, ungated GCS bucket
(gs://gresearch/sanpo_dataset/v0, CC-BY 4.0), reachable over plain HTTPS
with no auth. This loader downloads per-frame files directly and caches
them under paths.sanpo_cache_dir; no bulk download is required.

Format notes (the dataset's own docs say npz; the actual files are not):
- RGB: per-frame PNG at <session>/camera_head/left/video_frames/NNNNNN.png,
  at the session's calibrated resolution.
- Depth: <session>/camera_head/left/depth_maps/NNNNNN.float16.gz -- gzip
  of a 4-byte header (DEPTH_HEADER_BYTES) followed by a flat float16
  array. Depth and RGB can ship at different resolutions
  (`depth_resolution` lets a session override the default), so this
  loader resizes depth up to the RGB resolution before handing out a
  Frame. Values are metres; non-finite or <=0 is invalid.
- Pose: <session>/<camera>/fixed_camera_poses.csv, one row per frame
  (index-aligned, not timestamped), columns tracking_state,pos_x,pos_y,
  pos_z,q_x,q_y,q_z,q_w. A row with tracking_state != READY carries a
  meaningless (0,0,0) position and is skipped for the nearest READY row.
- Intrinsics (fx,fy,cx,cy,image size,distortion) ship per-session in
  description.json's session_camera_details[i].left_camera_params. Both
  rig cameras report the same image size, so the entry is picked by
  `model` (CAMERA_MODEL_ZED_2I = chest, CAMERA_MODEL_ZED_M = head), cross
  -checked against each camera's own stereo baseline so a wrong mapping
  raises instead of silently producing wrong-valued intrinsics.

- **Rotation convention: raw SANPO poses are already OpenCV-camera, in a
  Z-up world -- no correction.** `R_world_cam` is the quaternion's
  rotation matrix as-is, with the camera's local axes taken to be
  OpenCV's (+X right, +Y down, +Z forward, what `Intrinsics` assumes).
  Settled by multi-view depth reprojection across all signed axis
  permutations on both the camera and world side: see docs/CLAUDE.md for
  the measurements and the investigation that led here (an earlier,
  wrong "camera-local -X is forward" correction scored far worse and was
  in fact a reflection). Consumers pass `SanpoSequence.world_down` into
  the memory graph / `locus.geometry.frames`.

Not yet done: only camera_head is used by `load_sequence`'s default;
camera_chest is read by the lower-level helpers but no Frame-producing
pipeline uses it yet.
"""

from __future__ import annotations

import csv
import gzip
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests

from locus.config import DEFAULT_CONFIG_PATH, load_config
from locus.geometry.intrinsics import Intrinsics
from locus.types import Frame, Pose

GCS_BASE = "https://storage.googleapis.com/gresearch/sanpo_dataset/v0/sanpo-real"
DEPTH_HEADER_BYTES = 4  # verified: skipping exactly this many bytes yields a clean width*height float16 array

# Both rig cameras report the same image size but different focal
# lengths, so picking the wrong session_camera_details entry fails
# silently while producing wrong 3D positions downstream. Model name
# distinguishes them; _verify_mount_baseline cross-checks against each
# camera's own stereo baseline so a wrong mapping raises instead.
CAMERA_MODEL_BY_MOUNT = {"camera_head": "CAMERA_MODEL_ZED_M", "camera_chest": "CAMERA_MODEL_ZED_2I"}
EXPECTED_BASELINE_MM = {"camera_head": 62.958637, "camera_chest": 119.918846}
_BASELINE_TOLERANCE_MM = 5.0

# Gravity in SANPO's world frame -- see the module docstring's rotation
# convention note for how this (and "no camera-axis correction") was measured.
SANPO_WORLD_DOWN = (0.0, 0.0, -1.0)


def _download(url: str, dest: Path) -> Path:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        dest.write_bytes(response.content)
    return dest


def _session_url(session_id: str, camera: str, *parts: str) -> str:
    return "/".join([GCS_BASE, session_id, camera, "left", *parts])


def _verify_mount_baseline(camera: str, baseline_mm: float) -> None:
    expected = EXPECTED_BASELINE_MM[camera]
    if abs(baseline_mm - expected) > _BASELINE_TOLERANCE_MM:
        raise ValueError(
            f"{camera}: stereo baseline {baseline_mm:.1f}mm doesn't match the expected "
            f"~{expected:.0f}mm for this mount -- the model-name-to-mount mapping "
            "(CAMERA_MODEL_BY_MOUNT) may not hold for this session; verify before trusting its intrinsics."
        )


def load_intrinsics(session_id: str, cache_dir: Path, camera: str = "camera_head") -> Intrinsics:
    import json

    desc_url = "/".join([GCS_BASE, session_id, "description.json"])
    desc_path = _download(desc_url, cache_dir / "description.json")
    cameras = json.loads(desc_path.read_text())["session_camera_details"]
    expected_model = CAMERA_MODEL_BY_MOUNT[camera]
    matches = [c for c in cameras if c["model"] == expected_model]
    if len(matches) != 1:
        raise ValueError(f"{session_id}: expected exactly one {expected_model} camera for {camera}, found {len(matches)}")
    cam = matches[0]
    _verify_mount_baseline(camera, cam["stereo_transform"]["coeff"][3])

    params = cam["left_camera_params"]
    if any(abs(d) > 1e-9 for d in params["distortion"]):
        raise ValueError(
            f"{session_id}/{camera}: left camera has nonzero distortion ({params['distortion']}) -- "
            "this loader assumes a pinhole camera, verified true for the default session only."
        )
    return Intrinsics(
        fx=params["fx"], fy=params["fy"], cx=params["cx"], cy=params["cy"],
        width=params["image_width"], height=params["image_height"],
    )


def _load_poses(session_id: str, cache_dir: Path, camera: str = "camera_head") -> list[tuple[bool, np.ndarray, np.ndarray]]:
    """Returns one (is_ready, position_xyz, quaternion_xyzw) per frame index."""
    pose_url = "/".join([GCS_BASE, session_id, camera, "fixed_camera_poses.csv"])
    pose_path = _download(pose_url, cache_dir / "fixed_camera_poses.csv")
    poses = []
    with open(pose_path) as f:
        for row in csv.DictReader(f):
            ready = row["tracking_state"].strip().endswith("READY")
            pos = np.array([float(row["pos_x"]), float(row["pos_y"]), float(row["pos_z"])])
            quat = np.array([float(row["q_x"]), float(row["q_y"]), float(row["q_z"]), float(row["q_w"])])
            poses.append((ready, pos, quat))
    return poses


@dataclass
class SanpoSequence:
    session_id: str
    camera: str
    cache_dir: Path
    intrinsics: Intrinsics
    fps: float
    n_frames: int
    depth_resolution: tuple[int, int]  # (height, width), as actually stored
    _poses: list[tuple[bool, np.ndarray, np.ndarray]]
    world_down: tuple[float, float, float] = SANPO_WORLD_DOWN

    def _nearest_ready_pose(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._poses)
        for offset in range(n):
            for idx in (frame_index - offset, frame_index + offset):
                if 0 <= idx < n and self._poses[idx][0]:
                    return self._poses[idx][1], self._poses[idx][2]
        raise ValueError(f"{self.session_id}: no READY pose found anywhere in {n} frames")

    def prefetch(self, stride: int = 1, max_frames: int | None = None, start_frame: int = 0, workers: int = 16) -> None:
        """Downloads every RGB+depth file `iter_frames` would need, in
        parallel. Each is an independent HTTPS GET against a public bucket
        (no shared state, no rate-limit auth to worry about) and `_download`
        is already idempotent (skips anything already cached), so a thread
        pool is a straightforward, large win over iter_frames' one-at-a-time
        downloads -- sequential, this was the actual bottleneck processing a
        full ~540-frame session (minutes of pure network wait with zero
        CPU/GPU use), not SAM 3.1 or anything else in the pipeline."""
        from concurrent.futures import ThreadPoolExecutor

        end = self.n_frames if max_frames is None else min(self.n_frames, start_frame + max_frames * stride)
        urls_and_dests = []
        for i in range(start_frame, end, stride):
            name = f"{i:06d}"
            urls_and_dests.append((
                _session_url(self.session_id, self.camera, "video_frames", f"{name}.png"),
                self.cache_dir / "video_frames" / f"{name}.png",
            ))
            urls_and_dests.append((
                _session_url(self.session_id, self.camera, "depth_maps", f"{name}.float16.gz"),
                self.cache_dir / "depth_maps" / f"{name}.float16.gz",
            ))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda pair: _download(*pair), urls_and_dests))

    def iter_frames(self, stride: int = 1, max_frames: int | None = None, start_frame: int = 0) -> Iterator[Frame]:
        from scipy.spatial.transform import Rotation

        end = self.n_frames if max_frames is None else min(self.n_frames, start_frame + max_frames * stride)
        depth_h, depth_w = self.depth_resolution

        for i in range(start_frame, end, stride):
            name = f"{i:06d}"
            rgb_path = _download(
                _session_url(self.session_id, self.camera, "video_frames", f"{name}.png"),
                self.cache_dir / "video_frames" / f"{name}.png",
            )
            depth_path = _download(
                _session_url(self.session_id, self.camera, "depth_maps", f"{name}.float16.gz"),
                self.cache_dir / "depth_maps" / f"{name}.float16.gz",
            )

            rgb = cv2.cvtColor(cv2.imread(str(rgb_path)), cv2.COLOR_BGR2RGB)

            with gzip.open(depth_path, "rb") as f:
                raw = f.read()[DEPTH_HEADER_BYTES:]
            depth_small = np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(depth_h, depth_w)
            depth = cv2.resize(
                depth_small, (self.intrinsics.width, self.intrinsics.height), interpolation=cv2.INTER_NEAREST
            )
            depth[~np.isfinite(depth) | (depth <= 0)] = 0.0

            p_world, quat_xyzw = self._nearest_ready_pose(i)
            R_world_cam = Rotation.from_quat(quat_xyzw).as_matrix()
            t = i / self.fps
            pose = Pose(timestamp=t, R_world_cam=R_world_cam, t_world_cam=p_world)
            yield Frame(timestamp=t, rgb=rgb, depth=depth, pose=pose)


def load_sequence(
    session_id: str,
    config: dict | None = None,
    camera: str = "camera_head",
    fps: float = 15.0,
    depth_resolution: tuple[int, int] = (720, 1280),
) -> SanpoSequence:
    config = config or load_config(DEFAULT_CONFIG_PATH)
    cache_root = Path(config["paths"].get("sanpo_cache_dir", "/tmp/locus_sanpo_cache"))
    cache_dir = cache_root / session_id / camera

    intrinsics = load_intrinsics(session_id, cache_dir, camera=camera)
    poses = _load_poses(session_id, cache_dir, camera=camera)

    return SanpoSequence(
        session_id=session_id,
        camera=camera,
        cache_dir=cache_dir,
        intrinsics=intrinsics,
        fps=fps,
        n_frames=len(poses),
        depth_resolution=depth_resolution,
        _poses=poses,
    )
