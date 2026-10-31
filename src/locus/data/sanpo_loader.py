"""Real SANPO sequences -> locus.types.Frame.

Why this exists alongside aea_loader.py, not instead of it: SANPO fixes
AEA's biggest real-data gap -- dense depth. AEA's semi-dense point-cloud
projection covers ~18% of a frame; SANPO ships a dense CREStereo depth
map alongside every frame (measured directly: ~96% valid-pixel coverage
on the session this loader defaults to). It also ships already-pinhole
intrinsics (zero distortion coefficients in that session's
description.json) -- no fisheye correction needed, unlike AEA's Aria RGB
camera.

Access: the whole dataset is a public, ungated Google Cloud Storage
bucket (gs://gresearch/sanpo_dataset/v0, CC-BY 4.0) -- verified by
listing and downloading real files over plain HTTPS
(storage.googleapis.com/gresearch/...), no gcloud CLI or GCP auth needed.
This loader downloads per-frame files directly and caches them locally
under paths.sanpo_cache_dir; it does not require a bulk download first.

Format notes (verified against real downloaded files, not assumed from
the dataset's own docs, which say npz -- the actual files are not npz):
- RGB: per-frame PNG at <session>/camera_head/left/video_frames/NNNNNN.png,
  at the session's calibrated resolution (2208x1242 for the default
  session).
- Depth: <session>/camera_head/left/depth_maps/NNNNNN.float16.gz -- gzip
  of a 4-byte prefix (purpose unconfirmed -- not a width/height pair, that
  guess was checked and ruled out) followed by a flat float16 array.
  Verified at 1280x720 for the default session: a *different* resolution
  than its RGB frame, so this loader resizes depth up to the RGB
  resolution before handing out a Frame. Values are metres; non-finite or
  <=0 is invalid. The 1280x720 depth resolution is a per-session ZED
  setting, not assumed universal -- `depth_resolution` is a constructor
  arg specifically so a different session can override it.
- Pose: <session>/<camera>/fixed_camera_poses.csv, one row per frame
  (index-aligned with video_frames/depth_maps, not timestamped), columns
  tracking_state,pos_x,pos_y,pos_z,q_x,q_y,q_z,q_w. Rows with
  tracking_state != READY carry a meaningless (0,0,0) position and are
  skipped in favor of the nearest READY row, mirroring how
  aea_loader.py's AeaSequence does nearest-pose lookup.
- Intrinsics (fx,fy,cx,cy,image size,distortion) ship per-session in
  description.json's session_camera_details[i].left_camera_params. Both
  rig cameras report the SAME image_width/image_height (2208x1242,
  "HD2K" for both), so the entry can't be picked by shape -- it's picked
  by `model` (CAMERA_MODEL_ZED_2I = chest, CAMERA_MODEL_ZED_M = head),
  cross-checked against each camera's own stereo_transform baseline
  (~120mm chest, ~63mm head, matching the documented rig spec) so a wrong
  mapping raises instead of silently producing right-shaped, wrong-valued
  intrinsics. Caught and fixed after an earlier version of this loader
  used session_camera_details[0] unconditionally for whichever camera was
  requested -- it happened to be right for camera_head's *first* use only
  by coincidence of list order, and silently wrong for camera_chest.

- **Rotation convention -- the one correctness issue here that visual
  checks can't catch, and only partly resolved.** The pose's rotation is
  camera-to-world, but nothing in SANPO's docs, README, CHANGELOG or repo
  code states the camera's local "forward" axis. Assuming it matches
  `locus.geometry.intrinsics.Intrinsics`' OpenCV convention (+Z forward,
  +X right, +Y down) -- the same assumption aea_loader.py's calibration
  correctly uses for Aria -- turned out to be wrong for SANPO, caught only
  by cross-validating the head camera's velocity-derived heading against
  its own quaternion-derived heading (a check with nothing to do with
  RGB/depth, so a "does the image look right" visual check can't catch
  it). Systematically testing all six signed local axes as candidate
  "forward" directions against the head camera's own smoothed-velocity
  heading found a clean, small, consistent residual (median ~22 degrees,
  plausible gait/tracking noise) only for local -X; every other candidate
  gave essentially random-looking errors around 100+ degrees -- this part
  is solid.

  The up/right axes are a genuinely open question, not just an unverified
  algebraic afterthought. The repo names the quaternion feature
  `camera_quaternions_right_handed_y_up`, which reads as "translation Y is
  vertical" -- but checked directly against this loader's own
  fixed_camera_poses.csv, Y is *not* the bounded axis: on the default
  session (tagged elevation-flat), the whole-session translation range is
  X=2.8m, Y=14.0m, Z=1.1m. Z stays bounded like a vertical gait-bob axis;
  Y behaves like the dominant walking/horizontal axis -- the opposite of
  what the feature name implies. Diffing `fixed_camera_poses.csv` against
  the dataset's own *unfixed* `camera_poses.csv` for this same session
  rules out a simple constant axis permutation explaining this (the two
  trajectories diverge progressively over the session, not by a fixed
  offset), so "fixed" looks like a genuinely re-estimated trajectory, not
  a relabeling of "unfixed" -- meaning the "y_up" name may describe a
  different internal representation than this literal CSV column.
  `_R_SANPO_LOCAL_FROM_OPENCV` below assumes up=Z (the axis that's
  actually bounded in the file this loader reads), not up=Y (the literal
  reading of the feature name). Both choices give *identical* results on
  every heading check run so far, because a heading check only exercises
  the forward column of the correction matrix -- it cannot distinguish
  between them. This matters for lifted 3D points' lateral/vertical
  position; it does not affect depth-along-ray or heading, which is why
  the cross-camera validation below still means what it says.

  Applying the (forward-confirmed, up/right-unconfirmed) correction is
  what let a real cross-camera validation succeed on the part it can
  check: the head camera's velocity-derived body-frame heading agrees
  with the *literal* chest-camera-measured heading at a median of ~22
  degrees (occasional larger excursions correspond to turns, where a
  velocity-smoothed heading estimate lags real-time orientation by
  construction) -- the first validation of locus.geometry.frames' core
  approximation against a directly recorded body-frame reference, rather
  than just "it seems like a reasonable thing to assume." Resolving the
  up/right ambiguity with certainty needs either the dataset maintainers
  directly (sanpo_dataset@google.com, per the repo README) or a
  validation that actually exercises lateral position, not just heading.

Not yet done: only camera_head is used by `load_sequence`'s default;
camera_chest is read by the lower-level helpers (as used in the
validation above) but no Frame-producing pipeline uses it yet.
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

# Both rig cameras report the SAME image_width/image_height (both run in
# "HD2K" mode) but different focal lengths -- so picking the wrong
# session_camera_details entry fails silently (shapes still match) while
# quietly producing wrong 3D positions downstream. Model name is what
# actually distinguishes them; confirmed against the physical rig spec
# via each camera's own stereo_transform baseline (chest ZED-2i ~120mm,
# head ZED Mini ~63mm -- see _verify_mount_baseline).
CAMERA_MODEL_BY_MOUNT = {"camera_head": "CAMERA_MODEL_ZED_M", "camera_chest": "CAMERA_MODEL_ZED_2I"}
EXPECTED_BASELINE_MM = {"camera_head": 62.958637, "camera_chest": 119.918846}
_BASELINE_TOLERANCE_MM = 5.0

# Maps an OpenCV-convention camera-local vector (+X right, +Y down, +Z
# forward -- what Intrinsics.backproject/project assume) into SANPO's raw
# quaternion's local axes.
#
# Forward (-X): confirmed empirically, robustly -- see the module
# docstring's rotation-convention note.
#
# Up/right: UNRESOLVED, not independently confirmed. The repo names the
# quaternion feature `camera_quaternions_right_handed_y_up`, which reads
# as "translation Y is vertical" -- but checked directly against this
# loader's own fixed_camera_poses.csv data, Y is NOT the bounded axis: on
# the default session (tagged elevation-flat), translation ranges over the
# whole session are X=2.8m, Y=14.0m, Z=1.1m. Z is the one that stays
# bounded like a vertical gait-bob axis; Y behaves like the dominant
# walking/horizontal axis. Diffing fixed_camera_poses.csv against the
# dataset's own unfixed camera_poses.csv for this session rules out a
# simple constant axis permutation explaining this (the two diverge more
# and more over the session, not by a fixed offset), so "fixed" looks like
# a genuine re-estimated trajectory, not just a relabeling of "unfixed" --
# meaning the "y_up" name may describe a different internal representation
# than this literal CSV column. Below assumes up=Z (the empirically
# bounded axis in the actual file this loader reads), not up=Y (the
# literal reading of the feature name) -- chosen because it matches this
# loader's own data, but genuinely unconfirmed, and it only affects
# lateral/vertical placement of lifted 3D points, not depth-along-ray or
# the forward/heading direction (independently confirmed, see above, and
# unaffected by this choice: both the Y-up and Z-up versions of this
# matrix give identical results on every forward-heading check run so
# far, since only the up/right columns differ between them). Resolving
# this with certainty needs either the dataset maintainers directly
# (sanpo_dataset@google.com, per the repo README) or a validation that
# actually exercises lateral position, not just heading.
_R_SANPO_LOCAL_FROM_OPENCV = np.array([
    [0.0, 0.0, -1.0],
    [1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


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

    def _nearest_ready_pose(self, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
        n = len(self._poses)
        for offset in range(n):
            for idx in (frame_index - offset, frame_index + offset):
                if 0 <= idx < n and self._poses[idx][0]:
                    return self._poses[idx][1], self._poses[idx][2]
        raise ValueError(f"{self.session_id}: no READY pose found anywhere in {n} frames")

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
            R_world_sanpolocal = Rotation.from_quat(quat_xyzw).as_matrix()
            R_world_cam = R_world_sanpolocal @ _R_SANPO_LOCAL_FROM_OPENCV
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
