"""Mask -> 3D position via AEA's semi-dense point cloud.

For each SAM mask, project the sequence's mps_slam_points semi-dense
point cloud into the undistorted frame, take the median depth of points
that fall inside the mask, and lift to a 3D position. Positions are then
transformed into world frame using the sequence's open-loop 6-DoF pose
(mps_slam_trajectories) -- open-loop because it mirrors what an on-device
system would actually have, not the post-processed closed-loop trajectory.

A monocular depth model is only a fallback for masks with too few points
inside them; no such fallback is wired up yet.
"""


def lift_mask_to_3d(mask, point_cloud, pose, intrinsics):
    """Return a 3D world-frame position for one mask at one frame."""
    raise NotImplementedError
