"""Pinhole camera intrinsics: K, its inverse, and pixel<->ray conversions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @property
    def K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

    @property
    def K_inv(self) -> np.ndarray:
        return np.linalg.inv(self.K)

    def project(self, p_cam: np.ndarray) -> np.ndarray:
        """3D point in camera frame -> pixel (u, v). Caller checks p_cam[2] > 0."""
        x, y, z = p_cam
        u = self.fx * x / z + self.cx
        v = self.fy * y / z + self.cy
        return np.array([u, v])

    def backproject(self, pixel: np.ndarray, depth: float) -> np.ndarray:
        """Pixel (u, v) + depth -> 3D point in camera frame."""
        u, v = pixel
        x = (u - self.cx) * depth / self.fx
        y = (v - self.cy) * depth / self.fy
        return np.array([x, y, depth])

    def in_frustum(self, pixel: np.ndarray) -> bool:
        u, v = pixel
        return 0 <= u < self.width and 0 <= v < self.height
