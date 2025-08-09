import numpy as np
import cv2
from src.drone.core.types import CameraIntrinsics

def project_points(points_3d: np.ndarray, intr: CameraIntrinsics, rvec, tvec) -> np.ndarray:
    img_pts, _ = cv2.projectPoints(points_3d.reshape(-1,1,3), rvec, tvec, intr.K, intr.dist)
    return img_pts.reshape(-1, 2)

def pixel_to_normalized_ray(u: float, v: float, intr: CameraIntrinsics) -> np.ndarray:
    K_inv = np.linalg.inv(intr.K)
    pix = np.array([u, v, 1.0], dtype=np.float32)
    ray = K_inv @ pix
    ray = ray / np.linalg.norm(ray)
    return ray
