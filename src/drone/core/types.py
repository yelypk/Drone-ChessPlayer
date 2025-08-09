from dataclasses import dataclass
import numpy as np

@dataclass
class CameraIntrinsics:
    K: np.ndarray      # 3x3
    dist: np.ndarray   # distortion coefficients

@dataclass
class Pose:
    rvec: np.ndarray   # 3x1
    tvec: np.ndarray   # 3x1
