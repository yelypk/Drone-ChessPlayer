from typing import Tuple
import numpy as np
import cv2
from src.drone.core.types import CameraIntrinsics, Pose
from src.drone.core.calib import coordination_points
from src.drone.config import CHB_SIZE

def estimate_pose_from_chessboard(image_bgr, intr: CameraIntrinsics, board_size: Tuple[int,int] = CHB_SIZE):
    ok, corners = cv2.findChessboardCorners(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY), board_size, None)
    if not ok:
        return None
    objp = coordination_points(board_size)
    ok2, rvec, tvec = cv2.solvePnP(objp, corners, intr.K, intr.dist)
    if not ok2:
        return None
    return Pose(rvec=rvec, tvec=tvec)
