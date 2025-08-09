from typing import List, Tuple
import numpy as np
import cv2
from src.drone.core.types import CameraIntrinsics
from src.drone.config import CHB_SIZE, CRITERIA, CELL_MM

def coordination_points(board_size: Tuple[int, int] = CHB_SIZE, cell_mm: float = CELL_MM) -> np.ndarray:
    cop = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    cop[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    cop *= float(cell_mm)
    return cop

def calibrate_camera(deskpoints: List[np.ndarray], imgpoints: List[np.ndarray], image_size: Tuple[int, int]) -> CameraIntrinsics:
    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(deskpoints, imgpoints, image_size, None, None)
    if not ret:
        raise RuntimeError("Calibration failed")
    return CameraIntrinsics(K=K, dist=dist)

def find_chessboard_corners(image_bgr, board_size: Tuple[int,int] = CHB_SIZE):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not ret or corners.shape[0] != board_size[0]*board_size[1]:
        return False, None
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    return True, corners2

def undistort_image(image_bgr, intr: CameraIntrinsics):
    return cv2.undistort(image_bgr, intr.K, intr.dist, None, intr.K)
