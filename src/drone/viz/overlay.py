import cv2
import numpy as np

def draw_points(image, pts2d, radius=4, color=(255,0,0)):
    for u, v in np.asarray(pts2d):
        cv2.circle(image, (int(u), int(v)), radius, color, -1)
    return image

def draw_chessboard(image, board_size, corners, found: bool):
    return cv2.drawChessboardCorners(image.copy(), board_size, corners, found)

def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, axis_length=50):
    axis = np.float32([[axis_length,0,0], [0,axis_length,0], [0,0,axis_length]]).reshape(-1,3)
    imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, dist_coeffs)
    origin, x, y, z = tuple(np.int32(imgpts).reshape(-1,2)[0]), *np.int32(imgpts).reshape(-1,2)
    img = image.copy()
    cv2.line(img, origin, tuple(x), (0,0,255), 3)
    cv2.line(img, origin, tuple(y), (0,255,0), 3)
    cv2.line(img, origin, tuple(z), (255,0,0), 3)
    return img
