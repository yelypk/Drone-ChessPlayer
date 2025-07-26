import cv2
import numpy as np 
import glob
import os 

CHB_SIZE = (8, 6)
CRITERIES = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def coordination_points(board_size):
    cop = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    cop[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    cop = cop * 11.5
    return cop

def collections(image_folder, board_size):
    imgpoints = []
    deskpoints = []
    image_size = []

    cop = coordination_points(board_size)
    images = glob.glob(os.path.join(image_folder, '*.jpg'))

    for fname in images:
        print(f"Processed: {fname}")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        ret, corners = cv2.findChessboardCorners(gray, board_size)
        image_size = gray.shape[::-1]

        if ret:
            print("Founded!")
            deskpoints.append(cop)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIES)
            imgpoints.append(corners2)

            img_drawn = cv2.drawChessboardCorners(img, board_size, corners2, ret)
            cv2.imshow('Founded corners', img_drawn)
        else:
            print("NO CORNERS")
            cv2.imshow("Image without corners", img)

        cv2.waitKey(0)

    cv2.destroyAllWindows()
    return deskpoints, imgpoints, image_size

def calibrate_camera(deskpoints, imgpoints, image_size): 
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(deskpoints, imgpoints, image_size, None, None)
    return mtx, dist, rvecs, tvecs

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst

def estimate_pose_from_chessboard(img, board_size, mtx, dist):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    ret, corners = cv2.findChessboardCorners(gray, board_size)
    if not ret:
        print("Chessboard not found in the image for pose estimation.")
        return None, None, None

    objp = coordination_points(board_size)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIES)
    success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

    print("\nPose estimation:")
    print("Rotation vector:\n", rvec)
    print("Translation vector:\n", tvec)

    distance = np.linalg.norm(tvec)
    print(f"Відстань до об'єкта: {distance} мм")

    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print("Rotation matrix:\n", rotation_matrix)

    return rvec, tvec, rotation_matrix

def project_known_3d_point(img, point_3d, mtx, dist):

    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)

    image_points, _ = cv2.projectPoints(point_3d, rvec, tvec, mtx, dist)
    u, v = image_points[0, 0]
    u, v = int(round(u)), int(round(v))



    cv2.circle(img, (u, v), 7, (0, 0, 255), -1)
    cv2.imshow("Projected point", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def pixel_to_normalized_ray(u, v, mtx, dist):
    pixel = np.array([[[u, v]]], dtype=np.float32)
    norm = cv2.undistortPoints(pixel, mtx, dist)
    x = norm[0, 0, 0]
    y = norm[0, 0, 1]
    point_3d = np.array([x, y, 1.0])
    print(f"3D напрямок (Z=1): {point_3d}")
    return point_3d

def main():
    image_folder = 'frames' 
    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    example_image = images[0]
    img = cv2.imread(example_image)

    deskpoints, imgpoints, image_size = collections(image_folder, CHB_SIZE)
    mtx, dist, rvecs, tvecs = calibrate_camera(deskpoints, imgpoints, image_size)

    print("Camera matrix:\n", mtx)
    matrixreverse = np.linalg.inv(mtx)
    print("Reversed camera matrix:\n", matrixreverse)
    print("Distortion coefficients:\n", dist)

    np.savez('camera_calibration.npz', mtx=mtx, dist=dist)

    result = undistort_image(img, mtx, dist)
    cv2.imwrite('calibrated_result.jpg', result)
    print("Saved undistorted image as calibrated_result.jpg")

    rvec, tvec, rotation_matrix = estimate_pose_from_chessboard(img, CHB_SIZE, mtx, dist)

    pt3d = np.array([[0, 0, 1830]], dtype=np.float32)
    project_known_3d_point(img, pt3d, mtx, dist)

    u, v = 77, 100
    pixel_to_normalized_ray(u, v, mtx, dist)

if __name__ == "__main__":
    main()