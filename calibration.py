import cv2
import numpy as np 
import glob
import os 

CHB_SIZE = (8, 6)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def coordination_points(board_size):
    cop = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    cop[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    cop = cop * 11.5
    return cop

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
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
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

def print_camera_info(mtx, dist):
    print("\n" + "="*40)
    print("CAMERA MATRIX:\n", mtx)
    print("REVERSED MATRIX:\n", np.linalg.inv(mtx))
    print("DISTORTION COEFFICIENTS:\n", dist)
    print("="*40 + "\n")

def main():
    image_folder = 'frames' 
    valid_folder = 'valid'
    invalid_folder = 'invalid'
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(invalid_folder, exist_ok=True)

    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    if not images:
        print("No images in the folder")
        return

    print(f"\nFounded {len(images)} of images. Searching for corners...\n")

    imgpoints = []
    deskpoints = []
    image_size = []
    cop = coordination_points(CHB_SIZE)

    for fname in images:
        print(f"Processed: {fname}")
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(gray, CHB_SIZE)

        if ret and corners.shape[0] == CHB_SIZE[0] * CHB_SIZE[1]:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)
            deskpoints.append(cop)
            if not image_size:
                image_size = gray.shape[::-1]
            img_drawn = cv2.drawChessboardCorners(img, CHB_SIZE, corners2, ret)
            cv2.imwrite(os.path.join(valid_folder, os.path.basename(fname)), img_drawn)
            print("Courners have been founded and saved in valid/")
        else:
            cv2.imwrite(os.path.join(invalid_folder, os.path.basename(fname)), img)
            print("No corners. Have been saved in invalid/")

    print(f"\nGood process {len(imgpoints)} from {len(images)}")

    if len(imgpoints) < 3:
        print("Not enought images for calibration")
        return

    if not image_size:
        print("image_size пуст — ни одно изображение не дало размер!")
        return

    try:
        mtx, dist, rvecs, tvecs = calibrate_camera(deskpoints, imgpoints, image_size)
        print_camera_info(mtx, dist)
    except Exception as e:
        print("Ошибка при калибровке:", e)
        return

    example_image = images[0]
    img = cv2.imread(example_image)
    if img is None:
        print(f"Не удалось загрузить изображение: {example_image}")
        return

    try:
        result = undistort_image(img, mtx, dist)
        cv2.imwrite('calibrated_result.jpg', result)
        print("The processed images have been saved: calibrated_result.jpg")
    except Exception as e:
        print("Ошибка при коррекции изображения:", e)
        return

    try:
        rvec, tvec, rotation_matrix = estimate_pose_from_chessboard(img, CHB_SIZE, mtx, dist)
        pt3d = np.array([[0, 0, 1830]], dtype=np.float32)
        project_known_3d_point(img, pt3d, mtx, dist)
        u, v = 77, 100
        pixel_to_normalized_ray(u, v, mtx, dist)
    except Exception as e:
        print("Ошибка при расчёте позы или проекции:", e)

    print("END OF MAIN reached")

if __name__ == "__main__":
    main()