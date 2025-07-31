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
        print("Chessboard not found in the image for pose estimation.", flush=True)
        return None, None, None
    objp = coordination_points(board_size)
    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
    success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)
    print("\nPose estimation:", flush=True)
    print("Rotation vector:\n", rvec, flush=True)
    print("Translation vector:\n", tvec, flush=True)
    distance = np.linalg.norm(tvec)
    print(f"Відстань до об'єкта: {distance} мм", flush=True)
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    print("Rotation matrix:\n", rotation_matrix, flush=True)
    return rvec, tvec, rotation_matrix

def project_known_3d_point(img, point_3d, mtx, dist):
    rvec = np.zeros((3, 1), dtype=np.float32)
    tvec = np.zeros((3, 1), dtype=np.float32)
    try:
        image_points, _ = cv2.projectPoints(point_3d, rvec, tvec, mtx, dist)
        u, v = image_points[0, 0]
        u, v = int(round(u)), int(round(v))
        cv2.circle(img, (u, v), 7, (0, 0, 255), -1)
        cv2.imshow("Projected point", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Ошибка при проецировании точки:", e)

def pixel_to_normalized_ray(u, v, mtx, dist):
    try:
        pixel = np.array([[[u, v]]], dtype=np.float32)
        norm = cv2.undistortPoints(pixel, mtx, dist)
        x = norm[0, 0, 0]
        y = norm[0, 0, 1]
        point_3d = np.array([x, y, 1.0])
        print(f"3D направление (Z=1): {point_3d}", flush=True)
        return point_3d
    except Exception as e:
        print("Ошибка при переводе пикселя в 3D направление:", e)
        return None

def print_camera_info(mtx, dist, rvecs, tvecs):
    print("\n" + "="*40, flush=True)
    print("CAMERA MATRIX:\n", mtx, flush=True)
    print("REVERSED MATRIX:\n", np.linalg.inv(mtx), flush=True)
    print("DISTORTION COEFFICIENTS:\n", dist, flush=True)

def interactive_projection(img, mtx, dist, rvec, tvec, initial_i=0, initial_j=0, step=1, cell_size=11.5):
    print("W/A/S/D чтобы смещать точку по клеткам доски. +/- — изменить шаг. Esc — выход.")
    i, j = initial_i, initial_j
    current_step = step

    while True:
        dx = i * cell_size
        dy = j * cell_size
        pt3d = np.array([[[dx, dy, 0]]], dtype=np.float32)
        img_copy = img.copy()
        try:
            image_points, _ = cv2.projectPoints(pt3d, rvec, tvec, mtx, dist)
            u, v = image_points[0, 0]
            u, v = int(round(u)), int(round(v))
            cv2.circle(img_copy, (u, v), 7, (255, 0, 0), -1)
            cv2.putText(img_copy, f"i: {i}, j: {j}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(img_copy, f"step: {current_step} клеток", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        except Exception as e:
            print("Ошибка при проекции точки:", e)

        cv2.imshow("Control projection", img_copy)
        if cv2.getWindowProperty("Control projection", cv2.WND_PROP_VISIBLE) < 1:
            break

        key = cv2.waitKey(0)
        if key == 27:
            break
        elif key == ord('w'):
            j -= current_step
        elif key == ord('s'):
            j += current_step
        elif key == ord('a'):
            i -= current_step
        elif key == ord('d'):
            i += current_step
        elif key == ord('+') or key == ord('='):
            current_step += 1
        elif key == ord('-') and current_step > 1:
            current_step -= 1

    cv2.destroyAllWindows()

def main():
    image_folder = 'frames' 
    valid_folder = 'valid'
    invalid_folder = 'invalid'
    os.makedirs(valid_folder, exist_ok=True)
    os.makedirs(invalid_folder, exist_ok=True)

    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    if not images:
        print("No images in the folder", flush=True)
        return

    print(f"\nFounded {len(images)} of images. Searching for corners...\n", flush=True)

    imgpoints = []
    deskpoints = []
    image_size = None
    cop = coordination_points(CHB_SIZE)
    valid_images = []

    for fname in images:
        print(f"Processed: {fname}", flush=True)
        img = cv2.imread(fname)
        if img is None:
            print(f"Не удалось загрузить: {fname}", flush=True)
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        ret, corners = cv2.findChessboardCorners(gray, CHB_SIZE)

        if ret and corners.shape[0] == CHB_SIZE[0] * CHB_SIZE[1]:
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
            imgpoints.append(corners2)
            deskpoints.append(cop)
            valid_images.append(fname)
            if image_size is None:
                image_size = gray.shape[::-1]
            img_drawn = cv2.drawChessboardCorners(img, CHB_SIZE, corners2, ret)
            cv2.imwrite(os.path.join(valid_folder, os.path.basename(fname)), img_drawn)
            print("Corners have been founded and saved in valid/", flush=True)
        else:
            cv2.imwrite(os.path.join(invalid_folder, os.path.basename(fname)), img)
            print("No corners. Have been saved in invalid/", flush=True)

    print(f"\nGood process {len(imgpoints)} from {len(images)}", flush=True)

    if len(imgpoints) < 3:
        print("Not enough images for calibration", flush=True)
        return

    if image_size is None:
        print("image_size пуст — ни одно изображение не дало размер!", flush=True)
        return

    try:
        mtx, dist, rvecs, tvecs = calibrate_camera(deskpoints, imgpoints, image_size)
        print("--- Debug: about to print camera info ---", flush=True)
        print_camera_info(mtx, dist, rvecs, tvecs)
    except Exception as e:
        print("Ошибка при калибровке:", e, flush=True)
        return

    example_image = valid_images[0] if valid_images else None
    if not example_image:
        print("Нет изображений с найденными углами для вывода.", flush=True)
        return

    img = cv2.imread(example_image)
    if img is None:
        print(f"Не удалось загрузить изображение: {example_image}", flush=True)
        return

    try:
        result = undistort_image(img, mtx, dist)
        cv2.imwrite('calibrated_result.jpg', result)
        print("The processed images have been saved: calibrated_result.jpg", flush=True)
    except Exception as e:
        print("Ошибка при коррекции изображения:", e, flush=True)
        return

    rvec, tvec, rotation_matrix = estimate_pose_from_chessboard(img, CHB_SIZE, mtx, dist)
    if rvec is None:
        print("Не удалось определить позу.", flush=True)
        return

    pt3d = np.array([[0, 0, 1830]], dtype=np.float32)
    project_known_3d_point(img.copy(), pt3d, mtx, dist)
    u, v = 77, 100
    pixel_to_normalized_ray(u, v, mtx, dist)
    interactive_projection(img.copy(), mtx, dist, rvec, tvec)

    print("END OF MAIN reached", flush=True)

if __name__ == "__main__":
    main()