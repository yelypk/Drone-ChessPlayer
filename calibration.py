import cv2
import numpy as np 
import glob
import os 

CHB_SIZE=(8, 6)
CRITERIES=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def coordination_points(board_size):
    cop = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    cop[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    cop=cop * 11.5
    return cop

def collections(image_folder, board_size): #збір точок шахової дошки 
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

def undistort_image(image_path, mtx, dist):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    return dst

def main():
    image_folder = 'frames' 
    images = glob.glob(os.path.join(image_folder, '*.jpg'))
    example_image = images[0]

    deskpoints, imgpoints, image_size = collections(image_folder, CHB_SIZE)
    mtx, dist, rvecs, tvecs = calibrate_camera(deskpoints, imgpoints, image_size)

    print("Camera matrix:\n", mtx)
    matrixreverse=np.linalg.inv(mtx)
    print("Reversed camera matrix:\n", matrixreverse)
    print("Distortion coefficients:\n", dist)

    np.savez('camera_calibration.npz', mtx=mtx, dist=dist)

    result = undistort_image(example_image, mtx, dist)
    cv2.imwrite('calibrated_result.jpg', result)
    print("Saved undistorted image as calibrated_result.jpg")

    img = cv2.imread(example_image)
    # шукаю координати шахматної дошки
    # gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    # gray = cv2.equalizeHist(gray)

    # ret, corners = cv2.findChessboardCorners(gray, CHB_SIZE)

    # if ret:
    #     objp = coordination_points(CHB_SIZE)
    #     corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIES)

    #     success, rvec, tvec = cv2.solvePnP(objp, corners2, mtx, dist)

    #     print("\nPose estimation:")
    #     print("Rotation vector:\n", rvec)
    #     print("Translation vector:\n", tvec)

    #     distance = np.linalg.norm(tvec)
    #     print(f"Відстань до об'єкта: {distance} мм")

    #     rotation_matrix, _ = cv2.Rodrigues(rvec)
    #     print("Rotation matrix:\n", rotation_matrix)
    # else:
    #     print("Chessboard not found in the example image for pose estimation.")

    # шукаю на зображені обєкт координати якого в реальному світі я знаю
    # pt3d = np.array([[0, 0, 1830]], dtype=np.float32) 
    # rvec = np.zeros((3, 1), dtype=np.float32)
    # tvec = np.zeros((3, 1), dtype=np.float32)

    # image_points, _ = cv2.projectPoints(pt3d, rvec, tvec, mtx, dist)

    # u, v = image_points[0,0]
    # u, v = int(round(u)), int(round(v))

    # cv2.circle(img, (u, v), 7, (0, 0, 255), -1) 
    # cv2.imshow("test",img)
    # cv2.waitKey(0)

    u=77
    v=100

    pixel = np.array([[[u, v]]], dtype=np.float32) 

    norm = cv2.undistortPoints(pixel, mtx, dist)  
    x = norm[0,0,0]
    y = norm[0,0,1]
    point_3d = np.array([x, y, 1.0])
    print(f"3D напрямок (Z=1): {point_3d}")

if __name__ == "__main__": #для того, щоб використовувати код як підключаємий пакет
    main()