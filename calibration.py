import cv2
import numpy as np 
import glob
import os 

CHB_SIZE=(8, 6)
CRITERIES=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def coordination_points(board_size):
    cop = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    cop[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
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
    example_image = 'frames/kFM1C.jpg'

    deskpoints, imgpoints, image_size = collections(image_folder, CHB_SIZE)
    mtx, dist, rvecs, tvecs = calibrate_camera(deskpoints, imgpoints, image_size)

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    np.savez('camera_calibration.npz', mtx=mtx, dist=dist)

    result = undistort_image(example_image, mtx, dist)
    cv2.imwrite('calibrated_result.jpg', result)
    print("Saved undistorted image as calibrated_result.jpg")

if __name__ == "__main__": #для того, щоб використовувати код як підключаємий пакет
    main()