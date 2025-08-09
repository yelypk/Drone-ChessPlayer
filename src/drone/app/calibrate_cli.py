import argparse, glob, os
import cv2
from src.drone.core.calib import coordination_points, find_chessboard_corners, calibrate_camera
from src.drone.core.types import CameraIntrinsics
from src.drone.viz.overlay import draw_chessboard
from src.drone.config import CHB_SIZE

def main():
    parser = argparse.ArgumentParser(description='Batch chessboard calibration from folder of images.')
    parser.add_argument('--images', type=str, default='images/*.jpg')
    parser.add_argument('--valid_out', type=str, default='valid')
    parser.add_argument('--invalid_out', type=str, default='invalid')
    args = parser.parse_args()

    os.makedirs(args.valid_out, exist_ok=True)
    os.makedirs(args.invalid_out, exist_ok=True)

    objp = coordination_points()
    deskpoints, imgpoints, image_size = [], [], None

    images = sorted(glob.glob(args.images))
    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue
        ok, corners = find_chessboard_corners(img, CHB_SIZE)
        if ok:
            imgpoints.append(corners)
            deskpoints.append(objp)
            if image_size is None:
                image_size = (img.shape[1], img.shape[0])
            cv2.imwrite(os.path.join(args.valid_out, os.path.basename(fname)), draw_chessboard(img, CHB_SIZE, corners, True))
        else:
            cv2.imwrite(os.path.join(args.invalid_out, os.path.basename(fname)), img)
    print(f"Good process {len(imgpoints)} from {len(images)}")
    if not imgpoints:
        print("No valid chessboards found; aborting calibration.")
        return
    intr = calibrate_camera(deskpoints, imgpoints, image_size)
    # Save intrinsics to disk (npz)
    import numpy as np
    np.savez('intrinsics.npz', K=intr.K, dist=intr.dist, image_size=image_size)
    print('Saved intrinsics to intrinsics.npz')
if __name__ == '__main__':
    main()
