import argparse
import numpy as np
import cv2
from src.drone.core.types import CameraIntrinsics
from src.drone.core.pose import estimate_pose_from_chessboard
from src.drone.core.geom import project_points, pixel_to_normalized_ray
from src.drone.viz.overlay import draw_points, draw_axes
from src.drone.config import CHB_SIZE, CELL_MM

def load_intrinsics(path='intrinsics.npz'):
    import numpy as np
    data = np.load(path)
    return CameraIntrinsics(K=data['K'], dist=data['dist'])

def grid_ij_to_point_mm(i, j, cell_mm=CELL_MM):
    return np.array([[i * cell_mm, j * cell_mm, 0.0]], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description='Интерактивная проекция точки на шахматной доске (WASD).')
    parser.add_argument('--image', type=str, required=True, help='Путь к изображению с шахматной доской')
    parser.add_argument('--intr', type=str, default='intrinsics.npz', help='Файл с параметрами камеры')
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)

    intr = load_intrinsics(args.intr)
    pose = estimate_pose_from_chessboard(img, intr, CHB_SIZE)
    if pose is None:
        print('Не удалось определить позу на изображении.')
        return

    print("Управление: W/A/S/D — перемещение по клеткам, Q или Esc — выход.")

    i, j = CHB_SIZE[0] // 2, CHB_SIZE[1] // 2

    while True:
        pt3d = grid_ij_to_point_mm(i, j)
        uv = project_points(pt3d, intr, pose.rvec, pose.tvec)

        frame = img.copy()
        frame = draw_points(frame, uv, radius=6, color=(255, 0, 0))
        frame = draw_axes(frame, intr.K, intr.dist, pose.rvec, pose.tvec)
        cv2.putText(frame, f"cell: ({i},{j})", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imshow('Projection', frame)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):  # Q or Esc
            break
        elif key == ord('w'):
            j = max(0, j - 1)
        elif key == ord('s'):
            j = min(CHB_SIZE[1] - 1, j + 1)
        elif key == ord('a'):
            i = max(0, i - 1)
        elif key == ord('d'):
            i = min(CHB_SIZE[0] - 1, i + 1)

    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()

