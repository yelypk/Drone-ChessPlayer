import os
from datetime import datetime
import cv2

# Розміри екрана (в пікселях)
SCREEN_W = 1600
SCREEN_H = 800

def show_resized(win_name, img):
    h, w = img.shape[:2]
    scale = min(SCREEN_W / w, SCREEN_H / h)  # коефіцієнт зменшення
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    cv2.imshow(win_name, resized)

def generate_filename(folder='videos', prefix='video', ext='.avi'):
    os.makedirs(folder, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f')
    return os.path.join(folder, f"{prefix}_{timestamp}{ext}")

def open_camera(index=0):
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Can't open the camera")
    return cap

def open_writer(path, fourcc_str='XVID', fps=20.0, frame_size=None):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    if frame_size is None:
        raise ValueError("frame_size must be (w,h)")
    return cv2.VideoWriter(path, fourcc, fps, frame_size)

def capture_loop(camera_index=0):
    url1 = "rtsp://admin:klop100500@192.168.1.125:554/cam/realmonitor?channel=1&subtype=0"
    url2 = "rtsp://admin:klop100500@192.168.1.126:554/cam/realmonitor?channel=1&subtype=0"
    cap = open_camera(url1)
    try:
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError("Cannot read frame from camera")
        h, w = frame.shape[:2]
        out_path = generate_filename(folder='videos', prefix='video', ext='.avi')
        writer = open_writer(out_path, frame_size=(w, h))
        print(f"Recording to: {out_path}")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            show_resized('Recording...', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                img_path = generate_filename(folder='frames', prefix='frame', ext='.jpg')
                cv2.imwrite(img_path, frame)
                print(f"Saved frame: {img_path}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
