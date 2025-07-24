import cv2
import os
from datetime import datetime
# намагаюсь не перезаписувати файл завдяки генерації імені файла за часом 
def generate_video_filename(folder='videos', prefix='video', ext='.avi'):
    if not os.path.exists(folder):
        os.makedirs(folder)
    timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S-%f') # генерує унікальність часу, %f - з урахуванням мікросекунд 
    return os.path.join(folder, f"{prefix}_{timestamp}{ext}")

def capture_video():
    cap=cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cant open the camera")
        return

    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps=20.0

    filename=generate_video_filename()
    fourcc=cv2.VideoWriter_fourcc(*'XVID')
    out=cv2.VideoWriter(filename, fourcc, fps, (width, height))

    print("Press q to stop recording")
  # цикл читання з камери 
    while True:
        ret, frame=cap.read()
        if not ret:
            break
        out.write(frame)
        cv2.imshow('Recording...', frame)

        key=cv2.waitKey(1) & 0xFF
        if key==ord('q'):
            break
        elif key==ord('s'):
            img_filename=generate_video_filename(folder='frames', prefix='frame', ext='.jpg') #зберегання кадра як зображення 
            cv2.imwrite(img_filename, frame)
            print(f"Saved frame as: {img_filename}")
            
if __name__=="__main__":
    capture_video()

 