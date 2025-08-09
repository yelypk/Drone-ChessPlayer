import argparse
from src.drone.io.video import capture_loop
def main():
    parser = argparse.ArgumentParser(description='Capture video/frames from camera.')
    parser.add_argument('--camera', type=int, default=0)
    args = parser.parse_args()
    capture_loop(camera_index=args.camera)
if __name__ == '__main__':
    main()
