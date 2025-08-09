## Layout
See `src/drone/` for modules:
- `core/` math & vision
- `io/` I/O & camera
- `viz/` drawing
- `app/` runnable CLIs

## Quickstart
```bash
pip install -r requirements.txt
# 1) Calibrate from images:
python -m src.drone.app.calibrate_cli --images "frames/*.jpg"
# 2) Interactive projection on an image used for calibration:
python -m src.drone.app.project_cli --image frames/one.jpg --intr intrinsics.npz
# 3) Capture from camera:
python -m src.drone.app.capture_cli --camera 0
```
