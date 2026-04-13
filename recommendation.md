# CV Project: Lane Boundary Detection from Highway Video

### Abstract
Lane boundary extraction is a practical computer vision task that can be solved with classical methods. This project builds a lane-marking detector on highway video using filtering, edge detection, region masking, and line fitting. The output is a lane overlay and frame-level quality metrics.

### Current State of the Problem
- Lane visibility changes due to shadows, worn paint, and lighting variation.
- Deep learning solutions exist, but they require larger datasets and longer training cycles.
- A classical CV pipeline is faster to build, easier to explain, and better aligned with class scope.

### Desired Outcome
- A video-processing pipeline that outputs:
  - Left and right lane-line estimates per frame
  - Lane overlay visualization
  - Failure flags when confidence is below threshold

### Approach
1. Use the current public dashcam clips in `data/videos/` and define a road region-of-interest (ROI).
2. Convert frames to grayscale and apply smoothing to reduce noise.
3. Detect edges with Canny.
4. Detect lane candidates using Hough line transform.
5. Stabilize lane estimates with temporal averaging across frames.
6. Report per-frame detection success and error metrics.

### Data Inputs
- Current dataset: 3 public highway dashcam videos stored in `data/videos/`.
- Videos used: `project_video.mp4`, `challenge_video.mp4`, and `harder_challenge_video.mp4`.
- These clips provide a practical starting set with straight lanes, shadows, and more difficult curvature cases.
- Optional extension: add more public or self-collected clips if broader evaluation is needed.
- Optional benchmark: TuSimple lane dataset if Kaggle access is configured later.

### Techniques to Leverage
- Filtering + Canny edge detection
- Hough line transform for line fitting
- ROI masking and temporal smoothing
- Optional perspective transform (bird's-eye view)

### Metrics
- Frame-level lane detection rate
- False-positive line rate
- Lateral deviation from manually labeled lane points (sampled frames, if labels are added)
- Runtime (FPS)

### Potential Blockers
- Faded lane paint
- Strong shadows and glare
- Curved lanes causing unstable linear fits

### Primary References
- OpenCV Canny tutorial: https://docs.opencv.org/3.4/da/d22/tutorial_py_canny.html
- OpenCV Hough Line Transform tutorial: https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html
