import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.linear_model import RANSACRegressor


ROOT = Path(__file__).resolve().parent.parent
VIDEO_DIR = ROOT / "data" / "videos"
RESULTS_DIR = ROOT / "results"
VIDEO_OUTPUT_DIR = RESULTS_DIR / "videos"
SNAPSHOT_DIR = RESULTS_DIR / "snapshots"
METRICS_PATH = RESULTS_DIR / "lane_metrics.json"
SUMMARY_PATH = RESULTS_DIR / "lane_metrics_summary.md"

SOURCE_VIDEOS = [
    "project_video.mp4",
    "challenge_video.mp4",
    "harder_challenge_video.mp4",
]


@dataclass
class VideoMetrics:
    video_name: str
    frames_processed: int
    left_detections: int
    right_detections: int
    both_lane_detections: int
    failed_frames: int
    output_path: str
    processing_time_sec: float
    processing_fps: float
    source_fps: float
    lane_width_mean_px: float
    lane_width_std_px: float


class LaneTracker:
    def __init__(self):
        self.kf = cv2.KalmanFilter(3, 3)
        self.kf.transitionMatrix = np.eye(3, dtype=np.float32)
        self.kf.measurementMatrix = np.eye(3, dtype=np.float32)
        self.kf.processNoiseCov = np.eye(3, dtype=np.float32) * 1e-4
        self.kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
        self.kf.errorCovPost = np.eye(3, dtype=np.float32)
        self.initialized = False

    def update(self, coeffs):
        if coeffs is not None:
            measurement = np.asarray(coeffs, dtype=np.float32).reshape(3, 1)
            if not self.initialized:
                self.kf.statePost = measurement.copy()
                self.kf.statePre = measurement.copy()
                self.initialized = True
                return measurement.flatten()
            self.kf.predict()
            corrected = self.kf.correct(measurement)
            return corrected.flatten()

        if not self.initialized:
            return None

        predicted = self.kf.predict()
        return predicted.flatten()


def ensure_dirs():
    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)


def scaled_points(width, height):
    src = np.float32(
        [
            [0.15625 * width, 1.0 * height],
            [0.859375 * width, 1.0 * height],
            [0.464844 * width, 0.625 * height],
            [0.535156 * width, 0.625 * height],
        ]
    )
    dst = np.float32(
        [
            [0.234375 * width, 1.0 * height],
            [0.765625 * width, 1.0 * height],
            [0.234375 * width, 0.0],
            [0.765625 * width, 0.0],
        ]
    )
    return src, dst


def preprocess_frame(frame):
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white_mask = cv2.inRange(hls, np.array([0, 200, 0]), np.array([180, 255, 255]))
    yellow_mask = cv2.inRange(hls, np.array([15, 0, 100]), np.array([35, 255, 255]))
    combined_binary = cv2.bitwise_or(white_mask, yellow_mask)
    blurred = cv2.GaussianBlur(combined_binary, (5, 5), 0)
    return blurred


def region_of_interest(image):
    mask = np.zeros_like(image)
    height, width = image.shape[:2]
    vertices = np.array(
        [
            [
                (0, height),
                (int(0.45 * width), int(0.625 * height)),
                (int(0.55 * width), int(0.625 * height)),
                (width, height),
            ]
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, vertices, 255)
    return cv2.bitwise_and(image, mask)


def fit_lane(binary_slice, x_offset=0):
    y_coords, x_coords = binary_slice.nonzero()
    if len(x_coords) < 120:
        return None

    features = np.column_stack((y_coords**2, y_coords))
    model = RANSACRegressor(random_state=0)
    try:
        model.fit(features, x_coords)
    except ValueError:
        return None

    estimator = model.estimator_
    coeffs = np.array(
        [
            float(estimator.coef_[0]),
            float(estimator.coef_[1]),
            float(estimator.intercept_ + x_offset),
        ]
    )
    return coeffs


def valid_lane_pair(left_fit, right_fit, height):
    if left_fit is None or right_fit is None:
        return False

    y_bottom = height - 1
    y_mid = int(height * 0.7)
    left_bottom = np.polyval(left_fit, y_bottom)
    right_bottom = np.polyval(right_fit, y_bottom)
    left_mid = np.polyval(left_fit, y_mid)
    right_mid = np.polyval(right_fit, y_mid)

    width_bottom = right_bottom - left_bottom
    width_mid = right_mid - left_mid

    return (
        250 <= width_bottom <= 900
        and 250 <= width_mid <= 900
        and left_bottom < right_bottom
        and left_mid < right_mid
    )


def draw_lane_overlay(frame, left_fit, right_fit, inverse_matrix):
    height, width = frame.shape[:2]
    plot_y = np.linspace(0, height - 1, height)
    left_x = np.polyval(left_fit, plot_y)
    right_x = np.polyval(right_fit, plot_y)

    left_x = np.clip(left_x, 0, width - 1)
    right_x = np.clip(right_x, 0, width - 1)

    lane_overlay = np.zeros_like(frame)
    pts_left = np.column_stack((left_x, plot_y))
    pts_right = np.column_stack((right_x, plot_y))[::-1]
    polygon = np.vstack((pts_left, pts_right)).astype(np.int32)
    cv2.fillPoly(lane_overlay, [polygon], (0, 255, 0))

    unwarped = cv2.warpPerspective(lane_overlay, inverse_matrix, (width, height))
    blended = cv2.addWeighted(frame, 1.0, unwarped, 0.3, 0)
    lane_width = float(right_x[-1] - left_x[-1])
    return blended, lane_width


def annotate_frame(frame, metrics_line):
    cv2.putText(
        frame,
        metrics_line,
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def process_video(video_name):
    video_path = VIDEO_DIR / video_name
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    src, dst = scaled_points(frame_width, frame_height)
    matrix = cv2.getPerspectiveTransform(src, dst)
    inverse_matrix = cv2.getPerspectiveTransform(dst, src)

    output_path = VIDEO_OUTPUT_DIR / f"{video_path.stem}_lane_detection.mp4"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps or 20.0,
        (frame_width, frame_height),
    )

    left_tracker = LaneTracker()
    right_tracker = LaneTracker()
    left_detections = 0
    right_detections = 0
    both_detections = 0
    failed_frames = 0
    lane_widths = []
    frames_processed = 0
    snapshot_targets = {max(0, int(total_frames * ratio) - 1) for ratio in (0.25, 0.5, 0.75)}

    start_time = time.perf_counter()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        binary = preprocess_frame(frame)
        masked = region_of_interest(binary)
        warped = cv2.warpPerspective(masked, matrix, (frame_width, frame_height))

        midpoint = frame_width // 2
        left_raw = fit_lane(warped[:, :midpoint], x_offset=0)
        right_raw = fit_lane(warped[:, midpoint:], x_offset=midpoint)

        if left_raw is not None:
            left_detections += 1
        if right_raw is not None:
            right_detections += 1

        left_fit = left_tracker.update(left_raw)
        right_fit = right_tracker.update(right_raw)

        if valid_lane_pair(left_fit, right_fit, frame_height):
            both_detections += 1
            rendered, lane_width = draw_lane_overlay(frame, left_fit, right_fit, inverse_matrix)
            lane_widths.append(lane_width)
            status_line = f"Detection: stable | lane width: {lane_width:.0f}px"
        else:
            failed_frames += 1
            rendered = frame.copy()
            status_line = "Detection: fallback / missing lane estimate"

        annotate_frame(rendered, status_line)
        writer.write(rendered)

        if frames_processed in snapshot_targets:
            snapshot_path = SNAPSHOT_DIR / f"{video_path.stem}_frame_{frames_processed + 1}.png"
            cv2.imwrite(str(snapshot_path), rendered)

        frames_processed += 1

    processing_time = time.perf_counter() - start_time

    cap.release()
    writer.release()

    processing_fps = frames_processed / processing_time if processing_time else 0.0
    lane_width_mean = float(np.mean(lane_widths)) if lane_widths else 0.0
    lane_width_std = float(np.std(lane_widths)) if lane_widths else 0.0

    return VideoMetrics(
        video_name=video_name,
        frames_processed=frames_processed,
        left_detections=left_detections,
        right_detections=right_detections,
        both_lane_detections=both_detections,
        failed_frames=failed_frames,
        output_path=str(output_path.relative_to(ROOT)),
        processing_time_sec=round(processing_time, 2),
        processing_fps=round(processing_fps, 2),
        source_fps=round(source_fps, 2),
        lane_width_mean_px=round(lane_width_mean, 2),
        lane_width_std_px=round(lane_width_std, 2),
    )


def write_summary(metrics):
    lines = [
        "# Lane Detection Metrics",
        "",
        "| Video | Frames | Both Lanes | Detection Rate | Failed Frames | Processing FPS | Output |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for item in metrics:
        detection_rate = (item.both_lane_detections / item.frames_processed) if item.frames_processed else 0.0
        lines.append(
            f"| {item.video_name} | {item.frames_processed} | {item.both_lane_detections} | "
            f"{detection_rate:.1%} | {item.failed_frames} | {item.processing_fps:.2f} | {item.output_path} |"
        )

    SUMMARY_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ensure_dirs()
    metrics = [process_video(video_name) for video_name in SOURCE_VIDEOS]
    METRICS_PATH.write_text(
        json.dumps([asdict(item) for item in metrics], indent=2),
        encoding="utf-8",
    )
    write_summary(metrics)

    print(f"Saved metrics to {METRICS_PATH}")
    for item in metrics:
        detection_rate = (item.both_lane_detections / item.frames_processed) if item.frames_processed else 0.0
        print(
            f"{item.video_name}: {item.both_lane_detections}/{item.frames_processed} frames "
            f"({detection_rate:.1%}) with both lanes detected"
        )


if __name__ == "__main__":
    main()
