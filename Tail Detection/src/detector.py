import numpy as np
import cv2
from ultralytics import YOLO


def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


def get_center(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def euclidean(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def tail_diagonal(box):
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


def process_frame(model: YOLO, frame, conf: float = 0.5):
    results = model(frame, conf=conf)
    result = results[0]

    clamps = [box for box in result.boxes if int(box.cls) == 0]
    tails  = [box for box in result.boxes if int(box.cls) == 1]

    annotated_frame = result.plot()
    shortest_tail = None

    dist_tails_text = "N/A"
    if len(tails) >= 2:
        sorted_tails  = sorted(tails, key=tail_diagonal)
        shortest_tail = sorted_tails[0]
        longest_tail  = sorted_tails[-1]

        sc, lc = get_center(shortest_tail), get_center(longest_tail)
        dist_tails = euclidean(sc, lc)
        dist_tails_text = f"{dist_tails:.1f}px"
        mid = (int((sc[0] + lc[0]) / 2), int((sc[1] + lc[1]) / 2))

        cv2.line(annotated_frame, (int(sc[0]), int(sc[1])), (int(lc[0]), int(lc[1])), (255, 165, 0), 2)
        cv2.putText(annotated_frame, dist_tails_text, (mid[0], mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)

    elif len(tails) == 1:
        shortest_tail = tails[0]

    dist_clamp_text = "N/A"
    if shortest_tail is not None and clamps:
        sc = get_center(shortest_tail)
        clamp_centers = [get_center(c) for c in clamps]
        nearest_clamp = min(clamp_centers, key=lambda cc: euclidean(sc, cc))
        dist_clamp = euclidean(sc, nearest_clamp)
        dist_clamp_text = f"{dist_clamp:.1f}px"
        mid = (int((sc[0] + nearest_clamp[0]) / 2), int((sc[1] + nearest_clamp[1]) / 2))

        cv2.line(annotated_frame, (int(sc[0]), int(sc[1])),
                 (int(nearest_clamp[0]), int(nearest_clamp[1])), (0, 255, 255), 2)
        cv2.putText(annotated_frame, dist_clamp_text, (mid[0], mid[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # HUD overlay
    cv2.putText(annotated_frame, f"Clamp Count: {len(clamps)}",         (30, 50),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0),   3)
    cv2.putText(annotated_frame, f"Tail Count: {len(tails)}",           (30, 90),  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255),   3)
    cv2.putText(annotated_frame, f"Short-Long Tail: {dist_tails_text}", (30, 130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 165, 0), 2)
    cv2.putText(annotated_frame, f"Short Tail-Clamp: {dist_clamp_text}",(30, 165), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    return annotated_frame


def save_video_detection(model: YOLO, video_path: str, output_name: str, conf: float = 0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(f"{output_name}.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            annotated = process_frame(model, frame, conf=conf)
            out.write(annotated)
    finally:
        cap.release()
        out.release()

    print(f"Finished: {output_name}.mp4")
