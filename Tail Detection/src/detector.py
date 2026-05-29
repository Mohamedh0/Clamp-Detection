import numpy as np
import cv2
from ultralytics import YOLO


def load_model(weights_path: str) -> YOLO:
    return YOLO(weights_path)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def get_center(box):

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2
    )


def euclidean(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2
    )


def tail_diagonal(box):

    x1, y1, x2, y2 = box.xyxy[0].tolist()

    return np.sqrt(
        (x2 - x1) ** 2 +
        (y2 - y1) ** 2
    )


def box_center_x(box):

    x1, _, x2, _ = box.xyxy[0].tolist()

    return (x1 + x2) / 2


# ─────────────────────────────────────────────
# HUD
# ─────────────────────────────────────────────

def draw_hud(frame, rows):

    frame_h = frame.shape[0]

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness  = 2

    pad      = 18
    line_gap = 40

    # FULL text size
    sizes = [
        cv2.getTextSize(
            f"{label}: {text}",
            font,
            font_scale,
            thickness
        )[0]
        for label, text, _ in rows
    ]

    box_w = max(s[0] for s in sizes) + pad * 2
    box_h = len(rows) * line_gap + pad * 2

    margin = 20

    x0 = margin
    y0 = frame_h - box_h - margin

    overlay = frame.copy()

    # background
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + box_w, y0 + box_h),
        (20, 20, 20),
        -1
    )

    cv2.addWeighted(
        overlay,
        0.55,
        frame,
        0.45,
        0,
        frame
    )

    # border
    cv2.rectangle(
        frame,
        (x0, y0),
        (x0 + box_w, y0 + box_h),
        (200, 200, 200),
        2
    )

    # text
    for i, (label, text, color) in enumerate(rows):

        ty = y0 + pad + (i + 1) * line_gap - 6

        full_text = f"{label}: {text}"

        cv2.putText(
            frame,
            full_text,
            (x0 + pad, ty),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA
        )


# ─────────────────────────────────────────────
# Frame Processing
# ─────────────────────────────────────────────

def process_frame(model: YOLO, frame, conf: float = 0.5):

    results = model(frame, conf=conf)

    result = results[0]

    # class 0 = clamp
    # class 1 = tail

    clamps = [
        box for box in result.boxes
        if int(box.cls) == 0
    ]

    tails = [
        box for box in result.boxes
        if int(box.cls) == 1
    ]

    clamp_count = len(clamps)
    tail_count  = len(tails)

    annotated_frame = result.plot()

    # ─────────────────────────────────────────
    # Sort clamps
    # ─────────────────────────────────────────

    clamps_sorted = sorted(
        clamps,
        key=box_center_x
    )

    # ─────────────────────────────────────────
    # Shortest tail
    # ─────────────────────────────────────────

    shortest_tail = None

    if tails:

        shortest_tail = min(
            tails,
            key=tail_diagonal
        )

    # ─────────────────────────────────────────
    # Distance:
    # shortest tail ↔ longest tail
    # ─────────────────────────────────────────

    dist_tails_text = "N/A"

    if len(tails) >= 2:

        sorted_tails = sorted(
            tails,
            key=tail_diagonal
        )

        shortest = sorted_tails[0]
        longest  = sorted_tails[-1]

        shortest_center = get_center(shortest)
        longest_center  = get_center(longest)

        tail_distance = euclidean(
            shortest_center,
            longest_center
        )

        dist_tails_text = f"{tail_distance:.1f}px"

        # draw line
        cv2.line(
            annotated_frame,
            (
                int(shortest_center[0]),
                int(shortest_center[1])
            ),
            (
                int(longest_center[0]),
                int(longest_center[1])
            ),
            (255, 165, 0),
            2
        )

        # midpoint
        mid = (
            int(
                (shortest_center[0] + longest_center[0]) / 2
            ),
            int(
                (shortest_center[1] + longest_center[1]) / 2
            )
        )

        cv2.putText(
            annotated_frame,
            dist_tails_text,
            (mid[0], mid[1] - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 165, 0),
            2
        )

    # ─────────────────────────────────────────
    # Distance:
    # shortest tail → nearest RIGHT clamp
    # ─────────────────────────────────────────

    dist_clamp_text = "N/A"

    if shortest_tail is not None and len(clamps_sorted) > 0:

        tail_center = get_center(shortest_tail)

        tail_x = tail_center[0]

        # clamps on RIGHT side
        right_clamps = [
            c for c in clamps_sorted
            if box_center_x(c) > tail_x
        ]

        if right_clamps:

            nearest_right = min(
                right_clamps,
                key=lambda c: euclidean(
                    tail_center,
                    get_center(c)
                )
            )

            clamp_center = get_center(nearest_right)

            distance = euclidean(
                tail_center,
                clamp_center
            )

            dist_clamp_text = f"{distance:.1f}px"

            # draw line
            cv2.line(
                annotated_frame,
                (
                    int(tail_center[0]),
                    int(tail_center[1])
                ),
                (
                    int(clamp_center[0]),
                    int(clamp_center[1])
                ),
                (0, 255, 255),
                2
            )

            # midpoint
            mid = (
                int(
                    (tail_center[0] + clamp_center[0]) / 2
                ),
                int(
                    (tail_center[1] + clamp_center[1]) / 2
                )
            )

            cv2.putText(
                annotated_frame,
                dist_clamp_text,
                (mid[0], mid[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

    # ─────────────────────────────────────────
    # HUD
    # ─────────────────────────────────────────

    rows = [
        (
            "Clamp Count",
            str(clamp_count),
            (0, 255, 0)
        ),
        (
            "Tail Count",
            str(tail_count),
            (0, 0, 255)
        ),
        (
            "Short-Long Tail",
            dist_tails_text,
            (255, 165, 0)
        ),
        (
            "Tail→Clamp Before",
            dist_clamp_text,
            (0, 255, 255)
        ),
    ]

    draw_hud(
        annotated_frame,
        rows
    )

    return annotated_frame


# ─────────────────────────────────────────────
# Video Processing
# ─────────────────────────────────────────────

def save_video_detection(
    model: YOLO,
    video_path: str,
    output_name: str,
    conf: float = 0.5
):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Cannot open video: {video_path}"
        )

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30

    out = cv2.VideoWriter(
        f"{output_name}.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    try:

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            annotated = process_frame(
                model,
                frame,
                conf=conf
            )

            out.write(annotated)

    finally:

        cap.release()
        out.release()

    print(f"Finished: {output_name}.mp4")
