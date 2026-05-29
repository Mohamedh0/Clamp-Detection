import cv2
import os

# video = cv2.VideoCapture("./192.168.1.102_01_20260513114921968.mp4")
# total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

# frame_indices = [ i for i in range(total_frames)]

# os.makedirs("frames", exist_ok=True)
# for idx in frame_indices:
#     video.set(cv2.CAP_PROP_POS_FRAMES, idx)
#     ret, frame = video.read()
#     if ret:
#         cv2.imwrite(f"frames/{idx}.jpg", frame)
# video.release()

def split_video(video_path, num_parts):
    # 1. Open the source video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # 2. Get properties: FPS, dimensions, and total frames
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Define the codec (e.g., mp4v for .mp4)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    # 3. Calculate frames per part
    frames_per_part = total_frames // num_parts
    
    for i in range(num_parts):
        output_name = f"part_{i+1}.mp4"
        out = cv2.VideoWriter(output_name, fourcc, fps, (width, height))
        
        # Determine exactly where this part ends
        # The last part collects any leftover frames from rounding
        end_frame = (i + 1) * frames_per_part if i < num_parts - 1 else total_frames
        
        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
        
        out.release()
        print(f"Saved: {output_name}")

    cap.release()

# Usage
split_video('./192.168.1.102_01_20260513114921968.mp4', 5)