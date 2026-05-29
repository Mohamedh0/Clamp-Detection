import os
from detector import load_model, save_video_detection

WEIGHTS = "/path/to/your/model"
CLIPS_DIR = "/path/to/your/clips"   


def main():
    model = load_model(WEIGHTS)

    for clip in os.listdir(CLIPS_DIR):
        if not clip.endswith(".mp4"):
            continue
        video_path  = os.path.join(CLIPS_DIR, clip)
        output_name = f"output_{os.path.splitext(clip)[0]}"
        print(f"Processing: {clip} → {output_name}.mp4")
        save_video_detection(model, video_path, output_name)

    print("All clips processed!")


if __name__ == "__main__":
    main()