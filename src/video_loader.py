import cv2
import os

def extract_frames(input_path, output_dir, every_n=1):
    os.makedirs(output_dir, exist_ok=True)
    vid = cv2.VideoCapture(input_path)
    frame_idx = 0
    saved = 0
    while True:
        ret, frame = vid.read()
        if not ret:
            break
        if frame_idx % every_n == 0:
            fname = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(fname, frame)
            saved += 1
        frame_idx += 1
    vid.release()
    print(f"Frames extraídos: {saved}")

if __name__ == "__main__":
    extract_frames(
        input_path="data/raw/tokyo.mp4",
        output_dir="data/processed/frames",
        every_n=5
    )
