import os
import cv2
import torch
from concurrent.futures import ProcessPoolExecutor
import time
import csv

# CONFIG
VIDEO_PATH = 'data/raw/tokyo.mp4'
FRAME_DIR = 'data/processed/frames'
MODEL_PATH = 'models/mask_model.pt'
NUM_WORKERS = os.cpu_count()  # M1 = 8
FRAME_SKIP = 5  # salva 1 frame a cada 5

def extract_frames(video_path, out_dir, every_n=5):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    saved = 0
    while success:
        if count % every_n == 0:
            cv2.imwrite(f"{out_dir}/frame_{saved:05d}.jpg", image)
            saved += 1
        success, image = vidcap.read()
        count += 1
    print(f"Frames extraídos: {saved}")

def process_frame(frame_path):
    # Carrega modelo no subprocesso (não compartilha entre processos)
    model = torch.hub.load('./yolov5', 'custom', path=MODEL_PATH, source='local', force_reload=False)
    img = cv2.imread(frame_path)
    results = model(img)
    dets = []
    for *box, conf, cls in results.xyxy[0].cpu().numpy():
        xmin, ymin, xmax, ymax = map(int, box)
        dets.append({
            'frame': os.path.basename(frame_path),
            'xmin': xmin,
            'ymin': ymin,
            'xmax': xmax,
            'ymax': ymax,
            'class': 'mask' if int(cls) == 1 else 'no_mask',
            'conf': float(conf)
        })
    return dets

def main(num_workers=NUM_WORKERS):
    frame_files = sorted([os.path.join(FRAME_DIR, f) for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
    all_detections = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for dets in executor.map(process_frame, frame_files):
            all_detections.extend(dets)
    end = time.time()
    # Salva CSV
    with open(f"results_{num_workers}threads.csv", "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame','xmin','ymin','xmax','ymax','class','conf'])
        writer.writeheader()
        writer.writerows(all_detections)
    # Print info
    print(f"Threads: {num_workers} | Tempo: {end-start:.2f}s | Detecções: {len(all_detections)}")
    n_mask = sum(1 for r in all_detections if r['class'] == 'mask')
    n_no_mask = sum(1 for r in all_detections if r['class'] == 'no_mask')
    print(f"Total com máscara: {n_mask}, sem máscara: {n_no_mask}")

if __name__ == '__main__':
    # Descomente a linha abaixo só se precisar extrair os frames novamente!
    # extract_frames(VIDEO_PATH, FRAME_DIR, FRAME_SKIP)

    thread_counts = [1, 2, 4, 8, 16]  # Testar com diferentes contagens de threads
    for tc in thread_counts:
        print(f"\nExecutando pipeline com {tc} processos:")
        main(num_workers=tc)
