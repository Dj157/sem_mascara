import os
import cv2
import torch
import face_recognition
from concurrent.futures import ThreadPoolExecutor
import time

# Configuração
VIDEO_PATH = 'data/raw/tokyo.mp4'  # ajuste o caminho do seu vídeo
FRAME_DIR = 'data/processed/frames'  # ajuste se necessário
MODEL_PATH = 'models/mask_model.pt'  # seu modelo treinado de máscara
NUM_THREADS = 2  # Altere para 2,4,6,8,16 para testes

# 1. Carregar modelo de detecção de máscara
model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_PATH, force_reload=True)

# 2. Extrair frames do vídeo (se necessário)
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

# extract_frames(VIDEO_PATH, FRAME_DIR, every_n=5)  # Descomente se precisar gerar frames

# 3. Processar frames para detecção de máscara + embedding facial
def process_frame(frame_path):
    img = cv2.imread(frame_path)
    # Detecção de máscara (usando YOLO)
    results = model(img)
    # Supondo que class 0 = sem máscara, 1 = com máscara
    mask_results = results.xyxy[0].cpu().numpy()
    faces_no_mask = [x for x in mask_results if int(x[5]) == 0]
    faces_mask = [x for x in mask_results if int(x[5]) == 1]
    # Face embedding para cada face detectada
    encodings = []
    face_locations = face_recognition.face_locations(img)
    if face_locations:
        encodings = face_recognition.face_encodings(img, face_locations)
    return {
        'frame': frame_path,
        'no_mask': len(faces_no_mask),
        'mask': len(faces_mask),
        'encodings': encodings
    }

# 4. Remover redundância de pessoas entre frames (matching embeddings)
def deduplicate_people(results, tolerance=0.6):
    unique_encodings = []
    unique_people = []
    for r in results:
        for encoding in r['encodings']:
            matches = face_recognition.compare_faces(unique_encodings, encoding, tolerance)
            if not any(matches):
                unique_encodings.append(encoding)
                unique_people.append({'frame': r['frame'], 'encoding': encoding})
    return unique_people

# 5. Pipeline paralelo e medição de tempo
def main(num_threads=8):
    frame_files = sorted([os.path.join(FRAME_DIR, f) for f in os.listdir(FRAME_DIR) if f.endswith('.jpg')])
    start = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        results = list(executor.map(process_frame, frame_files))
    unique_people = deduplicate_people(results)
    end = time.time()
    print(f"Threads: {num_threads} | Pessoas únicas: {len(unique_people)} | Tempo: {end-start:.2f}s")
    # Extra: imprimir quantos com e sem máscara
    total_no_mask = sum(r['no_mask'] for r in results)
    total_mask = sum(r['mask'] for r in results)
    print(f"Total sem máscara: {total_no_mask}, com máscara: {total_mask}")

if __name__ == '__main__':
    # Descomente a linha abaixo se quiser extrair os frames sempre que rodar
    # extract_frames(VIDEO_PATH, FRAME_DIR, every_n=5)

    thread_counts = [2, 4, 6, 8, 16]
    for n in thread_counts:
        print(f"\nExecutando pipeline com {n} threads:")
        main(n)
