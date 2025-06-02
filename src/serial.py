import os
from frame_processor import load_model, classify_frame

if __name__ == "__main__":
    model = load_model()
    frames_dir = "data/processed/frames"
    frame_files = [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".jpg")]
    results = []
    for fp in frame_files:
        prob = classify_frame(fp, model)
        results.append((os.path.basename(fp), prob))
    # Salvar resultados em CSV
    with open("results_serial.csv", "w") as f:
        f.write("frame,sem_mascara_prob\n")
        for frame, prob in results:
            f.write(f"{frame},{prob:.4f}\n")
    print("Processamento serial concluído.")
