from torchvision import transforms
from PIL import Image
import torch

def load_model(model_path="models/mobilenetv2_mask.h5"):
    model = torch.load(model_path, map_location="cpu")
    model.eval()
    return model

preprocess = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
])

def classify_frame(img_path, model=None):
    img = Image.open(img_path).convert("RGB")
    inp = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        out = model(inp)
        prob = torch.softmax(out, dim=1)[0]
    return float(prob[1])  # probabilidade de “sem máscara”

if __name__ == "__main__":
    model = load_model()
    prob = classify_frame("data/processed/frames/frame_00000.jpg", model)
    print(f"P(sem máscara) = {prob:.2f}")
