import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

import torch
from PIL import Image
from torchvision import transforms
from data_loader import IAMDataset
from train import HybridHTR

def inference(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridHTR().to(device)
    model.load_state_dict(torch.load("checkpoints/model_epoch_10.pt"))
    model.eval()

    processor = IAMDataset().processor
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    image = Image.open(image_path).convert("RGB")
    pixel_values = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)
        predicted_ids = outputs.logits.argmax(-1)
        predicted_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    
    print(f"Predicted Text: {predicted_text}")
    return predicted_text

if __name__ == "__main__":
    inference("C:/Users/HP/Desktop/HYBRID_HTR/sample_image.png")