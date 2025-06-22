import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="evaluate.*")

import torch
from torch.utils.data import DataLoader
from evaluate import load
from data_loader import IAMDataset
from train import HybridHTR
import os

def evaluate(checkpoint_path="checkpoints/model_epoch_10.pt"):
    print("📝 Evaluating...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}. Run train.py first.")

    try:
        model = HybridHTR().to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()
        print(f"✅ Loaded model from {checkpoint_path}")
    except Exception as e:
        raise Exception(f"Model load failed: {e}")

    try:
        val_dataset = IAMDataset(split="validation")
        val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)
        print(f"✅ Validation dataset: {len(val_dataset)} samples")
    except Exception as e:
        raise Exception(f"Dataset load failed: {e}")

    try:
        cer_metric = load("cer")
        wer_metric = load("wer")
        print("✅ Metrics loaded")
    except Exception as e:
        raise Exception(f"Metrics load failed: {e}")

    cer_score = 0.0
    wer_score = 0.0
    valid_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if batch is None:
                continue
            try:
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(pixel_values, labels=labels)
                predicted_ids = outputs.logits.argmax(-1)
                predicted_texts = val_dataset.processor.batch_decode(predicted_ids, skip_special_tokens=True)
                true_texts = val_dataset.processor.batch_decode(labels, skip_special_tokens=True)
                cer_score += cer_metric.compute(predictions=predicted_texts, references=true_texts)
                wer_score += wer_metric.compute(predictions=predicted_texts, references=true_texts)
                valid_batches += 1
            except Exception as e:
                print(f"Batch error: {e}")
                continue

    if valid_batches == 0:
        raise ValueError("No valid batches processed.")

    print(f"Validation CER: {cer_score/valid_batches:.4f}")
    print(f"Validation WER: {wer_score/valid_batches:.4f}")

if __name__ == "__main__":
    try:
        evaluate()
    except Exception as e:
        print(f"Evaluation failed: {e}")