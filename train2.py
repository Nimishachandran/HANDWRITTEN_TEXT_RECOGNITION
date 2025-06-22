import warnings
warnings.filterwarnings("ignore")

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from evaluate import load
from data_loader import IAMDataset
from model.model_swin_transformer import SwinFeatureExtractor
from model.conformer import ConformerWrapper
from model.trocr_decoder import TrOCRDecoder
import traceback

class HybridHTR(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = SwinFeatureExtractor()
        self.conformer = ConformerWrapper()
        self.decoder = TrOCRDecoder()

    def forward(self, images, labels=None):
        x = self.swin(images)
        x = self.conformer(x)
        return self.decoder(x, labels=labels)

def train():
    print("🚀 Training started...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("📥 Loading IAM dataset...")
    train_dataset = IAMDataset(split="train")
    val_dataset = IAMDataset(split="validation")

    batch_size = 1
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"✅ Loaded {len(train_dataset)} training and {len(val_dataset)} validation samples.")

    model = HybridHTR().to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    scaler = GradScaler()

    cer_metric = load("cer")
    wer_metric = load("wer")
    print("✅ Metrics loaded")

    num_epochs = 10
    checkpoint_dir = r"C:\Users\anant\OneDrive\Desktop\HYBRID_HTR\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    start_epoch = 0
    for i in range(num_epochs, 0, -1):
        path = os.path.join(checkpoint_dir, f"model_epoch_{i}.pt")
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            start_epoch = i
            print(f"🔁 Resumed from epoch {i}")
            break

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        tqdm_train = tqdm(train_loader, desc=f"🔥 Epoch {epoch+1}/{num_epochs}", leave=False)
        for i, batch in enumerate(tqdm_train):
            if batch is None: continue
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            try:
                with autocast():
                    outputs = model(pixel_values, labels=labels)
                    loss = outputs.loss

                if loss is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    train_loss += loss.item()
                    tqdm_train.set_postfix(loss=loss.item())
            except RuntimeError as e:
                print(f"⚠️ Skipping batch due to error: {e}")
                torch.cuda.empty_cache()
                continue

        # ✅ Save model immediately after training (before validation)
        train_ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}_trainonly.pt")
        try:
            torch.save(model.state_dict(), train_ckpt_path)
            print(f"💾 Saved training-only checkpoint at epoch {epoch+1}: {train_ckpt_path}")
        except Exception as e:
            print(f"❌ Failed to save training checkpoint: {e}")

        # ======================== VALIDATION ========================
        model.eval()
        val_loss = 0.0
        cer_score = 0.0
        wer_score = 0.0
        val_batches = 0
        tqdm_val = tqdm(val_loader, desc=f"🧪 Validation {epoch+1}", leave=False)

        with torch.no_grad():
            for batch in tqdm_val:
                if batch is None: continue
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                try:
                    with autocast():
                        outputs = model(pixel_values, labels=labels)
                except RuntimeError as e:
                    print(f"⚠️ Skipping val batch due to error: {e}")
                    torch.cuda.empty_cache()
                    continue

                if outputs.loss is not None:
                    val_loss += outputs.loss.item()

                pred_ids = torch.argmax(outputs.logits, dim=-1)
                pred_texts = train_dataset.processor.batch_decode(pred_ids, skip_special_tokens=True)
                true_texts = train_dataset.processor.batch_decode(labels, skip_special_tokens=True)

                cer_score += cer_metric.compute(predictions=pred_texts, references=true_texts)
                wer_score += wer_metric.compute(predictions=pred_texts, references=true_texts)
                val_batches += 1

        torch.cuda.empty_cache()

        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"Train Loss: {train_loss / len(train_loader):.4f}")
        print(f"Val Loss: {val_loss / max(val_batches, 1):.4f}")
        print(f"CER: {cer_score / max(val_batches, 1):.4f}")
        print(f"WER: {wer_score / max(val_batches, 1):.4f}")

        # ✅ Save full epoch checkpoint (after validation)
        final_ckpt_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
        try:
            torch.save(model.state_dict(), final_ckpt_path)
            print(f"📂 Saved full checkpoint at epoch {epoch+1}: {final_ckpt_path}")
        except Exception as e:
            print(f"❌ Failed to save final checkpoint: {e}")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"❌ Training failed with error:\n{e}")
        traceback.print_exc()
