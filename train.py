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
from model.trocr_decoder_1 import TrOCRDecoder
import traceback

# Extra safety settings
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

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

    batch_size = 4
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
            if batch is None:
                continue

            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)

            # Skip empty label batches (all -100)
            if labels[labels != -100].numel() == 0:
                print(f"⚠️ Skipping batch {i} due to empty label input")
                continue

            optimizer.zero_grad()
            try:
                with autocast():
                    outputs = model(pixel_values, labels=labels)
                    loss = outputs.loss

                if loss is None or torch.isnan(loss) or torch.isinf(loss):
                    print(f"⚠️ Skipping batch {i} due to NaN/Inf loss.")
                    continue

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                train_loss += loss.item()
                tqdm_train.set_postfix(loss=loss.item())

            except RuntimeError as e:
                print(f"⚠️ CUDA error on batch {i}: {e}")
                torch.cuda.empty_cache()
                continue

        # Save model after training step
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
            for j, batch in enumerate(tqdm_val):
                if batch is None:
                    continue
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)

                if labels[labels != -100].numel() == 0:
                    continue

                try:
                    with autocast():
                        outputs = model(pixel_values, labels=labels)
                except RuntimeError as e:
                    print(f"⚠️ Skipping val batch {j} due to error: {e}")
                    torch.cuda.empty_cache()
                    continue

                if outputs.loss is None or torch.isnan(outputs.loss) or torch.isinf(outputs.loss):
                    continue

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

        # Save full model checkpoint
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
