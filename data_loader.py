import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

from datasets import load_dataset
from transformers import TrOCRProcessor
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import torch

class IAMDataset(Dataset):
    def __init__(self, dataset=None, split='train', processor_name="microsoft/trocr-base-handwritten"):
        print("📥 Loading IAM dataset...")
        self.processor = TrOCRProcessor.from_pretrained(processor_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

        if dataset is None:
            try:
                # Load full dataset
                full_dataset = load_dataset("gagan3012/IAM", split="train", trust_remote_code=True)
                print("✅ Dataset loaded.")

                # Filter out corrupt or incomplete samples
                full_dataset = full_dataset.filter(lambda x: x['text'] is not None and x['image'] is not None)
                print(f"✅ Filtered samples: {len(full_dataset)}")

                # Split into train/validation
                split_data = full_dataset.train_test_split(test_size=0.3, seed=42)
                self.dataset = split_data['train'] if split == 'train' else split_data['test']

            except Exception as e:
                print(f"❌ Dataset load failed: {e}")
                raise
        else:
            self.dataset = dataset

        print(f"✅ Loaded {len(self.dataset)} samples.")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            image = item["image"]

            # Ensure image is in PIL format
            if not isinstance(image, Image.Image):
                image = Image.fromarray(image)

            image = image.convert("RGB")
            pixel_values = self.transform(image)

            labels = self.processor.tokenizer(
                item["text"],
                padding="max_length",
                max_length=128,
                truncation=True,
                return_tensors="pt"
            ).input_ids.squeeze(0)

            return {"pixel_values": pixel_values, "labels": labels}

        except Exception as e:
            print(f"⚠️ Error at index {idx}: {e}")
            return None

# Only for testing dataset (not needed in training script)
if __name__ == "__main__":
    try:
        dataset = IAMDataset(split="train")
        sample = dataset[0]
        if sample:
            print(f"🔍 Sample keys: {sample.keys()}")
            print(f"🖼️ Pixel shape: {sample['pixel_values'].shape}")
            print(f"📝 Labels (first 10): {sample['labels'][:10]}")
    except Exception as e:
        print(f"Dataset test failed: {e}")
