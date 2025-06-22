
# Hybrid Handwritten Text Recognition (HTR)

This repository contains an implementation of a Hybrid Handwritten Text Recognition (HTR) model using:

- Swin Transformer (for visual feature extraction)
- Conformer (for sequence modeling)
- TrOCR Decoder (for text generation)

The model is trained and evaluated on the IAM Handwriting Dataset.

## Project Structure

```
HANDWRITTEN_TEXT_RECOGNITION/
├── data_loader.py               # Dataset loading and preprocessing
├── model/
│   ├── model_swin_transformer.py  # Swin Transformer module
│   ├── conformer.py               # Conformer module
│   └── trocr_decoder.py           # TrOCR decoder module
├── train2.py                    # Main training script with checkpointing
├── eval.py                      # Evaluation script (to be added)
├── inference.py                 # Inference/prediction script (to be added)
├── requirements.txt             # Required Python packages
└── .gitignore                   # Files and folders to ignore in version control
```

## Installation

```bash
# Create a virtual environment (optional)
python -m venv hybrid
# Activate the environment (Windows)
.\hybrid\Scriptsctivate

# Install dependencies
pip install -r requirements.txt
```

## Dataset

- IAM Handwriting Dataset (via Hugging Face Datasets)
- Automatically split into training and validation sets

## Training

```bash
python train2.py
```

Checkpoints will be saved in the `checkpoints/` directory after each epoch.

- `model_epoch_{n}_trainonly.pt` contains model after training step.
- `model_epoch_{n}.pt` contains the full model after validation.

Training automatically resumes from the latest checkpoint.

## Checkpoint Directory Ignored

Ensure that the `checkpoints/` folder is not pushed to GitHub. It is included in `.gitignore`.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- transformers
- datasets
- tqdm
- evaluate

Install with:

```bash
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License.
