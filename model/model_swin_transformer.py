import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

from transformers import SwinModel
import torch.nn as nn

class SwinFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = SwinModel.from_pretrained("microsoft/swin-tiny-patch4-window7-224")

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        x = outputs.last_hidden_state  # Shape: (batch_size, sequence_length, 768)
        return x