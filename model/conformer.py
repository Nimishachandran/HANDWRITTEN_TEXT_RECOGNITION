import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")

from espnet2.asr.encoder.conformer_encoder import ConformerEncoder
import torch.nn as nn
import torch

class ConformerWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.conformer = ConformerEncoder(
            input_size=768,  # Matches Swin output
            output_size=768,  # Matches TrOCRDecoder input
            attention_heads=4,
            linear_units=2048,
            num_blocks=4,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True
        )

    def forward(self, x):
        lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        output = self.conformer(x, lengths)[0]  # Take only the output tensor
        return output