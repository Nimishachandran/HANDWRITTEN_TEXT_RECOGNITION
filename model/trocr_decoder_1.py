import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="evaluate.*")

import torch
import torch.nn as nn
from transformers import VisionEncoderDecoderModel

class TrOCRDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pretrained VisionEncoderDecoderModel
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

        # ✅ Ensure proper decoder config for training
        self.model.config.decoder_start_token_id = self.model.config.pad_token_id or 0
        self.model.config.pad_token_id = self.model.config.pad_token_id or 0
        self.model.config.vocab_size = self.model.config.decoder.vocab_size

    def forward(self, encoder_hidden_states, labels=None):
        return self.model(
            encoder_outputs=(encoder_hidden_states,),  # pass encoder output directly
            labels=labels,
            return_dict=True
        )
