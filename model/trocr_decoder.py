import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers.*")
warnings.filterwarnings("ignore", category=UserWarning, module="espnet.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="espnet.*")

from transformers import VisionEncoderDecoderModel
import torch
import torch.nn as nn

class TrOCRDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
        self.decoder = model.decoder
        self.decoder.config.is_decoder = True
        self.decoder.config.add_cross_attention = True

    def forward(self, encoder_hidden_states, labels=None):
        outputs = self.decoder(
            input_ids=labels,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True
        )
        logits = outputs.logits
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
        return type('Output', (), {'loss': loss, 'logits': logits})()