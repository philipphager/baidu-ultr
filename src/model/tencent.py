from pathlib import Path
from typing import Dict

import torch
from torch import nn
from transformers import BertModel


class TencentModel(nn.Module):
    def __init__(self, model_directory: str, model_name: str, special_tokens: Dict):
        super().__init__()
        self.model_directory = Path(model_directory)
        self.model_name = model_name
        self.special_tokens = special_tokens
        self.model = None
        self.device = None

    def load(self, device):
        path = self.model_directory / self.model_name
        assert path.exists(), f"""
        Please download the Baidu 12Layer pre-trained language model from Google Drive:
        https://drive.google.com/file/d/1KWOd2TsFwgWIAMDBv9mVXQ4D5h6s5tWy/view
        
        and extract it in the model directory: {self.model_directory}
        """
        model = BertModel.from_pretrained(path, local_files_only=True)
        model.to(device)
        torch.compile(model)
        self.model = model
        self.device = device

    def forward(self, tokens, token_types):
        mask = self.mask_attention(tokens, self.special_tokens)
        output = self.model(
            input_ids=tokens.to(self.device),
            attention_mask=mask.to(self.device),
            token_type_ids=token_types.to(self.device),
        )

        return output.pooler_output

    @staticmethod
    def mask_attention(tokens: torch.IntTensor, special_token: Dict[str, int]):
        """
        Hugging face attention mask, is True for all tokens that should NOT be masked.
        """
        return tokens > special_token["PAD"]
