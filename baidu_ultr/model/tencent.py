import torch
from torch import nn
from transformers import BertModel

from baidu_ultr.const import TOKEN_OFFSET


class TencentModel(nn.Module):
    def __init__(self, path, device):
        super().__init__()
        self.path = path
        self.device = device

    def load(self):
        model = BertModel.from_pretrained(self.path, local_files_only=True)
        model.to(self.device)
        torch.compile(model)
        self.model = model

    def forward(self, tokens, token_types):
        mask = self.mask_attention(tokens, TOKEN_OFFSET)
        output = self.model(
            input_ids=tokens.to(self.device),
            attention_mask=mask.to(self.device),
            token_type_ids=token_types.to(self.device),
        )

        return output.pooler_output

    @staticmethod
    def mask_attention(tokens: torch.IntTensor, offset: int):
        """
        Hugging face attention mask, is True for all tokens that should NOT be masked.
        """
        return tokens >= offset
