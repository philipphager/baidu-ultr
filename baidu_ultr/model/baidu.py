import math
from pathlib import Path
from typing import Dict

import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.optim.lr_scheduler import LambdaLR

from baidu_ultr.const import BAIDU_SPECIAL_TOKENS
from baidu_ultr.util import download_model

BAIDU_CONFIG = {
    "ntoken": 22_000,
    "hidden": 768,
    "nhead": 12,
    "nlayers": 12,
    "dropout": 0.1,
    "mode": "pretrain",
}


class BaiduModel(nn.Module):
    def __init__(self, model_directory, model_name):
        super().__init__()
        self.model_directory = Path(model_directory)
        self.model_name = model_name
        self.path = self.model_directory / model_name
        self.model = None
        self.device = None

    def load(self, device):
        if not self.path.exists():
            download_model(self.model_directory, self.model_name)

        model = TransformerModel(**BAIDU_CONFIG)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(self.path, map_location=device))
        torch.compile(model)
        self.model = model
        self.device = device

    def forward(self, tokens, token_types):
        mask = self.mask_padding(tokens, BAIDU_SPECIAL_TOKENS)
        print(mask[0], tokens[0])
        return self.model(
            src=tokens.to(self.device),
            src_segment=token_types.to(self.device),
            src_padding_mask=mask.to(self.device),
        )

    @staticmethod
    def mask_padding(tokens: torch.IntTensor, special_token: Dict[str, int]):
        """
        Torch attention mask, is True for all tokens that are padding.
        """
        return tokens == special_token["PAD"]


class TransformerModel(nn.Module):
    def __init__(self, ntoken, hidden, nhead, nlayers, dropout, mode="finetune"):
        super().__init__()
        print("Transformer is used for {}".format(mode))
        self.pos_encoder = PositionalEncoding(hidden, dropout)
        encoder_layers = TransformerEncoderLayer(
            hidden, nhead, hidden, dropout, activation="gelu"
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.token_encoder = nn.Embedding(ntoken, hidden)
        self.segment_encoder = nn.Embedding(2, hidden)
        self.norm_layer = nn.LayerNorm(hidden)
        self.hidden = hidden
        self.mode = mode

        self.dropout = nn.Dropout(dropout)

        if mode == "pretrain":
            self.to_logics = nn.Linear(hidden, ntoken)
            self.decoder = nn.Linear(hidden, 1)
        elif mode == "finetune":
            self.act = nn.ELU(alpha=1.0)
            self.fc1 = nn.Linear(hidden, 512)
            self.fc2 = nn.Linear(512, 256)
            self.fc3 = nn.Linear(256, 128)
            self.fc4 = nn.Linear(128, 1)

    def forward(self, src, src_segment, src_padding_mask=None):
        src = src.t().contiguous()
        src_segment = src_segment.t().contiguous()
        src_padding_mask = src_padding_mask

        # transformer input
        pos_emb = self.pos_encoder(src)  # get position embedding
        token_emb = self.token_encoder(src)  # get token embedding
        seg_emb = self.segment_encoder(src_segment)  # get position embedding
        x = token_emb + pos_emb + seg_emb
        x = self.norm_layer(x)
        x = self.dropout(x)

        output = self.transformer_encoder(
            src=x,
            mask=None,
            src_key_padding_mask=src_padding_mask,
        )

        # Return the CLS token / first token in the sequence:
        return output[0, :, :]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 513):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return self.dropout(self.pe[: x.size(0)])


def get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, last_epoch=-1
):
    """Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step >= num_training_steps:  # end_learning_rate=8e-8
            return 0.04
        else:
            return float(num_training_steps - current_step) / float(
                max(1, num_training_steps - num_warmup_steps)
            )

    return LambdaLR(optimizer, lr_lambda, last_epoch)
