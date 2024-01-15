from pathlib import Path
from typing import Dict, Optional

import torch
from torch import LongTensor, IntTensor, BoolTensor
from torch import nn
from transformers import PretrainedConfig, AutoConfig, BertForPreTraining
from transformers.models.bert.modeling_bert import BertLMPredictionHead


class UvaModel(nn.Module):
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
        Please download the Baidu 12Layer pre-trained language model from:
        and extract it in the model directory: {self.model_directory}
        """
        config = AutoConfig.from_pretrained(path)
        model = BertModel.from_pretrained(path, local_files_only=True, config=config)
        model.to(device)
        torch.compile(model)
        self.model = model
        self.device = device

    def forward(
        self,
        tokens: LongTensor,
        token_types: IntTensor,
        **kwargs,
    ):
        mask = self.mask_attention(tokens, self.special_tokens)
        query_document_embedding = self.model(
            tokens=tokens.to(self.device),
            attention_mask=mask.to(self.device),
            token_types=token_types.to(self.device),
        )
        return query_document_embedding

    @staticmethod
    def mask_attention(tokens: torch.IntTensor, special_token: Dict[str, int]):
        """
        Hugging face attention mask, is True for all tokens that should NOT be masked.
        """
        return tokens > special_token["PAD"]


class BertModel(BertForPreTraining):
    def __init__(self, config: PretrainedConfig):
        super(BertModel, self).__init__(config)

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        **kwargs,
    ):
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            output_hidden_states=True,
            return_dict=True,
        )

        cls = outputs.hidden_states[-1][:, 0]
        return cls
