from pathlib import Path
from typing import Dict, Optional

import torch
from torch import LongTensor, IntTensor, BoolTensor, FloatTensor
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import BertModel, PretrainedConfig, AutoConfig, BertForPreTraining
from transformers.models.bert.modeling_bert import BertLMPredictionHead

from src.const import UVA_SPECIAL_TOKENS


class UvaModel(nn.Module):
    def __init__(self, model_directory: str, model_name: str):
        super().__init__()
        self.model_directory = Path(model_directory)
        self.model_name = model_name
        self.model = None
        self.device = None

    def load(self, device):
        path = self.model_directory / self.model_name
        assert path.exists(), f"""
        Please download the Baidu 12Layer pre-trained language model from:
        # FIXME!
        and extract it in the model directory: {self.model_directory}
        """
        config = AutoConfig.from_pretrained(path)
        model = CrossEncoder.from_pretrained(path, local_files_only=True, config=config)
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
        mask = self.mask_attention(tokens, UVA_SPECIAL_TOKENS)
        _, query_document_embedding = self.model(
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
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

    def __init__(self, config: PretrainedConfig):
        super(BertModel, self).__init__(config)
        self.mlm_head = BertLMPredictionHead(config)
        self.mlm_loss = CrossEntropyLoss()

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        **kwargs,
    ):
        loss = 0
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )
        query_document_embedding = outputs.pooler_output

        if labels is not None:
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, query_document_embedding

    def get_mlm_loss(self, sequence_output: FloatTensor, labels: LongTensor):
        token_scores = self.mlm_head(sequence_output)

        return self.mlm_loss(
            token_scores.view(-1, self.config.vocab_size),
            labels.view(-1),
        )


class CrossEncoder(BertModel):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. To reproduce the original
    model released by Baidu, we use clicks or annotations as the relevance signal.
    """

    def __init__(self, config: PretrainedConfig):
        super(CrossEncoder, self).__init__(config)
        self.click_head = nn.Linear(config.hidden_size, 1)
        self.click_loss = nn.BCEWithLogitsLoss()

    def forward(
        self,
        tokens: LongTensor,
        attention_mask: BoolTensor,
        token_types: IntTensor,
        labels: Optional[LongTensor] = None,
        clicks: Optional[FloatTensor] = None,
        **kwargs,
    ):
        loss = 0
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            token_type_ids=token_types,
            return_dict=True,
        )
        query_document_embedding = outputs.pooler_output

        if clicks is not None:
            click_scores = self.click_head(query_document_embedding).squeeze()
            loss += self.click_loss(click_scores, clicks)

        if labels is not None:
            loss += self.get_mlm_loss(outputs[0], labels)

        return loss, query_document_embedding
