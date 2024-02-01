from pathlib import Path
from typing import Dict

import flax.linen as nn
import jax
import optax
import rax
from flax.training import checkpoints
from jax import Array
from jax.random import KeyArray
from transformers import FlaxBertForPreTraining
from transformers.models.bert.configuration_bert import BertConfig

import flax.struct
from jax import Array
import jax.numpy as jnp


@flax.struct.dataclass
class Output:
    click: Array
    relevance: Array


@flax.struct.dataclass
class BertOutput:
    logits: Array
    query_document_embedding: Array


@flax.struct.dataclass
class BertLoss:
    loss = jnp.zeros(1)
    mlm_loss = jnp.zeros(1)

    def add(self, losses):
        return self.__class__(
            loss=self.loss + losses.loss,
            mlm_loss=self.mlm_loss + losses.mlm_loss
        )

    def mean(self):
        return self.__class__(
            loss=self.loss.mean(),
            mlm_loss=self.mlm_loss.mean()
        )


@flax.struct.dataclass
class CrossEncoderOutput(BertOutput, Output):
    click: Array
    relevance: Array
    logits: Array
    query_document_embedding: Array


@flax.struct.dataclass
class CrossEncoderLoss(BertLoss):
    loss = jnp.zeros(1)
    mlm_loss = jnp.zeros(1)
    click_loss = jnp.zeros(1)

    def add(self, losses):
        return self.__class__(
            loss=self.loss + losses.loss,
            mlm_loss=self.mlm_loss + losses.mlm_loss,
            click_loss=self.click_loss + losses.click_loss,
        )

    def mean(self):
        return self.__class__(
            loss=self.loss.mean(),
            mlm_loss=self.mlm_loss.mean(),
            click_loss=self.click_loss.mean(),
        )


@flax.struct.dataclass
class PBMCrossEncoderOutput(CrossEncoderOutput):
    click: Array
    relevance: Array
    examination: Array
    logits: Array
    query_document_embedding: Array

class BertModel(FlaxBertForPreTraining):
    """
    Basic BERT model pre-trained only on the MLM task (i.e. RoBERTa setup).
    The model can be further fine-tuned in a CrossEncoder or Condenser setup.
    """

    def __init__(self, config: BertConfig):
        super(BertModel, self).__init__(config)
        self.mlm_loss = optax.softmax_cross_entropy_with_integer_labels
        self.loss_dataclass = BertLoss

    def forward(
        self,
        batch: dict,
        params: dict,
        train: bool,
        **kwargs,
    ) -> BertOutput:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
            deterministic=not train,
            **kwargs,
        )

        return BertOutput(
            logits=outputs.prediction_logits,
            query_document_embedding=outputs.hidden_states[-1][:, 0],
        )

    def init(self, key: KeyArray, batch: dict) -> dict:
        outputs = self.module.apply(
            {"params": self.params},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            return_dict=True,
        )

        return {
            "bert": self.params["bert"],
            "cls": self.params["cls"],
        }

    def get_loss(self, outputs: BertOutput, batch: Dict) -> BertLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)
        return BertLoss(
            loss=mlm_loss,
            mlm_loss=mlm_loss,
        )

    def get_mlm_loss(self, outputs: BertOutput, batch: Dict) -> Array:
        logits = outputs.logits
        labels = batch["labels"]

        # Tokens with label -100 are ignored during the CE computation
        label_mask = jax.numpy.where(labels != -100, 1.0, 0.0)
        loss = self.mlm_loss(logits, labels) * label_mask

        return loss.sum() / label_mask.sum()


class CrossEncoder(BertModel):
    """
    BERT cross-encoder: https://arxiv.org/abs/1910.14424
    Query and document are concatenated in the input. The prediction targets are an MLM
    task and a relevance prediction task using the CLS token. To reproduce the original
    model released by Baidu, we use clicks or annotations as the relevance signal.
    """

    def __init__(self, config: BertConfig):
        super(CrossEncoder, self).__init__(config)
        self.click_head = nn.Dense(1)
        self.loss_dataclass = CrossEncoderLoss

    def forward(
        self,
        batch: Dict,
        params: Dict,
        train: bool,
        **kwargs,
    ) -> CrossEncoderOutput:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
            deterministic=not train,
            **kwargs,
        )

        query_document_embedding = outputs.hidden_states[-1][:, 0]

        click_scores = self.click_head.apply(
            params["click_head"], query_document_embedding
        )

        return CrossEncoderOutput(
            click=click_scores,
            relevance=click_scores,
            logits=outputs.prediction_logits,
            query_document_embedding=query_document_embedding,
        )

    def init(self, key: KeyArray, batch: Dict) -> Dict:
        outputs = self.module.apply(
            {"params": self.params},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
        )

        key, click_key = jax.random.split(key, 2)
        query_document_embedding = outputs.hidden_states[-1][:, 0]
        click_params = self.click_head.init(click_key, query_document_embedding)

        return {
            "bert": self.params["bert"],
            "cls": self.params["cls"],
            "click_head": click_params,
        }

    def get_loss(self, outputs: CrossEncoderOutput, batch: dict) -> CrossEncoderLoss:
        mlm_loss = self.get_mlm_loss(outputs, batch)

        click_loss = rax.pointwise_sigmoid_loss(
            outputs.click.reshape(-1),
            batch["clicks"].reshape(-1),
        ).mean()

        return CrossEncoderLoss(
            loss=mlm_loss + click_loss,
            mlm_loss=mlm_loss,
            click_loss=click_loss,
        )

    def predict_relevance(self, batch: Dict, params: Dict) -> Array:
        outputs = self.module.apply(
            {"params": {"bert": params["bert"], "cls": params["cls"]}},
            input_ids=batch["tokens"],
            attention_mask=batch["attention_mask"],
            token_type_ids=batch["token_types"],
            position_ids=None,
            head_mask=None,
            output_hidden_states=True,
            deterministic=True,
        )
        query_document_embedding = outputs.hidden_states[-1][:, 0]
        click_scores = self.click_head.apply(
            params["click_head"], query_document_embedding
        )
        return click_scores

if __name__ == '__main__':

    config = BertConfig(vocab_size=22_000)
    model = CrossEncoder(config)

    path = Path("/Users/philipphager/Documents/GitHub/baidu-ultr-features/models/uva/bert-12l-12h-mlm_ctr-2m")
    params = checkpoints.restore_checkpoint(ckpt_dir=path, target=None)["params"]

    output = model.forward({
        "tokens": jnp.array([[10, 11, 12, 13]]),
        "attention_mask": jnp.array([[1, 1, 1, 0]]),
        "token_types": jnp.array([[1, 1, 1, 2]]),
    }, params, train=False)

    print(output)
