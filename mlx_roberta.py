import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

from collections import OrderedDict

import numpy as np
import math

from custom.nn.layers.normalization import LayerNormBasselCorrected, LayerNormTorchAlike

from transformers import RobertaTokenizer

# utils
from dataclasses import dataclass


@dataclass
class ModelConfig:
    intermediate_size: int = 3072
    hidden_size: int = 768
    no_heads: int = 12
    hidden_layers: int = 12
    vocab_size: int = 50265
    attention_probs_dropout_prob: float = 0.1
    hidden_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-5
    max_position_embeddings: int = 514
    # QA model's parameters
    num_labels: int = 2
    type_vocab_size: int = 2
    pad_token_id: int = 1
    chunk_size_feed_forward: int = 0


model_configs = {
    "deepset/roberta-base-squad2": ModelConfig(),
    "roberta-base": ModelConfig(),
}

model_types = {
    "deepset/roberta-base-squad2": "qa",
    "roberta-base": "base",
}

class RobertaEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = LayerNormTorchAlike(config.hidden_size, eps=config.layer_norm_eps, correction=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.position_embedding_type = "absolute"

        self.position_ids = mx.broadcast_to(mx.arange(config.max_position_embeddings), (1, config.max_position_embeddings))

        self.token_type_ids = mx.zeros(self.position_ids.shape, dtype=mx.int64)

        self.padding_idx = config.pad_token_id

    def __call__(
            self, 
            input_ids = None,
            token_type_ids = None,
            position_ids = None,):
        
        input_shape = input_ids.shape

        seq_length = input_shape[1]

        # [batch_size, seq_len, hidden_size]
        input_embeds = self.word_embeddings(input_ids)
        
        # padding_idx
        # Boolean Indices not supported
        # input_embeds[input_ids == self.padding_idx] = 0
        
        if position_ids is None:
            if input_ids is not None:
                # # [1, seq_len]
                # position_ids = self.position_ids[:, :seq_length]
                position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx)
        # [1, seq_len, hidden_size]
        position_embeddings = self.position_embeddings(position_ids)
        
        if token_type_ids is None:
            # [batch_size, seq_len]
            token_type_ids = mx.zeros_like(input_ids)

        # [batch_size, seq_len, hidden_size]
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # [batch_size, seq_length, hidden_size] = [batch_size, seq_length, hidden_size] + 
        embeddings = input_embeds + position_embeddings + token_type_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    
    def create_position_ids_from_inputs_embeds(self, input_embeds):
        input_shape = input_embeds.shape[:-1]
        sequence_length = input_shape[1]

        position_ids = mx.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=mx.int64
        )
        return position_ids.unsqueeze(0).expand(input_shape)

class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        # Ensure hidden size is divisible by no. of attention heads
        assert config.hidden_size % config.no_heads == 0

        self.hidden_size = config.hidden_size
        self.no_heads = config.no_heads
        self.head_size = int(self.hidden_size/self.no_heads)
        
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_multi_head(self, x):
        # MultiHead Dimension
        mh_dim = x.shape[:-1] + [self.no_heads, self.head_size]
        x = x.reshape(mh_dim)
        return x.transpose(0, 2, 1, 3)
    def __call__(self, hidden_states, attention_mask = None,):
        query_layer = self.transpose_for_multi_head(self.query(hidden_states))
        key_layer = self.transpose_for_multi_head(self.key(hidden_states))
        value_layer = self.transpose_for_multi_head(self.value(hidden_states))
        
        
        attention_scores = mx.matmul(query_layer, key_layer.transpose(0, 1, 3, 2))

        attention_scores = attention_scores / math.sqrt(self.head_size)

        if attention_mask is not None:
            attention_mask = self.convert_mask_to_additive_causal_mask(attention_mask)
            attention_mask = mx.expand_dims(attention_mask, (1,2))
            attention_mask = mx.broadcast_to(attention_mask, attention_scores.shape)
            attention_scores = attention_scores + attention_mask.astype(attention_scores.dtype)
        
        # Normalize attention scores to probabilities
        attention_probs = mx.softmax(attention_scores, axis=-1)

        attention_probs = self.dropout(attention_probs)

        context_layer = mx.matmul(attention_probs, value_layer)

        context_layer = context_layer.transpose(0, 2, 1, 3)
        
        new_context_layer_shape = context_layer.shape[:-2] + [self.hidden_size]
        context_layer = context_layer.reshape(new_context_layer_shape)
        
        return context_layer
    
    def convert_mask_to_additive_causal_mask(self, mask, dtype = mx.float32):
        mask = mask == 0
        mask = mask.astype(dtype) * -3.4e38
        return mask
        

class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.hidden_dropout_prob = config.hidden_dropout_prob
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
        self.LayerNorm = LayerNormTorchAlike(self.hidden_size, eps=config.layer_norm_eps)
    def __call__(self, hidden_states, input_array):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_array)
        return hidden_states
    
class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.self = BertSelfAttention(config.no_heads, config.hidden_size, config.attention_probs_dropout_prob)
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        # self.pruned_heads = set()

    def prune_heads(self, heads):
        # To Do: Implement Prune Heads
        raise NotImplementedError
    
    def __call__(self, hidden_states, attention_mask):
        self_outputs = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_outputs, hidden_states)
        outputs = (attention_output,)
        return outputs

class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_activation_fn  = nn.GELU()
    def __call__(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_activation_fn(hidden_states)
        return hidden_states

class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNormTorchAlike(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states
    

class RobertaLayer(nn.Module):
    def __init__(self, config, ):
        super().__init__()
        # self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # self.seq_len_dim = 1
 
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
    def __call__(self, hidden_states, attention_mask=None):
        self_attention_outputs = self.attention(hidden_states=hidden_states, attention_mask=attention_mask)
        intermediate_output = self.intermediate(self_attention_outputs[0])
        layer_output = self.output(intermediate_output, self_attention_outputs[0])
        return layer_output

class RobertaEncoder(nn.Module):
    def __init__(self, config,):
        super().__init__()
        self.layer = [
            RobertaLayer(config) for _ in range(config.hidden_layers)
        ]
    def __call__(self, hidden_states, attention_mask=None):
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)
        return hidden_states
        

class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation_fn = mx.tanh
    def __call__(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation_fn(pooled_output)
        return pooled_output

class RobertaModel(nn.Module):
    """
    RoBERTa: A Robustly Optimized BERT Pretraining
    """
    def __init__(self, config, add_pooling_layer=True):
        super().__init__()
        self.embeddings = RobertaEmbeddings(config)
        self.encoder = RobertaEncoder(config)

        self.pooler = RobertaPooler(config) if add_pooling_layer else None

        # To Do: Initialize weights for pre-training
    
    def __call__(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            encoder_hidden_states = None,
            encoder_attention_mask = None,
            past_key_values = None,
            use_cache = None,
            output_attentions = None,
            output_hidden_states = None,
            return_dict = True):
        
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
        )
        encoder_outputs = self.encoder(embedding_output, attention_mask)
        sequence_output = encoder_outputs
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        roberta_out = OrderedDict()
        roberta_out["last_hidden_state"] = sequence_output
        roberta_out["pooler_output"] = pooled_output
        # roberta_out["past_key_values"] = encoder_outputs.past_key_values
        # roberta_out["hidden_states"] = encoder_outputs.hidden_states
        # roberta_out["attentions"] = encoder_outputs.attentions
        # roberta_out["cross_attentions"] = encoder_outputs.cross_attentions
        return roberta_out



class RobertaForQuestionAnswering(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_lables = config.num_labels
        self.hidden_size = config.hidden_size
        
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.qa_outputs = nn.Linear(self.hidden_size, self.num_lables)
        # To Do: Init weights and apply final processing
        

    def __call__(
            self, 
            input_ids = None, 
            attention_mask = None, 
            token_type_ids = None, 
            position_ids = None, 
            head_mask = None, 
            inputs_embeds = None, 
            output_attentions = False, 
            output_hidden_states = False, 
            return_dict=True):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs["last_hidden_state"]

        logits = self.qa_outputs(sequence_output)
        splitted_logits = logits.split(2, axis=-1)
        start_logits, end_logits = splitted_logits[0], splitted_logits[1]
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        qa_out = OrderedDict()
        qa_out["start_logits"] = start_logits
        qa_out["end_logits"] = end_logits
        qa_out["hidden_states"] = sequence_output
        
        return qa_out

def create_position_ids_from_input_ids(input_ids, padding_idx):
    mask = (input_ids != padding_idx).astype(mx.int64)
    # mlx's cumsum isn't working
    incremental_indices = (np.cumsum(mask, axis=1)).astype(np.int64) * mask
    return mx.array(incremental_indices).astype(mx.int64) + padding_idx

def load_base_model(model_name, weights_path):
    weights = mx.load(weights_path)
    weights = tree_unflatten(list(weights.items()))

    # Create and load the model with weights
    model = RobertaModel(model_configs[model_name])
    model.update(weights)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    return model, tokenizer


def load_qa_model(model_name, weights_path):
    weights = mx.load(weights_path)
    weights = tree_unflatten(list(weights.items()))

    # Create and load the model with weights
    model = RobertaForQuestionAnswering(model_configs[model_name])
    model.update(weights)

    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    return model, tokenizer


def run_base(model_name, weights_path, example_text):
    model, tokenizer = load_base_model(model_name, weights_path)
    model.eval()

    tokens = tokenizer(example_text, return_tensors="np", padding=True, truncation=True)
    tokens = {k: mx.array(v) for k, v in tokens.items()}

    outputs = model(**tokens)

    print(f"""MLX Roberta: {outputs["last_hidden_state"]}\n\n""")
    print(f"""MLX Pooled: {outputs["pooler_output"]}""")


def run_qa(model_name, weights_path, question, passage):
    model, tokenizer = load_qa_model(model_name, weights_path)
    model.eval()

    tokens = tokenizer(question, passage, return_tensors="np", padding=True, truncation=True)
    tokens = {k: mx.array(v) for k, v in tokens.items()}

    outputs = model(**tokens)

    # Convert them to 
    answer_start_index = outputs["start_logits"].argmax().tolist()
    answer_end_index = outputs["end_logits"].argmax().tolist()

    predict_answer_tokens = tokens["input_ids"][0, answer_start_index : answer_end_index + 1]
    predict_answer_tokens_np =  np.array(predict_answer_tokens)
    predict_answer = tokenizer.decode(predict_answer_tokens_np, skip_special_tokens=True)
    print(f"Strat Index: {answer_start_index} End Index: {answer_end_index} Answer: {predict_answer}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Roberta QA inference script")
    parser.add_argument(
        "-r",
        "--roberta-model",
        choices=[
            "roberta-base",
            "deepset/roberta-base-squad2",
        ],
        type=str,
        default="deepset/roberta-base-squad2",
        help="The Huggingface name or the path of the Roberta model",
    )
    parser.add_argument(
        "-s",
        "--saved_weights_path",
        type=str,
        default="weights/roberta-base-squad2.npz",
        help="The path of the stored MLX Roberta weights (npz file)."
    )
    parser.add_argument(
        "-q",
        "--question",
        type=str,
        default="Who was Jim Henson?",
        help="The question for the Roberta QA model to ansewr"
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default="Jim Henson was a nice puppet",
        help="The text for running Roberta or the passage for the Roberta QA model to read from and comprehend."
    )
    args = parser.parse_args()

    if "base" in model_types[args.roberta_model]:
        run_base(args.roberta_model, args.saved_weights_path, args.text)
        
    else:
        run_qa(args.roberta_model, args.saved_weights_path, args.question, args.text)

