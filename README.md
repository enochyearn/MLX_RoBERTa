# RoBERTa: A Robustly Optimized BERT Pretraining in MLX

An implementation of Roberta Question Answering using MLX. 

The design and functionality are inspired by the standards and style of HuggingFace's PyTorch models.

## Downloading and Converting Weights
The script `convert.py` utilizes HuggingFace's transformers library to download model weights, which are then transformed into MLX's compatible format and exported into a singular .npz file.

```
python convert.py \
    --roberta-model deepset/roberta-base-squad2
    --saved_weights_path weights/roberta-base-squad2.npz
```

## Usage
To integrate the `RobertaForQuestionAnswering` model into your own codebase, follow the instructions below for loading the model:

```python
from mlx_roberta import RobertaForQuestionAnswering, load_qa_model, run_qa

question = "Who was Jim Henson?"
passage = "Jim Henson was a nice puppet"

model, tokenizer = load_qa_model("deepset/roberta-base-squad2", "weights/roberta-base-squad2.npz")
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

```


## Export to Core ML (`.mlpackage`)
Use `export_roberta_qa_coreml.py` to export Hugging Face RoBERTa QA (`deepset/roberta-base-squad2`) into Core ML.

The script:
- Loads `AutoModelForQuestionAnswering` with eager attention.
- Traces and freezes a TorchScript wrapper.
- Converts to Core ML `mlprogram`.
- Runs a PyTorch vs Core ML sanity check (`max|start_err|`, `max|end_err|`) on macOS.

Fixed-length export:

```bash
python export_roberta_qa_coreml.py \
  --model deepset/roberta-base-squad2 \
  --max-len 384 \
  --out RobertaQA_384.mlpackage \
  --precision fp16
```

Flexible-length export (1..max_len):

```bash
python export_roberta_qa_coreml.py \
  --model deepset/roberta-base-squad2 \
  --max-len 512 \
  --flexible \
  --out RobertaQA_flex_512.mlpackage \
  --precision fp16
```

Skip sanity check (faster smoke test):

```bash
python export_roberta_qa_coreml.py \
  --model deepset/roberta-base-squad2 \
  --max-len 128 \
  --out RobertaQA_smoke.mlpackage \
  --precision fp16 \
  --skip-sanity-check
```

Useful flags:
- `--precision {fp16,fp32}` (default: `fp16`)
- `--skip-sanity-check`
- `--max-len` must be `> 0`
- `--out` must end with `.mlpackage`

Core ML model I/O:
- Inputs: `input_ids` (`int32`), `attention_mask` (`int32`)
- Outputs: `start_logits`, `end_logits`

## To Dos
- [x] Incorporate a Pooler Layer into the RobertaModel class.
- [ ] Validate the current implementation against the 🤗 `transformers` implementation using Unit Tests.
- [ ] Expand the codebase to include additional models from the RoBERTa family, each catering to different downstream tasks.
- [ ] Extend the existing codebase with capabilities for both training and pre-training.

## Contact Information
For any inquiries, please feel free to reach out via email at `enochyearncontact [at] gmail [dot] com`.
