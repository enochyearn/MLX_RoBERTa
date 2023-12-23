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


## To Dos
- [x] Incorporate a Pooler Layer into the RobertaModel class.
- [ ] Validate the current implementation against the 🤗 `transformers` implementation using Unit Tests.
- [ ] Expand the codebase to include additional models from the RoBERTa family, each catering to different downstream tasks.
- [ ] Extend the existing codebase with capabilities for both training and pre-training.

## Contact Information
For any inquiries, please feel free to reach out via email at `enochyearncontact [at] gmail [dot] com`.
