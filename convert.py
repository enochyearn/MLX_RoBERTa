from transformers import RobertaForQuestionAnswering, RobertaModel

import argparse
import numpy as np

# Utils
from pathlib import Path

model_types = {
    "roberta-base": RobertaModel,
    "roberta-base-qa": RobertaForQuestionAnswering,
}

# No keys are needed to be replaced at the moment
def replace_key(key):
    return key

def convert(model_path_or_name, saved_weights_path, model_type):
    transformers_model = model_types[model_type]
    model = transformers_model.from_pretrained(model_path_or_name)
    tensors = {
        replace_key(key): tensor.numpy() for key, tensor in model.state_dict().items()
    }
    Path(saved_weights_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(saved_weights_path, **tensors)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Roberta weights to MLX.")
    parser.add_argument(
        "-r",
        "--roberta-model",
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
        "-m",
        "--model_type",
        choices=[
            "roberta-base",
            "roberta-base-qa",
        ],
        type=str,
        default="roberta-base-qa",
        help="The type of task that Roberta is used for."
    )
    
    
    args = parser.parse_args()

    convert(args.roberta_model, args.saved_weights_path, args.model_type)

