"""Export a Hugging Face RoBERTa QA model to Core ML (`.mlpackage`).

Inputs:
  - input_ids: int32 tensor of shape [1, seq_len]
  - attention_mask: int32 tensor of shape [1, seq_len]

Outputs:
  - start_logits: float tensor of shape [1, seq_len]
  - end_logits: float tensor of shape [1, seq_len]
"""

import argparse
import platform

import numpy as np
import torch
import coremltools as ct
from transformers import AutoModelForQuestionAnswering, AutoConfig
from coremltools.converters.mil.frontend.torch.torch_op_registry import register_torch_op
from coremltools.converters.mil.frontend.torch.ops import _get_inputs
from coremltools.converters.mil.frontend.torch.utils import NUM_TO_DTYPE_STRING
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil.types.type_mapping import builtin_to_string


def _shape_to_int32(shape):
    # TorchScript sometimes represents "size" as a list of scalar Vars.
    if isinstance(shape, list):
        if len(shape) == 0:
            # size=()  -> scalar tensor => empty shape vector
            shape = mb.const(val=np.array([], dtype=np.int32))
        else:
            shape = mb.concat(values=shape, axis=0)

    # Core ML fill requires int32 shape
    return mb.cast(x=shape, dtype="int32")


def _dtype_str_from_new_factory_inputs(inputs, fallback_dtype_str: str) -> str:
    """
    new_ones/new_zeros TorchScript signature is typically:
      (self, size, dtype, layout, device, pin_memory, requires_grad)
    `dtype` (if present) is a constant integer enum; coremltools provides NUM_TO_DTYPE_STRING.
    """
    if len(inputs) >= 3:
        dt = inputs[2]
        dt_val = getattr(dt, "val", None)
        if dt is not None and dt_val is not None:
            try:
                return NUM_TO_DTYPE_STRING[int(dt_val)]
            except Exception:
                pass
    return fallback_dtype_str


@register_torch_op(torch_alias=["new_ones"], override=True)
def new_ones(context, node):
    inputs = _get_inputs(context, node, min_expected=2)
    base = inputs[0]
    shape = _shape_to_int32(inputs[1])

    fallback = builtin_to_string(base.dtype)
    out_dtype = _dtype_str_from_new_factory_inputs(inputs, fallback)

    out = mb.fill(shape=shape, value=1.0)
    out = mb.cast(x=out, dtype=out_dtype, name=node.name)
    context.add(out)


@register_torch_op(torch_alias=["new_zeros"], override=True)
def new_zeros(context, node):
    inputs = _get_inputs(context, node, min_expected=2)
    base = inputs[0]
    shape = _shape_to_int32(inputs[1])

    fallback = builtin_to_string(base.dtype)
    out_dtype = _dtype_str_from_new_factory_inputs(inputs, fallback)

    out = mb.fill(shape=shape, value=0.0)
    out = mb.cast(x=out, dtype=out_dtype, name=node.name)
    context.add(out)


class RobertaQAWrapper(torch.nn.Module):
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids, attention_mask):
        attention_mask = attention_mask.to(torch.bool)

        out = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=False,
        )
        return out[0], out[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepset/roberta-base-squad2")
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--flexible", action="store_true")
    ap.add_argument("--out", default="RobertaQA.mlpackage")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--skip-sanity-check", action="store_true")
    args = ap.parse_args()

    if args.max_len <= 0:
        ap.error("--max-len must be > 0")
    if not args.out.endswith(".mlpackage"):
        ap.error("--out must end with .mlpackage")

    # 1. Config + Eager Attention
    cfg = AutoConfig.from_pretrained(args.model, torchscript=True)
    hf_model = AutoModelForQuestionAnswering.from_pretrained(
        args.model,
        config=cfg,
        attn_implementation="eager"
    ).eval()

    wrapped = RobertaQAWrapper(hf_model).eval()

    # 2. Fast dummy inputs
    input_ids = torch.zeros((1, args.max_len), dtype=torch.int32)
    attention_mask = torch.ones((1, args.max_len), dtype=torch.int32)

    # 3. Trace and FREEZE the graph
    with torch.inference_mode():
        traced = torch.jit.trace(wrapped, (input_ids, attention_mask), strict=False)
        traced = torch.jit.freeze(traced)

    if args.flexible:
        seq = ct.RangeDim(lower_bound=1, upper_bound=args.max_len, default=args.max_len)
        shape = ct.Shape(shape=(1, seq))
    else:
        shape = (1, args.max_len)

    inputs = [
        ct.TensorType(name="input_ids", shape=shape, dtype=np.int32),
        ct.TensorType(name="attention_mask", shape=shape, dtype=np.int32),
    ]

    convert_kwargs = {
        "compute_precision": (
            ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32
        )
    }

    print("Converting to Core ML...")
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=[ct.TensorType(name="start_logits"), ct.TensorType(name="end_logits")],
        convert_to="mlprogram",
        **convert_kwargs,
    )

    mlmodel.save(args.out)
    print(f"Saved: {args.out}")

    # 4. Sanity check
    if args.skip_sanity_check:
        print("Skipping sanity check (--skip-sanity-check)")
        return

    if platform.system() != "Darwin":
        print("Skipping sanity check: Core ML runtime prediction is only supported on macOS.")
        return

    print("Running sanity check...")
    with torch.inference_mode():
        pt_start, pt_end = wrapped(input_ids, attention_mask)
    pred = mlmodel.predict(
        {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()}
    )

    start_err = np.max(np.abs(pred["start_logits"] - pt_start.numpy()))
    end_err = np.max(np.abs(pred["end_logits"] - pt_end.numpy()))
    print(f"max|start_err|={start_err:.6f}  max|end_err|={end_err:.6f}")


if __name__ == "__main__":
    main()
