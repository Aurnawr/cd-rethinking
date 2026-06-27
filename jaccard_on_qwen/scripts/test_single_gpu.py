"""Quick sanity check — runs 1 question through all 4 decoding modes on GPU.

Usage:
    cd jaccard_on_qwen
    python scripts/test_single_gpu.py
"""
import json
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))

from transformers import AutoProcessor, LogitsProcessorList, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from apc_logits_processor import APCLogitsProcessor
from vcd_logits_processor import VCDLogitsProcessor, add_diffusion_noise

MODEL_PATH = "/teamspace/lightning_storage/model/Qwen2.5_7b"
DATA_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "jaccard_experiment", "data", "llava_bench")
QUESTION_FILE = os.path.join(DATA_ROOT, "questions.jsonl")
IMAGE_FOLDER = os.path.join(DATA_ROOT, "images")

print(f"Loading model from {MODEL_PATH} ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print(f"Model loaded on device: {next(model.parameters()).device}\n")

line = json.loads(open(QUESTION_FILE).readline())
idx = line["question_id"]
qs = line["text"]
full_image_path = os.path.join(IMAGE_FOLDER, line["image"])
print(f"Question: {qs}")
print(f"Image:    {full_image_path}\n")

messages = [{"role": "user", "content": [
    {"type": "image", "image": full_image_path},
    {"type": "text",  "text": qs},
]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                   padding=True, return_tensors="pt").to(model.device)


def generate(extra_kwargs=None, logits_processor=None, label=""):
    lp = LogitsProcessorList(logits_processor) if logits_processor else None
    kw = {"logits_processor": lp} if lp else {}
    kw.update(extra_kwargs or {})
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=64, use_cache=True, **kw)
    txt = processor.batch_decode(out[:, inputs.input_ids.shape[1]:],
                                 skip_special_tokens=True)[0].strip()
    print(f"[{label}] {txt}\n")
    return txt


g  = generate({"do_sample": False, "num_beams": 1}, label="GREEDY")
s  = generate({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "num_beams": 1}, label="SAMPLE")
a  = generate({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "num_beams": 1},
              logits_processor=[APCLogitsProcessor(0.1)], label="APC(β=0.1)")
a1 = generate({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "num_beams": 1},
              logits_processor=[APCLogitsProcessor(1.0)], label="APC(β=1.0, should≈greedy)")

pixel_values_cd = add_diffusion_noise(inputs["pixel_values"].float(), 900).to(model.device, dtype=model.dtype)
v = generate({"do_sample": True, "temperature": 1.0, "top_p": 1.0, "num_beams": 1},
             logits_processor=[VCDLogitsProcessor(model, pixel_values_cd, inputs["image_grid_thw"], 1.0, 0.1)],
             label="VCD")

print("=== Sanity checks ===")
print(f"Greedy == APC(β=1.0): {g == a1}  (should be True or very close)")
print(f"Greedy != Sample:     {g != s}   (should usually be True)")
print("All decoding modes produced output. Ready for full GPU run.")
