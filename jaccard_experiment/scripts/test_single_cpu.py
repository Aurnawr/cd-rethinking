"""
Quick CPU sanity check: runs 1 question through the greedy inference pipeline.
Uses max_new_tokens=20 so it finishes in a reasonable time on CPU.
"""

import sys
import os

# Run from jaccard_experiment/ directory
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CD_RETHINK = os.path.dirname(REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "inference"))
sys.path.insert(0, os.path.join(CD_RETHINK, "LLava1.5-7B"))

import torch
import json
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

MODEL_PATH = "/teamspace/lightning_storage/model/llava-v1.5-7b"
QUESTION_FILE = os.path.join(REPO_ROOT, "data/llava_bench/questions.jsonl")
IMAGE_FOLDER  = os.path.join(REPO_ROOT, "data/llava_bench/images")

print("Loading model on CPU (this may take ~1-2 minutes) ...")
disable_torch_init()

# load_pretrained_model respects device_map; "cpu" forces CPU
tokenizer, model, image_processor, context_len = load_pretrained_model(
    MODEL_PATH,
    model_base=None,
    model_name=get_model_name_from_path(MODEL_PATH),
    device_map="cpu",
    device="cpu",
)
model.eval()
print("Model loaded.\n")

# Take only the first question
with open(QUESTION_FILE) as f:
    line = json.loads(f.readline())

idx        = line["question_id"]
image_file = line["image"]
qs         = line["text"]
category   = line.get("category", "")

print(f"Question ID : {idx}")
print(f"Image       : {image_file}")
print(f"Category    : {category}")
print(f"Question    : {qs}\n")

# Build prompt
qs_with_img = DEFAULT_IMAGE_TOKEN + '\n' + qs
conv = conv_templates["vicuna_v1"].copy()
conv.append_message(conv.roles[0], qs_with_img)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)

image = Image.open(os.path.join(IMAGE_FOLDER, image_file)).convert('RGB')
image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0].unsqueeze(0)

stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2

print("Running greedy generation (max_new_tokens=20) ...")
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=False,
        num_beams=1,
        max_new_tokens=20,
        use_cache=True,
    )

input_token_len = input_ids.shape[1]
outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0].strip()
if outputs.endswith(stop_str):
    outputs = outputs[:-len(stop_str)].strip()

print(f"\nModel output: {outputs}")
print("\nSanity check PASSED — pipeline works correctly.")
