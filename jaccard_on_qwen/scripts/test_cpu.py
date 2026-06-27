"""CPU sanity check — 3 questions, greedy + sample + APC(β=0.1) + APC(β=1.0).

Uses very small image resolution (max 256 patches → ~260 tokens) so inference
finishes in a few minutes on CPU. The full GPU runs use max_pixels=1280*28*28.

Verifies:
  - Outputs are semantically meaningful (not gibberish)
  - Output format matches expected JSONL structure
  - APC(β=1.0) ≈ greedy  (deterministic argmax — should match)
  - sample  ≠ greedy      (stochastic)

VCD is skipped — its per-step full forward pass is too slow on CPU.

Usage:
    cd jaccard_on_qwen
    python scripts/test_cpu.py
"""
import json, os, sys, torch, shortuuid
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inference"))

from transformers import AutoProcessor, LogitsProcessorList, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from apc_logits_processor import APCLogitsProcessor

MODEL_PATH    = "/teamspace/lightning_storage/model/Qwen2.5_7b"
DATA_ROOT     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                             "jaccard_experiment", "data", "llava_bench")
QUESTION_FILE = os.path.join(DATA_ROOT, "questions.jsonl")
IMAGE_FOLDER  = os.path.join(DATA_ROOT, "images")

N_QUESTIONS    = 3
MAX_NEW_TOKENS = 48
# Small cap so CPU prefill stays under ~260 tokens total
CPU_MIN_PIXELS = 256 * 28 * 28   # 200,704
CPU_MAX_PIXELS = 256 * 28 * 28   # 200,704  (~256 patches per image)

print(f"Loading Qwen2.5-VL-7B on CPU (bfloat16) ...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype=torch.bfloat16,
    device_map="cpu",
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("Model loaded.\n")

questions = [json.loads(q) for q in open(QUESTION_FILE)][:N_QUESTIONS]
results = []

for line in questions:
    idx  = line["question_id"]
    qs   = line["text"]
    full_image_path = os.path.join(IMAGE_FOLDER, line["image"])

    print(f"{'='*70}")
    print(f"Q{idx}: {qs}")
    print(f"Image: {line['image']}")

    messages = [{"role": "user", "content": [
        {"type": "image", "image": full_image_path,
         "min_pixels": CPU_MIN_PIXELS, "max_pixels": CPU_MAX_PIXELS},
        {"type": "text", "text": qs},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    n_tokens = inputs.input_ids.shape[1]
    print(f"  Input tokens: {n_tokens}")

    def run(label, do_sample, temperature=None, top_p=None, beta=None):
        lp = LogitsProcessorList([APCLogitsProcessor(beta)]) if beta is not None else None
        kw = {"do_sample": do_sample}
        if do_sample:
            kw["temperature"] = temperature
            kw["top_p"]       = top_p
        if lp:
            kw["logits_processor"] = lp
        with torch.inference_mode():
            out = model.generate(
                **inputs, num_beams=1,
                max_new_tokens=MAX_NEW_TOKENS, use_cache=True, **kw
            )
        txt = processor.batch_decode(
            out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0].strip()
        print(f"  [{label:22s}] {txt}")
        return txt

    greedy = run("GREEDY",                do_sample=False)
    sample = run("SAMPLE (temp=1.0)",     do_sample=True, temperature=1.0, top_p=1.0)
    apc01  = run("APC β=0.10",            do_sample=True, temperature=1.0, top_p=1.0, beta=0.1)
    apc1   = run("APC β=1.00 (≈greedy)", do_sample=True, temperature=1.0, top_p=1.0, beta=1.0)

    jsonl_line = json.dumps({
        "question_id": idx, "prompt": qs, "text": greedy,
        "answer_id": shortuuid.uuid(), "model_id": "qwen2.5-vl-7b", "metadata": {}
    })
    print(f"\n  FORMAT CHECK → valid JSONL: {bool(json.loads(jsonl_line))}")
    print(f"  JSONL: {jsonl_line[:120]}...")

    results.append({
        "question_id": idx, "image": line["image"], "question": qs,
        "greedy": greedy, "sample": sample, "apc_0.1": apc01, "apc_1.0": apc1,
        "greedy_eq_apc1":   greedy == apc1,
        "greedy_neq_sample": greedy != sample,
    })
    print()

print("="*70)
print("SUMMARY")
print("="*70)
for r in results:
    print(f"\nQ{r['question_id']} [{r['image']}]")
    print(f"  greedy == apc(β=1.0): {r['greedy_eq_apc1']}  (expect True)")
    print(f"  greedy != sample:     {r['greedy_neq_sample']}  (expect True usually)")
    print(f"  Greedy answer: \"{r['greedy'][:80]}\"")
