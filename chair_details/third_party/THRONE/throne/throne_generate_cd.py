# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0
#
# NEW FILE — does not modify any existing THRONE or repo file.
# Drop-in replacement for throne_generate.py for cd-rethinking experiments.
# Drives LLaVA_CD (inference/throne_cd_evaluatee.py) without touching
# evaluated_models.py or any other vendored THRONE file.

"""THRONE generation step (step 1) for cd-rethinking contrastive-decoding methods.

Mirrors the structure of throne_generate.py but imports LLaVA_CD directly so no
registry modification is needed.  The THRONE vendored tree is completely unmodified.

Prerequisites
-------------
- ``inference/`` on PYTHONPATH (done by ``scripts/throne_generate.sh``)
- LLaVA-v1.5 checkpoint available locally
- COCO val2017 images + instances_val2017.json (see ``scripts/throne_fetch_coco.sh``)

Single-GPU usage
----------------
    python third_party/THRONE/throne/throne_generate_cd.py \\
        --coco_file data/coco/annotations/instances_val2017.json \\
        --coco_image_dir data/coco/val2017 \\
        --save_path outputs/throne/vcd/responses.json \\
        LLaVA_CD \\
            --model_path /path/to/llava-v1.5-7b \\
            --cd_method vcd

Multi-GPU via torchrun
----------------------
    torchrun --nproc_per_node 8 third_party/THRONE/throne/throne_generate_cd.py \\
        ... same args ...

Output format (identical to throne_generate.py)
------------------------------------------------
    {"prompts": ["Describe this image in detail."],
     "responses": [[prompt_idx, coco_id, caption_text], ...]}
"""

import argparse
import json
import os
from io import BytesIO

import requests
import torch
import torch.distributed as dist
import tqdm
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset, Sampler
from transformers import set_seed

# LLaVA_CD lives in inference/ (added to PYTHONPATH by throne_generate.sh)
from throne_cd_evaluatee import CD_METHODS, LLaVA_CD

GENERATION_PROMPT = "Describe this image in detail."


# ---------------------------------------------------------------------------
# Data-loading helpers (copied from throne_generate.py; that file has a
# module-level ``os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]``
# that runs at import time and would crash a single-GPU launch, so we
# reproduce only what we need here rather than importing from it).
# ---------------------------------------------------------------------------


class InferenceSampler(Sampler):
    """Produce per-rank indices covering every sample exactly once."""

    def __init__(self, size: int, rank: int, world_size: int):
        self._size = size
        assert size > 0
        self._rank = rank
        self._world_size = world_size
        self._local_indices = self._get_local_indices(size, world_size, rank)

    @staticmethod
    def _get_local_indices(total_size, world_size, rank):
        shard_size = total_size // world_size
        left = total_size % world_size
        shard_sizes = [shard_size + int(r < left) for r in range(world_size)]
        begin = sum(shard_sizes[:rank])
        end = min(sum(shard_sizes[: rank + 1]), total_size)
        return range(begin, end)

    def __iter__(self):
        yield from self._local_indices

    def __len__(self):
        return len(self._local_indices)


def init_distributed_mode():
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://")


def load_image(image_file):
    if image_file.startswith("http") or image_file.startswith("https"):
        response = requests.get(image_file)
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(image_file).convert("RGB")


class COCOImageDataset(Dataset):
    def __init__(self, coco_gt: COCO, coco_image_dir, coco_ids, skip_load_image=False):
        self.coco_gt = coco_gt
        self.image_dir = coco_image_dir
        self.coco_ids = sorted(coco_ids)
        self.skip_load_image = skip_load_image

    def __len__(self):
        return len(self.coco_ids)

    def __getitem__(self, index):
        coco_id = self.coco_ids[index]
        image_file = self.coco_gt.loadImgs(coco_id)[0]["file_name"]
        if not self.skip_load_image:
            image = load_image(os.path.join(self.image_dir, image_file))
        else:
            image = os.path.join(self.image_dir, image_file)
        return coco_id, image


def simple_collate(batch):
    total = len(batch[0])
    return [[b[i] for b in batch] for i in range(total)]


CHECKPOINT_EVERY = 100  # save partial results every N images


def _load_checkpoint(ckpt_path):
    """Return (done_ids_set, responses_list) from a checkpoint file, or (set(), [])."""
    if os.path.exists(ckpt_path):
        try:
            with open(ckpt_path) as f:
                data = json.load(f)
            responses = data.get("responses", [])
            done_ids = {int(r[1]) for r in responses}
            print(f"[checkpoint] Resuming from {ckpt_path} — {len(done_ids)} images already done.")
            return done_ids, responses
        except Exception as e:
            print(f"[checkpoint] Warning: could not read checkpoint ({e}), starting fresh.")
    return set(), []


def _save_checkpoint(ckpt_path, responses):
    """Atomically write a checkpoint so a crash never corrupts the file."""
    tmp = ckpt_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"prompts": [GENERATION_PROMPT], "responses": responses}, f)
    os.replace(tmp, ckpt_path)


def batch_generate(model, generation_prompt, image_dataloader, rank=0,
                   checkpoint_path=None, existing_responses=None):
    response_tuple = list(existing_responses) if existing_responses else []
    gen_prompt_idx = 0
    input_ids = model.tokenize_prompt(generation_prompt)
    iterable = tqdm.tqdm(
        image_dataloader,
        desc="generating",
        dynamic_ncols=True,
        initial=len(response_tuple),
        total=len(response_tuple) + len(image_dataloader.dataset),
    ) if rank == 0 else image_dataloader
    for _idx, (indices, imgs) in enumerate(iterable):
        batch_outputs = model.generate_batch(input_ids, imgs)
        response_tuple += list(zip([gen_prompt_idx] * len(indices), indices, batch_outputs))
        # checkpoint every N images (rank-0 only in single-GPU mode)
        if checkpoint_path and rank == 0 and len(response_tuple) % CHECKPOINT_EVERY == 0:
            _save_checkpoint(checkpoint_path, response_tuple)
    return response_tuple


# ---------------------------------------------------------------------------
# Main generation driver
# ---------------------------------------------------------------------------


def generate_responses(args):
    save_dir, save_file = os.path.split(args.save_path)
    save_file, _ext = os.path.splitext(save_file)
    os.makedirs(save_dir, exist_ok=True)

    final_path = os.path.join(save_dir, f"{save_file}.json")
    ckpt_path = os.path.join(save_dir, f"{save_file}.ckpt.json")

    # Skip entirely if the final output already exists
    if os.path.exists(final_path):
        print(f"[skip] {final_path} already exists — skipping generation.")
        return

    model = LLaVA_CD(args)
    generation_prompt = model.format_prompt(GENERATION_PROMPT)
    model.load()

    coco_gt = COCO(args.coco_file)
    if not args.coco_subset:
        all_coco_ids = sorted(coco_gt.getImgIds())
    else:
        with open(args.coco_subset) as f:
            all_coco_ids = sorted(int(x.strip()) for x in f if x.strip())

    # Load checkpoint — skip already-processed IDs
    done_ids, existing_responses = _load_checkpoint(ckpt_path)
    coco_ids = [cid for cid in all_coco_ids if cid not in done_ids]
    print(f"[info] {len(done_ids)} done, {len(coco_ids)} remaining out of {len(all_coco_ids)} total.")

    if not coco_ids:
        # All done — write final file from checkpoint
        results = dict(prompts=[GENERATION_PROMPT], responses=existing_responses)
        with open(final_path, "w") as f:
            json.dump(results, f)
        print(f"[done] All images already processed. Wrote {final_path}")
        return

    dataset = COCOImageDataset(
        coco_gt=coco_gt,
        coco_image_dir=args.coco_image_dir,
        coco_ids=coco_ids,
        skip_load_image=False,
    )

    ddp = "RANK" in os.environ
    if not ddp:
        image_loader = DataLoader(
            dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=simple_collate,
            drop_last=False,
        )
        outputs = batch_generate(
            model, generation_prompt, image_dataloader=image_loader,
            checkpoint_path=ckpt_path, existing_responses=existing_responses,
        )
        results = dict(prompts=[GENERATION_PROMPT], responses=outputs)
        with open(final_path, "w") as f:
            json.dump(results, f)
        # Clean up checkpoint once final file is written
        if os.path.exists(ckpt_path):
            os.remove(ckpt_path)
        print(f"[done] Wrote {final_path}")
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["LOCAL_RANK"]
        init_distributed_mode()
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        dist.barrier()

        sampler = InferenceSampler(len(dataset), world_size=world_size, rank=rank)
        image_loader = DataLoader(
            dataset,
            batch_size=args.per_device_batch_size,
            shuffle=False,
            collate_fn=simple_collate,
            sampler=sampler,
            drop_last=False,
        )
        outputs = batch_generate(
            model, generation_prompt, image_dataloader=image_loader, rank=rank
        )
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{save_file}_{rank}_{world_size}.json"), "w") as f:
            json.dump(outputs, f)
        dist.barrier()
        if rank == 0:
            print("grouping worker jsons …")
            combined_outputs = []
            for worker in range(world_size):
                with open(
                    os.path.join(save_dir, f"{save_file}_{worker}_{world_size}.json")
                ) as f:
                    combined_outputs += json.load(f)
            results = dict(prompts=[GENERATION_PROMPT], responses=combined_outputs)
            with open(os.path.join(save_dir, f"{save_file}.json"), "w") as f:
                json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="THRONE step 1: generate free-form captions with LLaVA + CD method."
    )
    parser.add_argument(
        "--coco_file", type=str, required=True,
        help="Path to COCO annotations (instances_val2017.json or instances_train2017.json)"
    )
    parser.add_argument(
        "--coco_image_dir", type=str, required=True,
        help="Directory containing the COCO images (val2017/)"
    )
    parser.add_argument(
        "--coco_subset", type=str, default=None,
        help="Optional text file with COCO image IDs to evaluate (one per line)"
    )
    parser.add_argument(
        "--save_path", type=str, required=True,
        help="Output JSON path, e.g. outputs/throne/vcd/responses.json"
    )
    parser.add_argument(
        "--per_device_batch_size", type=int, default=1,
        help="Images processed per GPU per step. Default 1 preserves per-image CD math."
    )
    parser.add_argument("--seed", type=int, default=42)

    subparsers = parser.add_subparsers(help="Model subcommand")
    parser_cd = subparsers.add_parser(
        "LLaVA_CD",
        help="LLaVA-v1.5 with contrastive-decoding method"
    )
    parser_cd.set_defaults(modelclass="LLaVA_CD")
    parser_cd.add_argument(
        "--model_path", type=str, required=True,
        help="Local path or HF hub ID for the LLaVA-v1.5 checkpoint"
    )
    parser_cd.add_argument(
        "--model_base", type=str, default=None,
        help="Base model path (only needed for LoRA adapter checkpoints)"
    )
    parser_cd.add_argument(
        "--conv_template_name", type=str, default="vicuna_v1",
        help="Conversation template name (llava-v1.5 default: vicuna_v1)"
    )
    parser_cd.add_argument(
        "--cd_method", type=str, default="none", choices=list(CD_METHODS),
        help=(
            "Contrastive decoding method.  "
            "none=standard greedy/sampling baseline; "
            "vcd=Visual Contrastive Decoding; "
            "icd=Instruction Contrastive Decoding; "
            "sid=Self-Introspective Decoding; "
            "apc=Adaptive Plausibility Constraint sampling."
        )
    )
    parser_cd.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (0 = greedy)"
    )
    parser_cd.add_argument("--top_p", type=float, default=None)
    parser_cd.add_argument(
        "--noise_step", type=int, default=900,
        help="DDPM noise step for VCD (ignored for other methods)"
    )

    args = parser.parse_args()
    if not hasattr(args, "modelclass"):
        parser.error("Subcommand required: LLaVA_CD")
    set_seed(args.seed)
    generate_responses(args)
