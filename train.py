import os, math, json, argparse
from dataclasses import dataclass, field
from typing import Optional, List

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType

ATTN_TARGETS = ["q_proj","k_proj","v_proj","o_proj"]
MLP_TARGETS  = ["gate_proj","up_proj","down_proj"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    # data
    p.add_argument("--data_path", type=str, required=True,
                   help="HF dataset path like `HuggingFaceH4/ultrachat_200k` OR local .jsonl")
    p.add_argument("--data_split", type=str, default="train")
    p.add_argument("--data_field", type=str, default="text",
                   help="Column or jsonl key that contains plain text prompts/responses concatenated")
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--packing", type=str, default="true")

    # training
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--num_train_epochs", type=int, default=1)
    p.add_argument("--warmup_ratio", type=float, default=0.03)
    p.add_argument("--save_steps", type=int, default=1000)
    p.add_argument("--log_steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   choices=["cosine","linear","constant","cosine_with_restarts"])
    p.add_argument("--optimizer", type=str, default="paged_adamw_32bit",
                   choices=["adamw_torch","paged_adamw_32bit","adamw_bnb_8bit"])

    # precision & memory
    p.add_argument("--bf16", type=str, default="true")
    p.add_argument("--flash_attn", type=str, default="true")
    p.add_argument("--gradient_checkpointing", type=str, default="true")

    # LoRA
    p.add_argument("--lora_target", type=str, default="attention_only",
                   choices=["attention_only","attn_mlp","custom"])
    p.add_argument("--lora_modules", type=str, default="",
                   help="Comma-separated list of substrings to match module names when --lora_target=custom")
    p.add_argument("--lora_target_ratio", type=float, default=0.01, help="~fraction of base params to train")
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)

    return p.parse_args()

def str2bool(x: str) -> bool:
    return str(x).lower() in ["1","true","t","yes","y"]

def load_text_dataset(path: str, split: str, field: str):
    if path.endswith(".jsonl"):
        ds = load_dataset("json", data_files=path, split="train")
    else:
        ds = load_dataset(path, split=split)
    if field not in ds.column_names:
        # try to map generic formats into a single text column
        def _combine(example):
            # Try common patterns (instruction datasets)
            for key in ["text","content","prompt","instruction","question"]:
                if key in example and isinstance(example[key], str) and example[key].strip():
                    return {"text": example[key]}
            # Fallback: stringify everything
            return {"text": json.dumps(example, ensure_ascii=False)}
        ds = ds.map(_combine, remove_columns=ds.column_names)
        field = "text"
    return ds, field

def pick_lora_targets(model, mode: str, custom_modules: Optional[List[str]]):
    # Inspect module names and pick targets by substring.
    names = []
    for n, _m in model.named_modules():
        names.append(n)

    if mode == "custom" and custom_modules:
        keys = [k.strip() for k in custom_modules if k.strip()]
    elif mode == "attn_mlp":
        keys = ATTN_TARGETS + MLP_TARGETS
    else:
        keys = ATTN_TARGETS

    # Keep unique substrings that actually appear
    found = sorted({k for k in keys for n in names if k in n})
    return found

def estimate_lora_params(model, target_modules: List[str], r: int):
    total = sum(p.numel() for p in model.parameters())
    # Rough estimate: for each linear with weight [out,in], LoRA adds A:[out,r], B:[r,in] → r*(in+out)
    # We approximate by scanning named_modules with weight shapes if available.
    lora_params = 0
    for n, m in model.named_modules():
        if any(t in n for t in target_modules):
            w = getattr(m, "weight", None)
            if w is not None and w.ndim == 2:
                out, in_ = w.shape
                lora_params += r * (in_ + out)
    return total, lora_params

def find_r_for_ratio(model, targets: List[str], target_ratio: float, candidate_rs=(4,8,16,32,48,64,96,128)):
    best_r, best_diff = 16, 1e9
    total = sum(p.numel() for p in model.parameters())
    for r in candidate_rs:
        _, lora_p = estimate_lora_params(model, targets, r)
        ratio = lora_p / total
        diff = abs(ratio - target_ratio)
        if diff < best_diff:
            best_diff, best_r = diff, r
    return best_r

def main():
    args = parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True  # safe & faster on Hopper
    device_map = {"": 0}

    attn_impl = "flash_attention_2" if str2bool(args.flash_attn) else "eager"

    config = AutoConfig.from_pretrained(args.model_name, trust_remote_code=True)
    config._attn_implementation = attn_impl

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16 if str2bool(args.bf16) else torch.float16,
        attn_implementation=attn_impl,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    )

    if str2bool(args.gradient_checkpointing):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # Choose LoRA targets
    custom = [x for x in args.lora_modules.split(",")] if args.lora_target == "custom" else None
    target_modules = pick_lora_targets(model, args.lora_target, custom)

    # Pick r to hit ~target_ratio (default ~1%)
    r = find_r_for_ratio(model, target_modules, args.lora_target_ratio)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )

    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()  # logs % trainable

    # Dataset
    ds, text_field = load_text_dataset(args.data_path, args.data_split, args.data_field)

    # Optimizer choice
    if args.optimizer == "paged_adamw_32bit":
        from transformers import AdamW
        optimizer = None  # let TRL create paged optimizer via accelerate/bnb (requires bitsandbytes)
        optim = "paged_adamw_32bit"
    elif args.optimizer == "adamw_bnb_8bit":
        optim = "adamw_bnb_8bit"
        optimizer = None
    else:
        optim = "adamw_torch"
        optimizer = None

    training_args = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        bf16=str2bool(args.bf16),
        fp16=not str2bool(args.bf16),
        optim=optim,
        lr_scheduler_type=args.lr_scheduler,
        max_seq_length=args.max_seq_len,
        packing=str2bool(args.packing),
        gradient_checkpointing=str2bool(args.gradient_checkpointing),
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=ds,
        dataset_text_field=text_field,
        args=training_args,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
