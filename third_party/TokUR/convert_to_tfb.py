"""
Convert a base HuggingFace model to a TFB (Training-Free Bayesian) model.

This script computes SVD basis vectors for target attention layers (q_proj, v_proj)
and saves them alongside the original weights, producing a TFB model that can be
loaded directly by the bayesian_transformer package and vLLM.

Usage:
    python convert_to_tfb.py \
        --model-path /path/to/base-model \
        --output-path /path/to/output-tfb-model \
        --architecture llama \
        --rank 8
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil

import safetensors
import torch
from safetensors.torch import save_file


ARCHITECTURE_TO_MODEL_TYPE = {
    "llama": "tfb_llama",
    "qwen2": "tfb_qwen2",
    "phi3": "tfb_llama",
    "opt": "tfb_llama",
}

ARCHITECTURE_TO_VLLM_CLASS = {
    "llama": "VllmTFBLLamaForCausalLM",
    "qwen2": "VllmTFBQwen2ForCausalLM",
    "phi3": "VllmTFBLLamaForCausalLM",
    "opt": "VllmTFBLLamaForCausalLM",
}

# (weight_key_substrings, architecture-specific handler)
TARGET_MODULES = ("q_proj", "v_proj")


def _shard_total(filename: str) -> int | None:
    match = re.search(r"-of-(\d+)\.safetensors$", filename)
    return int(match.group(1)) if match else None


def _weight_shard_files(path: str) -> list[str]:
    files = sorted(
        f for f in os.listdir(path)
        if f.endswith(".safetensors") and _shard_total(f) is not None
    )
    return files


def load_safetensors(path: str):
    """Load all safetensor shard files from a model directory into a single dict."""
    tensor_files = _weight_shard_files(path)
    if not tensor_files:
        raise FileNotFoundError(f"No sharded .safetensors files found in {path}")
    tensors = {}
    for filename in tensor_files:
        filepath = os.path.join(path, filename)
        with safetensors.safe_open(filepath, framework="pt") as f:
            for k in f.keys():
                if k in tensors:
                    raise ValueError(f"Duplicate key {k} found in {filename}")
                tensors[k] = f.get_tensor(k)
    return tensors, tensor_files


def load_pytorch_bins(path: str):
    """Load all pytorch_model*.bin shards into a single dict."""
    tensor_files = sorted(
        f for f in os.listdir(path)
        if f.startswith("pytorch_model") and f.endswith(".bin")
    )
    if not tensor_files:
        raise FileNotFoundError(f"No pytorch_model*.bin files found in {path}")
    tensors = {}
    for filename in tensor_files:
        filepath = os.path.join(path, filename)
        shard = torch.load(filepath, map_location="cpu", weights_only=False)
        for k, v in shard.items():
            if k in tensors:
                raise ValueError(f"Duplicate key {k} found in {filename}")
            tensors[k] = v
    return tensors, tensor_files


def load_model_weights(path: str, architecture: str):
    safetensor_shards = _weight_shard_files(path)
    if safetensor_shards:
        return load_safetensors(path)
    if architecture == "opt":
        return load_pytorch_bins(path)
    single = os.path.join(path, "model.safetensors")
    if os.path.isfile(single):
        with safetensors.safe_open(single, framework="pt") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        return tensors, [single]
    raise FileNotFoundError(f"No supported weight files found in {path}")


def _split_phi3_qkv(weight: torch.Tensor, num_q_heads: int, num_kv_heads: int):
    hidden = weight.shape[1]
    head_dim = hidden // num_q_heads
    q_rows = num_q_heads * head_dim
    kv_rows = num_kv_heads * head_dim
    q = weight[:q_rows]
    v = weight[q_rows + kv_rows : q_rows + kv_rows + kv_rows]
    return q, v


def compute_basis_vectors(model_tensors, rank, bayes_noise="right", architecture="llama"):
    """Compute SVD basis vectors for target modules (q_proj, v_proj)."""
    basis_vectors = {}
    basis_idx = list(range(rank))

    for key, weight in model_tensors.items():
        targets: list[tuple[str, torch.Tensor]] = []

        if architecture == "phi3" and key.endswith("qkv_proj.weight"):
            cfg_hint = weight.shape[1]
            num_q = cfg_hint  # filled from tensor shape below
            # hidden=3072, qkv=[9216,3072] -> num_q_heads inferred via config at call site
            # We infer from weight shape: q_rows = weight.shape[0] // 3 when MHA
            if weight.shape[0] % 3 == 0:
                chunk = weight.shape[0] // 3
                q = weight[:chunk]
                v = weight[2 * chunk : 3 * chunk]
                prefix = key.replace(".qkv_proj.weight", "")
                targets = [
                    (f"{prefix}.q_proj.basis_vectors", q),
                    (f"{prefix}.v_proj.basis_vectors", v),
                ]
        elif architecture == "opt":
            if key.endswith("self_attn.q_proj.weight"):
                prefix = key.replace(".q_proj.weight", "")
                targets = [(f"{prefix}.q_proj.basis_vectors", weight)]
            elif key.endswith("self_attn.v_proj.weight"):
                prefix = key.replace(".v_proj.weight", "")
                targets = [(f"{prefix}.v_proj.basis_vectors", weight)]
        else:
            is_target = any(f".{mod}.weight" in key for mod in TARGET_MODULES)
            if is_target:
                bv_key = key.replace(".weight", ".basis_vectors")
                targets = [(bv_key, weight)]

        for bv_key, mat in targets:
            print(f"  Computing SVD for {bv_key} (shape: {mat.shape})")
            old_dtype = mat.dtype
            u, _, v = torch.svd(mat.float())
            if bayes_noise == "right":
                bv = u[:, basis_idx].to(old_dtype)
            else:
                bv = v[:, basis_idx].to(old_dtype)
            basis_vectors[bv_key] = bv

    return basis_vectors


def save_single_file_model(model_tensors, basis_vectors, output_path):
    """Save combined model as a single safetensors file."""
    combined = {**model_tensors, **basis_vectors}
    output_file = os.path.join(output_path, "model.safetensors")
    print(f"  Saving to {output_file}")
    save_file(combined, output_file, metadata={"format": "safetensors", "framework": "pt"})


def save_sharded_model(model_tensors, basis_vectors, output_path, source_path, source_tensor_files):
    """Save model in sharded format, adding basis vectors as an extra shard."""
    old_total = _shard_total(source_tensor_files[0])
    if old_total is None:
        raise ValueError(f"Cannot infer shard count from {source_tensor_files[0]}")

    for filename in source_tensor_files:
        src = os.path.join(source_path, filename)
        dst = os.path.join(output_path, filename)
        print(f"  Copying {filename}")
        shutil.copy2(src, dst)

    new_total = old_total + 1
    new_shard_name = f"model-{new_total:05d}-of-{new_total:05d}.safetensors"

    for old_filename in source_tensor_files:
        old_path = os.path.join(output_path, old_filename)
        new_filename = re.sub(
            r"-of-\d+\.safetensors$",
            f"-of-{new_total:05d}.safetensors",
            old_filename,
        )
        new_path = os.path.join(output_path, new_filename)
        if old_path != new_path:
            os.rename(old_path, new_path)

    new_shard_path = os.path.join(output_path, new_shard_name)
    print(f"  Saving basis vectors to {new_shard_name}")
    save_file(basis_vectors, new_shard_path, metadata={"format": "safetensors", "framework": "pt"})

    index_path = os.path.join(source_path, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    for key in index_data["weight_map"]:
        old_val = index_data["weight_map"][key]
        index_data["weight_map"][key] = re.sub(
            r"-of-\d+\.safetensors$",
            f"-of-{new_total:05d}.safetensors",
            old_val,
        )

    for key in basis_vectors:
        index_data["weight_map"][key] = new_shard_name

    output_index_path = os.path.join(output_path, "model.safetensors.index.json")
    with open(output_index_path, "w") as f:
        json.dump(index_data, f, indent=4)


def save_opt_as_safetensors(model_tensors, basis_vectors, output_path, source_path):
    """Convert OPT pytorch bins to a single safetensors TFB checkpoint."""
    combined = {**model_tensors, **basis_vectors}
    output_file = os.path.join(output_path, "model.safetensors")
    print(f"  Converting OPT bins to {output_file}")
    save_file(combined, output_file, metadata={"format": "safetensors", "framework": "pt"})


def create_tfb_config(source_path, output_path, architecture, rank, bayes_noise):
    """Create TFB config.json by extending the base model's config."""
    config_path = os.path.join(source_path, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)

    config["model_type"] = ARCHITECTURE_TO_MODEL_TYPE[architecture]
    config["architectures"] = [ARCHITECTURE_TO_VLLM_CLASS[architecture]]
    config["basis_idx"] = list(range(rank))
    config["bayes_noise"] = bayes_noise
    config["bayes_sigma"] = 0.1
    config["sample"] = True
    config["num_samples"] = 10
    config["lowrank"] = True
    config["init_basis_vectors"] = True
    config["use_tfb_for_generation"] = False

    output_config_path = os.path.join(output_path, "config.json")
    with open(output_config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"  Saved config.json with model_type={config['model_type']}")


def copy_tokenizer_files(source_path, output_path):
    """Copy tokenizer and other non-weight files from the source model."""
    skip_extensions = {".safetensors", ".bin", ".pt", ".pth"}
    skip_files = {"config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"}

    for filename in os.listdir(source_path):
        if filename in skip_files:
            continue
        if any(filename.endswith(ext) for ext in skip_extensions):
            continue
        src = os.path.join(source_path, filename)
        if os.path.isfile(src):
            dst = os.path.join(output_path, filename)
            print(f"  Copying {filename}")
            shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a base HuggingFace model to a TFB model"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the base HuggingFace model directory",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save the converted TFB model",
    )
    parser.add_argument(
        "--architecture",
        type=str,
        required=True,
        choices=list(ARCHITECTURE_TO_MODEL_TYPE.keys()),
        help="Model architecture (llama, qwen2, phi3, opt)",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=8,
        help="Rank for low-rank basis vectors (default: 8)",
    )
    parser.add_argument(
        "--bayes-noise",
        type=str,
        default="right",
        choices=["right", "left"],
        help="Noise direction: 'right' uses left singular vectors (U), "
             "'left' uses right singular vectors (V) (default: right)",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"Model path not found: {args.model_path}")

    os.makedirs(args.output_path, exist_ok=True)

    print("Loading model weights...")
    model_tensors, tensor_files = load_model_weights(args.model_path, args.architecture)
    print(f"  Loaded {len(model_tensors)} tensors from {len(tensor_files)} file(s)")

    print("Computing SVD basis vectors...")
    basis_vectors = compute_basis_vectors(
        model_tensors,
        rank=args.rank,
        bayes_noise=args.bayes_noise,
        architecture=args.architecture,
    )
    print(f"  Computed {len(basis_vectors)} basis vector tensors")
    if not basis_vectors:
        raise RuntimeError("No basis vectors computed; check architecture / weight keys")

    print("Saving TFB model weights...")
    if args.architecture == "opt":
        save_opt_as_safetensors(model_tensors, basis_vectors, args.output_path, args.model_path)
    elif len(tensor_files) > 1:
        save_sharded_model(
            model_tensors, basis_vectors, args.output_path,
            args.model_path, tensor_files,
        )
    else:
        save_single_file_model(model_tensors, basis_vectors, args.output_path)

    print("Creating TFB config...")
    create_tfb_config(
        args.model_path, args.output_path, args.architecture,
        args.rank, args.bayes_noise,
    )

    print("Copying tokenizer files...")
    copy_tokenizer_files(args.model_path, args.output_path)

    print(f"\nDone! TFB model saved to {args.output_path}")


if __name__ == "__main__":
    main()
