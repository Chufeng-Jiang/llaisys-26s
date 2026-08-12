import json
import struct
from pathlib import Path

model_path = Path(Path("tmp/model_path.txt").read_text(encoding="utf-8").strip())

print("Model path:")
print(model_path)

config_path = model_path / "config.json"

with config_path.open(
    "r",
    encoding="utf-8",
) as file:
    config = json.load(file)

print("\n===== Important config fields =====")

config_keys = [
    "model_type",
    "architectures",
    "torch_dtype",
    "dtype",
    "num_hidden_layers",
    "hidden_size",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "intermediate_size",
    "max_position_embeddings",
    "vocab_size",
    "rms_norm_eps",
    "rope_theta",
    "eos_token_id",
    "bos_token_id",
    "tie_word_embeddings",
    "attention_bias",
]

for key in config_keys:
    print(f"{key}: {config.get(key)}")


def read_safetensors_header(file_path):
    with file_path.open("rb") as file:
        header_size_bytes = file.read(8)

        if len(header_size_bytes) != 8:
            raise RuntimeError(f"Invalid safetensors file: {file_path}")

        header_size = struct.unpack(
            "<Q",
            header_size_bytes,
        )[0]

        header_bytes = file.read(header_size)

    return json.loads(header_bytes.decode("utf-8"))


safetensor_files = sorted(model_path.glob("*.safetensors"))

print("\n===== Safetensors files =====")
print("File count:", len(safetensor_files))

for file_path in safetensor_files:
    print(file_path.name)

all_tensors = {}

for file_path in safetensor_files:
    header = read_safetensors_header(file_path)

    for name, info in header.items():
        if name == "__metadata__":
            continue

        if name in all_tensors:
            raise RuntimeError(f"Duplicate tensor name: {name}")

        all_tensors[name] = {
            "file": file_path.name,
            "dtype": info["dtype"],
            "shape": info["shape"],
        }

print("\n===== Tensor summary =====")
print("Tensor count:", len(all_tensors))

print("\n===== Global weights =====")

global_names = [
    "model.embed_tokens.weight",
    "model.norm.weight",
    "lm_head.weight",
]

for name in global_names:
    print(
        name,
        "->",
        all_tensors.get(name),
    )

print("\n===== Layer 0 weights =====")

layer_zero_prefix = "model.layers.0."

for name in sorted(all_tensors):
    if name.startswith(layer_zero_prefix):
        info = all_tensors[name]

        print(f"{name}: dtype={info['dtype']}, shape={info['shape']}, file={info['file']}")

layer_indices = set()

for name in all_tensors:
    prefix = "model.layers."

    if not name.startswith(prefix):
        continue

    remainder = name[len(prefix) :]
    layer_text = remainder.split(".", 1)[0]

    if layer_text.isdigit():
        layer_indices.add(int(layer_text))

print("\n===== Layer checks =====")
print("Layer indices:", sorted(layer_indices))
print("Layer count:", len(layer_indices))

expected_layer_count = int(config["num_hidden_layers"])

if len(layer_indices) != expected_layer_count:
    raise RuntimeError(f"Layer count mismatch: config={expected_layer_count}, weights={len(layer_indices)}")

required_suffixes = [
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_proj.bias",
    "self_attn.k_proj.weight",
    "self_attn.k_proj.bias",
    "self_attn.v_proj.weight",
    "self_attn.v_proj.bias",
    "self_attn.o_proj.weight",
    "post_attention_layernorm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
]

print("\n===== Missing required weights =====")

missing = []

for layer_index in range(expected_layer_count):
    for suffix in required_suffixes:
        name = f"model.layers.{layer_index}.{suffix}"

        if name not in all_tensors:
            missing.append(name)

if missing:
    for name in missing:
        print(name)
else:
    print("None")

print("\nModel manifest inspection passed.")
