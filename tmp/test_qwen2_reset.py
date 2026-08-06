from pathlib import Path

from transformers import AutoTokenizer

from llaisys.libllaisys import DeviceType
from llaisys.models.qwen2 import Qwen2
from collections.abc import Mapping

def normalize_input_ids(encoded):
    """
    Convert tokenizer output into a flat list[int].

    Supports:
    - list[int]
    - list[list[int]]
    - BatchEncoding
    - dict-like tokenizer outputs
    - PyTorch / NumPy tensors
    """
    if isinstance(encoded, Mapping):
        if "input_ids" not in encoded:
            raise KeyError(
                "Tokenizer output does not contain input_ids."
            )

        encoded = encoded["input_ids"]

    if hasattr(encoded, "tolist"):
        encoded = encoded.tolist()

    if isinstance(encoded, tuple):
        encoded = list(encoded)

    # Handle batched output: [[token1, token2, ...]]
    if (
        isinstance(encoded, list)
        and len(encoded) == 1
        and isinstance(encoded[0], (list, tuple))
    ):
        encoded = list(encoded[0])

    if not isinstance(encoded, list):
        raise TypeError(
            "Unsupported tokenizer output type: "
            f"{type(encoded)}"
        )

    if any(
        isinstance(token, (list, tuple))
        for token in encoded
    ):
        raise ValueError(
            "Expected one token sequence, but received "
            "multiple or nested sequences."
        )

    return [
        int(token)
        for token in encoded
    ]


def main():
    model_path = Path(
        Path("tmp/model_path.txt").read_text(
            encoding="utf-8"
        ).strip()
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    messages = [
        {
            "role": "user",
            "content": "Who are you?",
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
    )

    input_ids = normalize_input_ids(
        encoded
    )

    print("Prompt token count:", len(input_ids))
    print("Prompt tokens:", input_ids)

    with Qwen2(
        model_path,
        DeviceType.NVIDIA,
    ) as model:
        first = model.generate(
            input_ids,
            max_new_tokens=10,
            top_k=1,
        )

        second = model.generate(
            input_ids,
            max_new_tokens=10,
            top_k=1,
        )

    print("\nFirst generation:")
    print(first)

    print("\nSecond generation:")
    print(second)

    if first != second:
        raise RuntimeError(
            "KV cache reset test failed: "
            "two generations produced different tokens."
        )

    generated_tokens = first[
        len(input_ids):
    ]

    print("\nGenerated tokens:")
    print(generated_tokens)

    print("\nDecoded result:")
    print(
        tokenizer.decode(
            first,
            skip_special_tokens=False,
        )
    )

    print(
        "\nQwen2 KV cache reset test passed."
    )


if __name__ == "__main__":
    main()