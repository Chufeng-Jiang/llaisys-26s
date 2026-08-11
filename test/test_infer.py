import argparse
import gc
import io
import os
import sys
import time

import llaisys
import torch

from huggingface_hub import snapshot_download
from test_utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer


sys.stdout = io.TextIOWrapper(
	sys.stdout.buffer,
	encoding="utf-8",
)


MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"


def load_hf_model(
	model_path=None,
	device_name="cpu",
):
	if (
		model_path
		and os.path.isdir(model_path)
	):
		print(
			f"Loading model from local path: "
			f"{model_path}"
		)
	else:
		print(
			f"Loading model from Hugging Face: "
			f"{MODEL_ID}"
		)

		model_path = snapshot_download(
			MODEL_ID
		)

	tokenizer = AutoTokenizer.from_pretrained(
		model_path,
		trust_remote_code=True,
	)

	model = AutoModelForCausalLM.from_pretrained(
		model_path,
		torch_dtype=torch.bfloat16,
		device_map=torch_device(
			device_name
		),
		trust_remote_code=True,
	)

	return (
		tokenizer,
		model,
		model_path,
	)


def hf_infer(
	prompt,
	tokenizer,
	model,
	max_new_tokens=128,
	top_p=0.8,
	top_k=50,
	temperature=0.8,
):
	input_content = tokenizer.apply_chat_template(
		conversation=[
			{
				"role": "user",
				"content": prompt,
			}
		],
		add_generation_prompt=True,
		tokenize=False,
	)

	inputs = tokenizer.encode(
		input_content,
		return_tensors="pt",
	).to(
		model.device
	)

	with torch.no_grad():
		outputs = model.generate(
			inputs,
			max_new_tokens=max_new_tokens,
			top_k=top_k,
			top_p=top_p,
			temperature=temperature,
		)

	result = tokenizer.decode(
		outputs[0],
		skip_special_tokens=True,
	)

	return (
		outputs[0].tolist(),
		result,
	)


def load_llaisys_model(
	model_path,
	device_name,
):
	model = llaisys.models.Qwen2(
		model_path,
		llaisys_device(
			device_name
		),
	)

	return model


def llaisys_infer(
	prompt,
	tokenizer,
	model,
	max_new_tokens=128,
	top_p=0.8,
	top_k=50,
	temperature=0.8,
):
	input_content = tokenizer.apply_chat_template(
		conversation=[
			{
				"role": "user",
				"content": prompt,
			}
		],
		add_generation_prompt=True,
		tokenize=False,
	)

	inputs = tokenizer.encode(
		input_content
	)

	outputs = model.generate(
		inputs,
		max_new_tokens=max_new_tokens,
		top_k=top_k,
		top_p=top_p,
		temperature=temperature,
	)

	return (
		outputs,
		tokenizer.decode(
			outputs,
			skip_special_tokens=True,
		),
	)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	# ============================================================
	# LLAISYS target device
	# ============================================================

	parser.add_argument(
		"--device",
		default="cpu",
		choices=[
			"cpu",
			"nvidia",
			"metax",
		],
		type=str,
		help="Device used by LLAISYS.",
	)

	# ============================================================
	# Hugging Face reference device
	#
	# HF/PyTorch and LLAISYS do not have to use the same backend.
	# In particular:
	#
	#     --device metax
	#
	# can use:
	#
	#     --hf_device cpu
	#
	# as the correctness reference.
	# ============================================================

	parser.add_argument(
		"--hf_device",
		default=None,
		choices=[
			"cpu",
			"nvidia",
		],
		type=str,
		help=(
			"Device used by Hugging Face reference. "
			"If omitted, uses the LLAISYS device for "
			"cpu/nvidia and cpu for metax."
		),
	)

	parser.add_argument(
		"--model",
		default=None,
		type=str,
	)

	parser.add_argument(
		"--prompt",
		default="Who are you?",
		type=str,
	)

	parser.add_argument(
		"--max_steps",
		default=128,
		type=int,
	)

	parser.add_argument(
		"--top_p",
		default=0.8,
		type=float,
	)

	parser.add_argument(
		"--top_k",
		default=50,
		type=int,
	)

	parser.add_argument(
		"--temperature",
		default=1.0,
		type=float,
	)

	parser.add_argument(
		"--test",
		action="store_true",
	)

	args = parser.parse_args()

	# ============================================================
	# Resolve HF reference device
	# ============================================================

	if args.hf_device is not None:
		hf_device_name = (
			args.hf_device
		)
	elif args.device == "metax":
		hf_device_name = "cpu"
	else:
		hf_device_name = (
			args.device
		)

	print(
		"============================================================"
	)

	print(
		"Inference configuration"
	)

	print(
		f"LLAISYS device : "
		f"{args.device}"
	)

	print(
		f"HF device      : "
		f"{hf_device_name}"
	)

	print(
		f"Prompt         : "
		f"{args.prompt}"
	)

	print(
		f"Max new tokens : "
		f"{args.max_steps}"
	)

	print(
		"============================================================"
	)

	top_p = args.top_p
	top_k = args.top_k
	temperature = args.temperature

	# ============================================================
	# Deterministic / greedy correctness mode
	# ============================================================

	if args.test:
		top_p = 1.0
		top_k = 1
		temperature = 1.0

	# ============================================================
	# Hugging Face reference
	# ============================================================

	tokenizer, model, model_path = (
		load_hf_model(
			args.model,
			hf_device_name,
		)
	)

	start_time = time.time()

	tokens, output = hf_infer(
		args.prompt,
		tokenizer,
		model,
		max_new_tokens=args.max_steps,
		top_p=top_p,
		top_k=top_k,
		temperature=temperature,
	)

	end_time = time.time()

	hf_elapsed = (
		end_time
		- start_time
	)

	del model

	gc.collect()

	if (
		hf_device_name == "nvidia"
		and torch.cuda.is_available()
	):
		torch.cuda.empty_cache()

	print(
		"\n=== Hugging Face Reference ===\n"
	)

	print(
		"Tokens:"
	)

	print(
		tokens
	)

	print(
		"\nContents:"
	)

	print(
		output
	)

	print(
		"\n"
	)

	print(
		f"Time elapsed: "
		f"{hf_elapsed:.2f}s\n"
	)

	# ============================================================
	# LLAISYS
	# ============================================================

	print(
		f"Loading LLAISYS model on "
		f"{args.device}..."
	)

	model = load_llaisys_model(
		model_path,
		args.device,
	)

	start_time = time.time()

	llaisys_tokens, llaisys_output = (
		llaisys_infer(
			args.prompt,
			tokenizer,
			model,
			max_new_tokens=args.max_steps,
			top_p=top_p,
			top_k=top_k,
			temperature=temperature,
		)
	)

	end_time = time.time()

	llaisys_elapsed = (
		end_time
		- start_time
	)

	print(
		"\n=== LLAISYS Result ===\n"
	)

	print(
		"Tokens:"
	)

	print(
		llaisys_tokens
	)

	print(
		"\nContents:"
	)

	print(
		llaisys_output
	)

	print(
		"\n"
	)

	print(
		f"Time elapsed: "
		f"{llaisys_elapsed:.2f}s\n"
	)

	# ============================================================
	# Exact token verification
	# ============================================================

	if args.test:
		assert (
			llaisys_tokens
			== tokens
		), (
			"Token mismatch between "
			"Hugging Face and LLAISYS.\n"
			f"HF tokens:\n{tokens}\n"
			f"LLAISYS tokens:\n"
			f"{llaisys_tokens}"
		)

		print(
			"\033[92m"
			"Test passed!"
			"\033[0m\n"
		)