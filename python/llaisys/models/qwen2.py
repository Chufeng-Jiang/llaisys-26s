import ctypes
import json
import mmap
import struct
from pathlib import Path
from typing import Sequence

from ..libllaisys import (
	DataType,
	DeviceType,
	LIB_LLAISYS,
	LlaisysQwen2Meta,
)


class Qwen2:

	_SAFETENSORS_DTYPES = {
		"BF16": (DataType.BF16, 2),
		"F16": (DataType.F16, 2),
		"F32": (DataType.F32, 4),
		"I64": (DataType.I64, 8),
		"I32": (DataType.I32, 4),
	}

	def __init__(
		self,
		model_path,
		device: DeviceType = DeviceType.CPU,
	):
		self.model_path = Path(model_path)

		self._model = None
		self._weights_pointer = None
		self._weights = None
		self._weight_tensors = {}
		self._weights_loaded = False

		if not self.model_path.is_dir():
			raise FileNotFoundError(
				f"Model directory does not exist: {self.model_path}"
			)

		config_path = self.model_path / "config.json"

		if not config_path.is_file():
			raise FileNotFoundError(
				f"Qwen2 config.json not found: {config_path}"
			)

		with config_path.open(
			"r",
			encoding="utf-8",
		) as file:
			self.config = json.load(file)

		self.device = DeviceType(device)
		self.device_id = 0

		dtype_name = (
			self.config.get("torch_dtype")
			or self.config.get("dtype")
			or "bfloat16"
		)

		dtype = self._parse_dtype(dtype_name)

		hidden_size = int(
			self.config["hidden_size"]
		)

		num_attention_heads = int(
			self.config["num_attention_heads"]
		)

		if hidden_size % num_attention_heads != 0:
			raise ValueError(
				"hidden_size must be divisible by "
				"num_attention_heads."
			)

		head_dim = int(
			self.config.get("head_dim")
			or hidden_size // num_attention_heads
		)

		eos_token_id = self.config.get(
			"eos_token_id",
			-1,
		)

		if isinstance(eos_token_id, list):
			if not eos_token_id:
				raise ValueError(
					"eos_token_id must not be an empty list."
				)

			eos_token_id = eos_token_id[0]

		self.meta = LlaisysQwen2Meta(
			dtype=int(dtype),
			nlayer=int(
				self.config["num_hidden_layers"]
			),
			hs=hidden_size,
			nh=num_attention_heads,
			nkvh=int(
				self.config["num_key_value_heads"]
			),
			dh=head_dim,
			di=int(
				self.config["intermediate_size"]
			),
			maxseq=int(
				self.config["max_position_embeddings"]
			),
			voc=int(
				self.config["vocab_size"]
			),
			epsilon=float(
				self.config["rms_norm_eps"]
			),
			theta=float(
				self.config.get(
					"rope_theta",
					10000.0,
				)
			),
			end_token=int(eos_token_id),
		)

		self._device_ids = (
			ctypes.c_int * 1
		)(self.device_id)

		try:
			self._create_backend_model()

			self._safetensor_files = sorted(
				self.model_path.glob(
					"*.safetensors"
				)
			)

			# 保留无权重的假模型配置测试能力。
			if self._safetensor_files:
				self._load_weights()

		except Exception:
			self.close()
			raise

	def _create_backend_model(self):
		self._model = (
			LIB_LLAISYS.llaisysQwen2ModelCreate(
				ctypes.byref(self.meta),
				int(self.device),
				self._device_ids,
				1,
			)
		)

		if not self._model:
			raise RuntimeError(
				"Failed to create Qwen2 backend model."
			)

		self._weights_pointer = (
			LIB_LLAISYS.llaisysQwen2ModelWeights(
				self._model
			)
		)

		if not self._weights_pointer:
			raise RuntimeError(
				"Failed to retrieve Qwen2 weights."
			)

		self._weights = (
			self._weights_pointer.contents
		)

	def _parse_dtype(self, dtype_name):
		normalized = str(dtype_name).lower()

		dtype_mapping = {
			"float32": DataType.F32,
			"fp32": DataType.F32,
			"torch.float32": DataType.F32,

			"float16": DataType.F16,
			"fp16": DataType.F16,
			"half": DataType.F16,
			"torch.float16": DataType.F16,

			"bfloat16": DataType.BF16,
			"bf16": DataType.BF16,
			"torch.bfloat16": DataType.BF16,
		}

		if normalized not in dtype_mapping:
			raise ValueError(
				f"Unsupported Qwen2 dtype: {dtype_name}"
			)

		return dtype_mapping[normalized]

	@staticmethod
	def _read_safetensors_header(file_path):
		with file_path.open("rb") as file:
			header_length_bytes = file.read(8)

			if len(header_length_bytes) != 8:
				raise RuntimeError(
					f"Invalid safetensors file: {file_path}"
				)

			header_length = struct.unpack(
				"<Q",
				header_length_bytes,
			)[0]

			header_bytes = file.read(
				header_length
			)

			if len(header_bytes) != header_length:
				raise RuntimeError(
					"Failed to read complete safetensors "
					f"header: {file_path}"
				)

		header = json.loads(
			header_bytes.decode("utf-8")
		)

		return header, header_length

	def _build_tensor_manifest(self):
		manifest = {}

		for file_path in self._safetensor_files:
			header, header_length = (
				self._read_safetensors_header(
					file_path
				)
			)

			for tensor_name, tensor_info in header.items():
				if tensor_name == "__metadata__":
					continue

				if tensor_name in manifest:
					raise RuntimeError(
						"Duplicate tensor in safetensors files: "
						f"{tensor_name}"
					)

				manifest[tensor_name] = {
					"file_path": file_path,
					"header_length": header_length,
					"dtype": tensor_info["dtype"],
					"shape": tensor_info["shape"],
					"data_offsets": tensor_info[
						"data_offsets"
					],
				}

		return manifest

	def _build_weight_targets(self):
		targets = {
			"model.embed_tokens.weight": (
				"in_embed",
				None,
			),
			"lm_head.weight": (
				"out_embed",
				None,
			),
			"model.norm.weight": (
				"out_norm_w",
				None,
			),
		}

		layer_fields = {
			"input_layernorm.weight":
				"attn_norm_w",

			"self_attn.q_proj.weight":
				"attn_q_w",
			"self_attn.q_proj.bias":
				"attn_q_b",

			"self_attn.k_proj.weight":
				"attn_k_w",
			"self_attn.k_proj.bias":
				"attn_k_b",

			"self_attn.v_proj.weight":
				"attn_v_w",
			"self_attn.v_proj.bias":
				"attn_v_b",

			"self_attn.o_proj.weight":
				"attn_o_w",

			"post_attention_layernorm.weight":
				"mlp_norm_w",

			"mlp.gate_proj.weight":
				"mlp_gate_w",
			"mlp.up_proj.weight":
				"mlp_up_w",
			"mlp.down_proj.weight":
				"mlp_down_w",
		}

		for layer_index in range(
			int(self.meta.nlayer)
		):
			prefix = (
				f"model.layers.{layer_index}."
			)

			for suffix, field_name in (
				layer_fields.items()
			):
				targets[prefix + suffix] = (
					field_name,
					layer_index,
				)

		return targets

	@staticmethod
	def _numel(shape):
		result = 1

		for dimension in shape:
			result *= int(dimension)

		return result

	def _create_tensor_from_mmap(
		self,
		mapped,
		tensor_name,
		tensor_info,
	):
		safetensors_dtype = tensor_info["dtype"]

		if safetensors_dtype not in self._SAFETENSORS_DTYPES:
			raise ValueError(
				"Unsupported safetensors dtype "
				f"{safetensors_dtype} for {tensor_name}."
			)

		llaisys_dtype, element_size = (
			self._SAFETENSORS_DTYPES[
				safetensors_dtype
			]
		)

		shape_values = [
			int(value)
			for value in tensor_info["shape"]
		]

		if not shape_values:
			raise ValueError(
				f"Scalar tensor is not supported: {tensor_name}"
			)

		relative_start, relative_end = (
			tensor_info["data_offsets"]
		)

		data_section_start = (
			8 + tensor_info["header_length"]
		)

		absolute_start = (
			data_section_start + relative_start
		)

		absolute_end = (
			data_section_start + relative_end
		)

		actual_byte_count = (
			absolute_end - absolute_start
		)

		expected_byte_count = (
			self._numel(shape_values)
			* element_size
		)

		if actual_byte_count != expected_byte_count:
			raise RuntimeError(
				"Tensor byte-size mismatch for "
				f"{tensor_name}: "
				f"expected={expected_byte_count}, "
				f"actual={actual_byte_count}."
			)

		ShapeArray = (
			ctypes.c_size_t * len(shape_values)
		)

		shape = ShapeArray(
			*shape_values
		)

		tensor = LIB_LLAISYS.tensorCreate(
			shape,
			len(shape_values),
			int(llaisys_dtype),
			int(self.device),
			self.device_id,
		)

		if not tensor:
			raise RuntimeError(
				f"tensorCreate failed for {tensor_name}."
			)

		raw_byte = None
		raw_pointer = None

		try:
			raw_byte = (
				ctypes.c_ubyte.from_buffer(
					mapped,
					absolute_start,
				)
			)

			raw_pointer = ctypes.c_void_p(
				ctypes.addressof(raw_byte)
			)

			LIB_LLAISYS.tensorLoad(
				tensor,
				raw_pointer,
			)

		except Exception:
			LIB_LLAISYS.tensorDestroy(
				tensor
			)

			raise

		finally:
			if raw_pointer is not None:
				del raw_pointer

			if raw_byte is not None:
				del raw_byte

		return tensor

	def _assign_weight(
		self,
		field_name,
		layer_index,
		tensor,
	):
		if layer_index is None:
			setattr(
				self._weights,
				field_name,
				tensor,
			)

			return

		field_pointer = getattr(
			self._weights,
			field_name,
		)

		field_pointer[layer_index] = tensor

	def _load_weights(self):
		manifest = self._build_tensor_manifest()
		targets = self._build_weight_targets()

		missing_weights = [
			tensor_name
			for tensor_name in targets
			if tensor_name not in manifest
		]

		if missing_weights:
			missing_text = "\n".join(
				missing_weights
			)

			raise RuntimeError(
				"Required Qwen2 weights are missing:\n"
				f"{missing_text}"
			)

		targets_by_file = {}

		for tensor_name, target in targets.items():
			tensor_info = manifest[tensor_name]
			file_path = tensor_info["file_path"]

			targets_by_file.setdefault(
				file_path,
				[],
			).append(
				(
					tensor_name,
					target,
					tensor_info,
				)
			)

		total_weights = len(targets)
		loaded_weights = 0

		print(
			f"Loading {total_weights} Qwen2 weights "
			f"to {self.device.name}...",
			flush=True,
		)

		for file_path, file_targets in (
			targets_by_file.items()
		):
			with file_path.open("rb") as file:
				mapped = mmap.mmap(
					file.fileno(),
					length=0,
					access=mmap.ACCESS_COPY,
				)

				try:
					for (
						tensor_name,
						target,
						tensor_info,
					) in file_targets:
						field_name, layer_index = (
							target
						)

						tensor = (
							self._create_tensor_from_mmap(
								mapped,
								tensor_name,
								tensor_info,
							)
						)

						self._assign_weight(
							field_name,
							layer_index,
							tensor,
						)

						self._weight_tensors[
							tensor_name
						] = tensor

						loaded_weights += 1

						if (
							loaded_weights % 25 == 0
							or loaded_weights
							== total_weights
						):
							print(
								"Loaded Qwen2 weights: "
								f"{loaded_weights}/"
								f"{total_weights}",
								flush=True,
							)

				finally:
					mapped.close()

		self._weights_loaded = True

		print(
			"Qwen2 weight loading completed.",
			flush=True,
		)

	def _destroy_weight_tensors(self):
		seen_addresses = set()

		for tensor in self._weight_tensors.values():
			if not tensor:
				continue

			if isinstance(tensor, int):
				address = tensor
			else:
				address = ctypes.cast(
					tensor,
					ctypes.c_void_p,
				).value

			if not address:
				continue

			if address in seen_addresses:
				continue

			seen_addresses.add(address)

			LIB_LLAISYS.tensorDestroy(
				tensor
			)

		self._weight_tensors.clear()
		self._weights_loaded = False

	def close(self):
		model = getattr(
			self,
			"_model",
			None,
		)

		# C++ model only borrows weight handles, so destroy
		# the model before destroying the actual tensors.
		if model:
			LIB_LLAISYS.llaisysQwen2ModelDestroy(
				model
			)

			self._model = None

		self._weights_pointer = None
		self._weights = None

		weight_tensors = getattr(
			self,
			"_weight_tensors",
			None,
		)

		if weight_tensors is not None:
			self._destroy_weight_tensors()

	def __enter__(self):
		return self

	def __exit__(
		self,
		exception_type,
		exception_value,
		traceback,
	):
		self.close()

	def __del__(self):
		try:
			self.close()
		except Exception:
			pass

	def generate(
		self,
		inputs: Sequence[int],
		max_new_tokens: int = None,
		top_k: int = 1,
		top_p: float = 0.8,
		temperature: float = 0.8,
	):
		if not self._weights_loaded:
			raise RuntimeError(
				"Qwen2 weights have not been loaded."
			)

		if not self._model:
			raise RuntimeError(
				"Qwen2 backend model has been closed."
			)

		if top_k != 1:
			raise NotImplementedError(
				"Current Qwen2 implementation only supports "
				"greedy argmax generation with top_k=1."
			)

		if max_new_tokens is None:
			max_new_tokens = 128

		max_new_tokens = int(
			max_new_tokens
		)

		if max_new_tokens < 0:
			raise ValueError(
				"max_new_tokens must not be negative."
			)

		output_tokens = [
			int(token)
			for token in inputs
		]

		if not output_tokens:
			raise ValueError(
				"Qwen2 generation requires at least "
				"one input token."
			)

		if len(output_tokens) > int(
			self.meta.maxseq
		):
			raise ValueError(
				"Qwen2 input exceeds maximum "
				"sequence length."
			)

		LIB_LLAISYS.llaisysQwen2ModelReset(
			self._model
		)

		# 第一次调用处理完整 prompt。
		pending_tokens = output_tokens.copy()

		for _ in range(max_new_tokens):
			if (
				len(output_tokens)
				>= int(self.meta.maxseq)
			):
				break

			TokenArray = (
				ctypes.c_int64
				* len(pending_tokens)
			)

			token_buffer = TokenArray(
				*pending_tokens
			)

			next_token = int(
				LIB_LLAISYS.llaisysQwen2ModelInfer(
					self._model,
					token_buffer,
					len(pending_tokens),
				)
			)

			if next_token < 0:
				raise RuntimeError(
					"Qwen2 backend inference failed."
				)

			output_tokens.append(
				next_token
			)

			if next_token == int(
				self.meta.end_token
			):
				break

			# 后续调用只处理刚刚生成的一个 token。
			pending_tokens = [
				next_token
			]

		return output_tokens