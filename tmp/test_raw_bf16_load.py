import ctypes
import json
import mmap
import struct
from pathlib import Path

import numpy as np

from llaisys.libllaisys import (
	DataType,
	DeviceType,
	LIB_LLAISYS,
)


MODEL_PATH_FILE = Path("tmp/model_path.txt")
TENSOR_NAME = "model.layers.0.input_layernorm.weight"
TEST_ELEMENT_COUNT = 8
BF16_ELEMENT_SIZE = 2


def read_safetensors_header(file_path: Path):
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

		header_bytes = file.read(header_length)

		if len(header_bytes) != header_length:
			raise RuntimeError(
				"Failed to read the complete safetensors header."
			)

	header = json.loads(
		header_bytes.decode("utf-8")
	)

	return header, header_length


def decode_bf16(raw_uint16: np.ndarray):
	"""
	Convert BF16 bit patterns to float32 for display only.

	The original BF16 bytes are not modified.
	"""
	float32_bits = (
		raw_uint16.astype(np.uint32) << 16
	)

	return float32_bits.view(np.float32)


def main():
	if not MODEL_PATH_FILE.is_file():
		raise FileNotFoundError(
			f"Model path file not found: {MODEL_PATH_FILE}"
		)

	model_path = Path(
		MODEL_PATH_FILE.read_text(
			encoding="utf-8"
		).strip()
	)

	if not model_path.is_dir():
		raise FileNotFoundError(
			f"Model directory not found: {model_path}"
		)

	tensor_file = model_path / "model.safetensors"

	if not tensor_file.is_file():
		raise FileNotFoundError(
			f"Safetensors file not found: {tensor_file}"
		)

	header, header_length = read_safetensors_header(
		tensor_file
	)

	if TENSOR_NAME not in header:
		raise KeyError(
			f"Tensor not found: {TENSOR_NAME}"
		)

	tensor_info = header[TENSOR_NAME]

	tensor_dtype = tensor_info["dtype"]
	tensor_shape = tensor_info["shape"]
	relative_start, relative_end = (
		tensor_info["data_offsets"]
	)

	print("Tensor name:", TENSOR_NAME)
	print("Tensor dtype:", tensor_dtype)
	print("Tensor shape:", tensor_shape)
	print(
		"Data offsets:",
		tensor_info["data_offsets"],
	)

	if tensor_dtype != "BF16":
		raise RuntimeError(
			f"Expected BF16 tensor, got {tensor_dtype}."
		)

	if TEST_ELEMENT_COUNT <= 0:
		raise ValueError(
			"TEST_ELEMENT_COUNT must be greater than zero."
		)

	total_elements = int(
		np.prod(tensor_shape)
	)

	if TEST_ELEMENT_COUNT > total_elements:
		raise ValueError(
			"TEST_ELEMENT_COUNT exceeds the tensor size."
		)

	# Safetensors data_offsets are relative to the beginning
	# of the data section.
	data_section_start = 8 + header_length

	absolute_start = (
		data_section_start + relative_start
	)

	absolute_end = (
		data_section_start + relative_end
	)

	full_byte_count = (
		absolute_end - absolute_start
	)

	expected_byte_count = (
		total_elements * BF16_ELEMENT_SIZE
	)

	print("Full tensor bytes:", full_byte_count)
	print("Expected bytes:", expected_byte_count)

	if full_byte_count != expected_byte_count:
		raise RuntimeError(
			"Safetensors byte size does not match "
			"the BF16 tensor shape."
		)

	test_byte_count = (
		TEST_ELEMENT_COUNT * BF16_ELEMENT_SIZE
	)

	tensor = None

	with tensor_file.open("rb") as file:
		# ACCESS_COPY creates a private writable mapping.
		# Changes would not be written back to the model file.
		mapped = mmap.mmap(
			file.fileno(),
			length=0,
			access=mmap.ACCESS_COPY,
		)

		try:
			# Copy the first BF16 values only for verification.
			# Because of .copy(), this NumPy array does not retain
			# a reference to the mmap object.
			raw_uint16 = np.frombuffer(
				mapped,
				dtype="<u2",
				count=TEST_ELEMENT_COUNT,
				offset=absolute_start,
			).copy()

			decoded_float32 = decode_bf16(
				raw_uint16
			)

			print("Raw BF16 bits:", raw_uint16)
			print(
				"Decoded values:",
				decoded_float32,
			)

			shape = (
				ctypes.c_size_t * 1
			)(TEST_ELEMENT_COUNT)

			tensor = LIB_LLAISYS.tensorCreate(
				shape,
				1,
				int(DataType.BF16),
				int(DeviceType.CPU),
				0,
			)

			if not tensor:
				raise RuntimeError(
					"tensorCreate returned a null pointer."
				)

			# Obtain the address of the first byte of the selected
			# safetensors data without copying the full tensor.
			raw_byte = ctypes.c_ubyte.from_buffer(
				mapped,
				absolute_start,
			)

			raw_pointer = ctypes.c_void_p(
				ctypes.addressof(raw_byte)
			)

			try:
				# tensorLoad copies exactly the number of bytes
				# required by the destination tensor.
				LIB_LLAISYS.tensorLoad(
					tensor,
					raw_pointer,
				)

				print("\nLLAISYS BF16 tensor:")
				LIB_LLAISYS.tensorDebug(
					tensor
				)

			finally:
				# Both ctypes objects must be deleted before
				# mmap.close(), otherwise mmap reports:
				# BufferError: cannot close exported pointers exist
				del raw_pointer
				del raw_byte

		finally:
			if tensor:
				LIB_LLAISYS.tensorDestroy(
					tensor
				)

				tensor = None

			mapped.close()

	print("\nRaw BF16 loading test passed.")


if __name__ == "__main__":
	main()