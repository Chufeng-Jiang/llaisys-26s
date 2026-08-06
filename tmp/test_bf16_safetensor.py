from pathlib import Path

import safetensors


model_path = Path(
	Path("tmp/model_path.txt").read_text(
		encoding="utf-8"
	).strip()
)

tensor_path = model_path / "model.safetensors"

with safetensors.safe_open(
	tensor_path,
	framework="numpy",
	device="cpu",
) as data:
	name = "model.layers.0.input_layernorm.weight"
	array = data.get_tensor(name)

	print("Name:", name)
	print("Python type:", type(array))
	print("dtype:", array.dtype)
	print("dtype name:", array.dtype.name)
	print("shape:", array.shape)
	print("strides:", array.strides)
	print("itemsize:", array.itemsize)
	print("nbytes:", array.nbytes)
	print("C contiguous:", array.flags.c_contiguous)
	print("First values:", array[:8])

	print("Array interface:")
	print(array.__array_interface__)
