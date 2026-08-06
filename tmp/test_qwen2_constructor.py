from llaisys.models.qwen2 import Qwen2
from llaisys.libllaisys import DeviceType


model = Qwen2(
	"tmp/fake_qwen2",
	DeviceType.CPU,
)

print("Qwen2 Python constructor passed.")
print("Number of layers:", model.meta.nlayer)
print("Hidden size:", model.meta.hs)
print("Attention heads:", model.meta.nh)
print("KV heads:", model.meta.nkvh)
print("Head dimension:", model.meta.dh)
print("Vocabulary size:", model.meta.voc)
print("EOS token:", model.meta.end_token)
print("Safetensor files:", model._safetensor_files)

del model

print("Qwen2 model destroyed.")
