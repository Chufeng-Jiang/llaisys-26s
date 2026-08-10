git push origin Cuda-Compatible-Kernel


find src test \
    \( -name "*.c" -o -name "*.cc" -o -name "*.cpp" \
       -o -name "*.h" -o -name "*.hpp" \
       -o -name "*.cu" -o -name "*.cuh" \) \
    -exec clang-format -i {} +

find src test -name "*.py" -exec black {} +

Add       ✅
SwiGLU    ✅
   ↓
Embedding ← next
   ↓
RoPE
   ↓
RMSNorm
   ↓
Argmax
   ↓
Linear
   ↓
SelfAttention
   ↓
Full Qwen

--------------------------------------------------
cd ~/Desktop/InfiniTensor/llaisys-26s
rm -rf .xmake
rm -rf build
xmake f -c \
    --nv-gpu=y \
    -cv

xmake -r -vD 2>&1 \
    | tee /tmp/cuda12-build.log

xmake install

-----------------------------------------------------

cd /data/llaisys-26s

export MACA_PATH=/opt/maca
export XMAKE_ROOT=y
export PYTHONPATH=/data/llaisys-26s/python:$PYTHONPATH

xmake f -c \
    --nv-gpu=n \
    --metax-gpu=y \
    -cv

xmake -r -vD 2>&1 | tee /data/metax-build.log

xmake install

pip install -e ./python

python test/test_runtime.py --device metax

python test/ops/add.py --device metax

-------------------------------------------

python test/test_runtime.py --device nvidia

python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia


python test/test_runtime.py --device cpu
python test/ops/add.py --device cpu
python test/ops/argmax.py --device cpu
python test/ops/embedding.py --device cpu
python test/ops/linear.py --device cpu
python test/ops/rms_norm.py --device cpu
python test/ops/rope.py --device cpu
python test/ops/self_attention.py --device cpu
python test/ops/swiglu.py --device cpu
python test/test_runtime.py --device cpu


MODEL_PATH="$(cat tmp/model_path.txt)"
python test/test_infer.py \
        --model "$MODEL_PATH" \
        --test \
        --device nvidia \
        --prompt "Who are you?" \
        --max_steps 70


xmake f -c --nv-gpu=n -cv
xmake -r
xmake install

python test/test_runtime.py --device cpu
python test/test_tensor.py --device cpu
python test/test_ops.py --device cpu





xmake f -c \
    --nv-gpu=y \
    -cv

xmake run test-context-device-switch
(llaisys) chufeng@Chufeng:~/Desktop/InfiniTensor/llaisys-26s$ xmake run test-context-device-switch
NVIDIA device headers: /usr/local/cuda/include
llaisys-core CUDA headers: /usr/local/cuda/include
cuDNN Backend header: /usr/include/x86_64-linux-gnu/cudnn.h
cuDNN Frontend header: /usr/local/include/cudnn_frontend.h
NVRTC header: /usr/local/cuda/include/nvrtc.h
cuDNN library: libcudnn.so
NVRTC library: libnvrtc.so
cuDNN Self-Attention enabled
[ 88%]: <test-context-device-switch> linking.release test-context-device-switch
===== Context Device Switch Test =====
[1] Activate CPU
CPU Runtime address: 0x5d5aba333440

[2] Activate NVIDIA
NVIDIA Runtime address: 0x5d5ab978e3e0

[3] Switch back to CPU
CPU Runtime address: 0x5d5aba333440
CPU Runtime reuse verified.

[4] Switch back to NVIDIA
NVIDIA Runtime address: 0x5d5ab978e3e0
NVIDIA Runtime reuse verified.

[5] Repeated switching
Iteration 1/10 passed.
Iteration 2/10 passed.
Iteration 3/10 passed.
Iteration 4/10 passed.
Iteration 5/10 passed.
Iteration 6/10 passed.
Iteration 7/10 passed.
Iteration 8/10 passed.
Iteration 9/10 passed.
Iteration 10/10 passed.

Context Runtime ownership test passed.
(llaisys) chufeng@Chufeng:~/Desktop/InfiniTensor/llaisys-26s$ 