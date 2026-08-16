git push origin metax-port


find . \
    \( -name "*.c" -o -name "*.cc" -o -name "*.cpp" \
       -o -name "*.h" -o -name "*.hpp" \
       -o -name "*.cu" -o -name "*.cuh" \
        -o -name "*.maca" \) \
    -exec clang-format -i {} +

ruff format .

Add             ✓ template finished
SwiGLU          ← next
Embedding
RMSNorm
RoPE
Argmax
Linear
Self-Attention
--------------------------------------------------
cd ~/Desktop/InfiniTensor/llaisys-26s
xmake clean -a
rm -rf .xmake
rm -rf build
xmake f -c \
    --nv-gpu=y \
    -cv

xmake -r -vD 2>&1 \
    | tee /tmp/cuda12-build.log

xmake install

pip install ./python/

-----------------------------------------------------

cd /data/llaisys-26s

export MACA_PATH=/opt/maca
export XMAKE_ROOT=y
export PYTHONPATH=/data/llaisys-26s/python:$PYTHONPATH

xmake f -c \
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



python test/test_runtime.py --device metax
python test/ops/add.py --device metax
python test/ops/argmax.py --device metax
python test/ops/embedding.py --device metax
python test/ops/linear.py --device metax
python test/ops/rms_norm.py --device metax
python test/ops/rope.py --device metax
python test/ops/self_attention.py --device metax
python test/ops/swiglu.py --device metax
python test/test_runtime.py --device metax


python test/test_runtime.py --device nvidia     --backend triton
python test/ops/add.py --device nvidia     --backend triton
python test/ops/argmax.py --device nvidia     --backend triton
python test/ops/embedding.py --device nvidia     --backend triton
python test/ops/linear.py --device nvidia     --backend triton
python test/ops/rms_norm.py --device nvidia     --backend triton
python test/ops/rope.py --device nvidia     --backend triton
python test/ops/self_attention.py --device nvidia     --backend triton
python test/ops/swiglu.py --device nvidia     --backend triton




MODEL_PATH="$(cat tmp/model_path.txt)"
python test/test_infer.py \
        --model "$MODEL_PATH" \
        --test \
        --device nvidia \
        --prompt "Who are you?" \
        --max_steps 70

MODEL_PATH="$(cat tmp/model_path.txt)"
python test/test_infer.py \
        --model "$MODEL_PATH" \
        --test \
        --device cpu \
        --prompt "Who are you?" \
        --max_steps 70

cd /data/llaisys-26s

python test/test_infer.py \
	--model /data/huggingface_home/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-1.5B/snapshots/ad9f0ae0864d7fbcd1cd905e3c6c5b069cc8b562 \
	--test \
	--device metax \
	--prompt "Who are you?" \
	--max_steps 10

=====================NV and Triton=====================
export LLAISYS_DEBUG=1 \
export LLAISYS_ADD_ENABLE_VECTORIZED=1  \
LLAISYS_BLOCK_SIZE=256 \
python test/ops/add.py \
    --device nvidia \
    --backend native \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/add

LLAISYS_TRITON_BLOCK_SIZE=256 \
LLAISYS_TRITON_NUM_WARPS=4 \
python test/ops/add.py \
    --device nvidia \
    --backend triton \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/add

----------------------------------------


LLAISYS_TRITON_BLOCK_SIZE=256 \
LLAISYS_TRITON_NUM_WARPS=4 \
python test/ops/swiglu.py \
    --device nvidia \
    --backend triton \
    --backend-variant baseline \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/swiglu

LLAISYS_BLOCK_SIZE=256 \
python test/ops/swiglu.py \
    --device nvidia \
    --backend native \
    --backend-variant baseline \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/swiglu

    --------------------------------------

LLAISYS_TRITON_BLOCK_SIZE=128 \
LLAISYS_TRITON_NUM_WARPS=4 \
python test/ops/embedding.py \
    --device nvidia \
    --backend triton \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/embedding

LLAISYS_TRITON_BLOCK_SIZE=128 \
LLAISYS_TRITON_NUM_WARPS=4 \
python test/ops/embedding.py \
    --device nvidia \
    --backend native \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/embedding

    -------------------------------------------

    LLAISYS_BLOCK_SIZE=256 \
python test/ops/rms_norm.py \
    --device nvidia \
    --backend native \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/rms_norm

RMSNorm 这里我建议baseline 先不要手动固定 BLOCK_SIZE=256。
因为当前 RMSNorm baseline 本身就是根据 ncol 动态解析

    python test/ops/rms_norm.py \
    --device nvidia \
    --backend triton \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/rms_norm

    -------------------------------

    python test/ops/argmax.py \
    --device nvidia \
    --backend triton \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/argmax

    python test/ops/argmax.py \
    --device nvidia \
    --backend native \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/argmax

    -----------------------------

    python test/ops/linear.py \
    --device nvidia \
    --backend triton \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/linear



    python test/ops/linear.py \
    --device nvidia \
    --backend native \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/linear

----------------

python test/ops/self_attention.py \
    --device nvidia \
    --backend triton \
    --backend-variant baseline \
    --benchmark-order alternating \
    --profile \
    --profile-suite all \
    --show-config \
    --show-bandwidth \
    --show-throughput \
    --output-dir result/self_attention