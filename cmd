git push origin metax-port


find src test \
    \( -name "*.c" -o -name "*.cc" -o -name "*.cpp" \
       -o -name "*.h" -o -name "*.hpp" \
       -o -name "*.cu" -o -name "*.cuh" \
        -o -name "*.maca" \) \
    -exec clang-format -i {} +

find src test -name "*.py" -exec black {} +

Tensor → Add → SwiGLU → Rearrange → Embedding → Argmax → RMSNorm → RoPE → Linear → Self-Attention
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

