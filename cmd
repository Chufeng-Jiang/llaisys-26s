(llaisys) chufeng@Chufeng:~/Desktop/InfiniTensor/llaisys-26s$ MODEL_PATH="$(cat tmp/model_path.txt)"

python test/test_infer.py \
        --device cpu \
        --model "$MODEL_PATH" \
        --prompt "Who are you?" \
        --max_steps 1 \
        --test