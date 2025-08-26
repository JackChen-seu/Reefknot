export CUDA_VISIBLE_DEVICES=2
python LLaVA/infer_LLaVA_yesandno.py \
    --model-path /hpc2hdd/home/yuxuanzhao/lijungang/nk_code/LLM/liuhaotian/llava-v1.5-13b \
    --question-file Dataset/YESNO.jsonl \
    --answers-file Result/YesNo_results.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --apha 0.1 \
    --layer 38 \
    --threshold 0.9 
