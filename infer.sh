export CUDA_VISIBLE_DEVICES=0
python /home/ubuntu/junkai/LLaVA/llava/eval/infer_LLaVA_yesandno.py \
    --model-path /home/ubuntu/junkai/LLaVA/llava-v1.5-7b \
    --question-file /home/ubuntu/junkai/object_hallu/POPE_processed/gqa.jsonl \
    --image-folder /home/ubuntu/junkai/VisualGenome \
    --answers-file /home/ubuntu/junkai/object_hallu/eval_result/gqa/LLaVA-7b.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --apha 0.1 \
    --layer 38\
    --threshold 0.9\
    --model_type llava-v1.5-7b
