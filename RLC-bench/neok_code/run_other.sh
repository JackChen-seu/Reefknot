export CUDA_VISIBLE_DEVICES=4
python /home/ubuntu/hallu_team/RLC-bench/neok_code/run_other.py \
    --model_type qwen-vl-chat \
    --question-file /home/ubuntu/hallu_team/RLC-bench/Dataset/bench/Relation/cogintive/benchmark-3-multichoice.jsonl \
    --answers-file /home/ubuntu/hallu_team/kening/kening_results/test0.jsonl \
    --category  yesno+\
    --device cuda:0 