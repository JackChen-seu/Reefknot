# 设置第一个命令所用的 GPU 设备
CUDA_VISIBLE_DEVICES=6 python /home/ubuntu/hallu_team/RLC-bench/neok_code/run_other.py \
    --model_type minicpm-v-v2_5-chat \
    --question-file /home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/bench/Object/object.jsonl \
    --answers-file /home/ubuntu/hallu_team/RLC-bench/kening/kening_results/cpm25_Multichoice_POPE.jsonl \
    --category yesno \
    --device cuda:0 &

# 设置第二个命令所用的 GPU 设备
CUDA_VISIBLE_DEVICES=7 python /home/ubuntu/hallu_team/RLC-bench/neok_code/run_other.py \
    --model_type minicpm-v-v2_5-chat \
    --question-file /home/ubuntu/hallu_team/RLC-bench/Dataset/release_v0/YESNO.jsonl \
    --answers-file /home/ubuntu/hallu_team/kening/kening_results/cpm25_Multichoice_RLC.jsonl \
    --category yesno \
    --device cuda:0 &

# 等待所有后台进程完成
wait