#!/bin/bash

# 公共参数
export CUDA_VISIBLE_DEVICES="7"
MODEL_TYPE="qwen-vl-chat"
DEVICE="cuda:0"
BASE_PATH="/home/ubuntu/kening/kening/RLC-bench/RLC-bench/Dataset/release_v0"
PYTHON_PATH='/home/ubuntu/kening/kening/RLC-bench/RLC-bench/neok_code/run_other.py'
RESULTS_DIR="/home/ubuntu/kening/kening/kening_results/$MODEL_TYPE"

# 检查并创建结果路径
if [ ! -d "$RESULTS_DIR" ]; then
  mkdir -p "$RESULTS_DIR"
fi

QUESTION_FILE1="$BASE_PATH/YESNO.jsonl"
QUESTION_FILE2="$BASE_PATH/Multichoice.jsonl"
QUESTION_FILE3="$BASE_PATH/VQA.jsonl"
ANSWERS_FILE1="$RESULTS_DIR/yesno_result.jsonl"
ANSWERS_FILE2="$RESULTS_DIR/Multichoice_result.jsonl"
ANSWERS_FILE3="$RESULTS_DIR/VQA_result.jsonl"

# python $PYTHON_PATH \
#     --model_type $MODEL_TYPE \
#     --question-file $QUESTION_FILE1 \
#     --answers-file $ANSWERS_FILE1 \
#     --category 'yesno' \
#     --device $DEVICE 

# python $PYTHON_PATH \
#     --model_type $MODEL_TYPE \
#     --question-file $QUESTION_FILE2 \
#     --answers-file $ANSWERS_FILE2 \
#     --category 'multichoice' \
#     --device $DEVICE

python $PYTHON_PATH \
    --model_type $MODEL_TYPE \
    --question-file $QUESTION_FILE3 \
    --answers-file $ANSWERS_FILE3 \
    --category 'vqa' \
    --device $DEVICE



python /home/ubuntu/junkai/DeBERTa_for_VQA_judge.py \
    --yesno_file $ANSWERS_FILE1 \
    --multichoice_file $ANSWERS_FILE2 \
    --vqa_file $ANSWERS_FILE3