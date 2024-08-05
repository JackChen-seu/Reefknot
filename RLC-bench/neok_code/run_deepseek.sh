#!/bin/bash

# 公共参数
export CUDA_VISIBLE_DEVICES="6"
MODEL_TYPE="deepseek-vl-7b-chat"
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


# 初始化 conda
__conda_setup="$('/home/ubuntu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/ubuntu/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/ubuntu/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# conda init
conda activate deberta
python /home/ubuntu/junkai/DeBERTa_for_VQA_judge.py \
    --yesno_file $ANSWERS_FILE1 \
    --multichoice_file $ANSWERS_FILE2 \
    --vqa_file $ANSWERS_FILE3