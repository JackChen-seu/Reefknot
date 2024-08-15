
#!/bin/bash
declare -A env_map
env_map=(
    ["phi3-vision-128k-instruct"]="MiniCPM-V"
    ["qwen-vl-chat"]="mmstar"
    ["glm4v-9b-chat"]="dyk_glm"
    ["cogvlm2-19b-chat"]="mmstar"
    ["minicpm-v-v2-chat"]="mmstar"
    ["deepseek-vl-7b-chat"]="MiniCPM-V"
    ["minicpm-v-v2_5-chat"]="MiniCPM-V"
    ["llava-llama-3-8b-v1_1"]="dyk_llava"
    ["llava1_6-mistral-7b-instruct"]="dyk_llava"
    ["llava1_6-yi-34b-instruct"]="MiniCPM-V"
    ["yi-vl-6b-chat"]="dyk_llava"
    ["yi-vl-34b-chat"]="dyk_llava"
    ["internvl-chat-v1_5"]="dyk_llava"
)

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


activate_conda_env() {
    local model_type=$1
    local env_name=${env_map[$model_type]}
    if [ -n "$env_name" ]; then
        echo "Activating conda environment: $env_name"
        conda activate "$env_name"
    else
        echo "No environment found for MODEL_TYPE: $model_type"
    fi
}

export CUDA_VISIBLE_DEVICES="7"
MODEL_TYPE="phi3-vision-128k-instruct"
activate_conda_env "$MODEL_TYPE"
BASE_PATH="/home/ubuntu/kening/kening/RLC-bench/RLC-bench/Dataset/release_v1"
PYTHON_PATH='/home/ubuntu/kening/kening/RLC-bench/RLC-bench/neok_code/yunkai_run.py'
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

python $PYTHON_PATH \
    --model_type $MODEL_TYPE \
    --question-file $QUESTION_FILE1 \
    --answers-file $ANSWERS_FILE1 \
    --category 'yesno' 

python $PYTHON_PATH \
    --model_type $MODEL_TYPE \
    --question-file $QUESTION_FILE2 \
    --answers-file $ANSWERS_FILE2 \
    --category 'multichoice' 

python $PYTHON_PATH \
    --model_type $MODEL_TYPE \
    --question-file $QUESTION_FILE3 \
    --answers-file $ANSWERS_FILE3 \
    --category 'vqa' 


conda activate deberta
python /home/ubuntu/junkai/DeBERTa_for_VQA_judge.py \
  --yesno_file $ANSWERS_FILE1 \
  --multichoice_file $ANSWERS_FILE2 \
  --vqa_file $ANSWERS_FILE3 