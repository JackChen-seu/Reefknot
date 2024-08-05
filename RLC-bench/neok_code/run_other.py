import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from http import HTTPStatus
import json

from tqdm import tqdm
from utils import read_jsonl
import datetime
import random
import re
import torch
import sys
sys.path.append('/home/ubuntu/kening/kening/RLC-bench/RLC-bench/neok_code')
from utils import get_path,save_jsonl,read_jsonl,calculate_multichoice_accuracy
from datasets import load_dataset
import argparse
from swift.llm import (
        get_model_tokenizer, get_template, inference, ModelType,
        get_default_template_type, inference_stream
    )
from swift.utils import seed_everything
import logging
from modelscope import (
    snapshot_download, AutoModelForCausalLM, AutoTokenizer, GenerationConfig
)
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    # parser.add_argument('--local_image_path', type=str, default="/hpc2hdd/home/yxu409/dyk/Uncertainty_MLLMs/MME/test")
    parser.add_argument('--model_type', type=str, default='qwen-vl-chat')
    parser.add_argument('--tempeature', type=float, default=0 )
    parser.add_argument("--question-file", type=str, default="/data2/zkn/neok_code/Datasets/benchmark-3-percetion/benchmark-3-YesandNo.jsonl") 
    parser.add_argument("--answers-file", type=str, default="TestResult/Perception_yes+no_qwen.jsonl")
    parser.add_argument("--category", type=str, default='yesno')
    parser.add_argument("--device", type=str, default="cuda:2")
    args = parser.parse_args()
    
    # log_filename = f'{args.model_type}.txt'
    # logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.model_type == 'llava1_6-yi-34b-instruct' or 'llava1_6-mistral-7b-instruct' or 'cogvlm2-19b-chat' or 'minicpm-v-v2_5-chat' or 'minicpm-v-v2-chat' or 'qwen-vl-chat' or 'glm4v-9b-chat' or 'deepseek-vl-7b-chat':
        model_type = args.model_type
        template_type = get_default_template_type(model_type)
        device=args.device
        model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_kwargs={'device_map': device})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(42)
    else:
        raise NotImplementedError

    data=read_jsonl(args.question_file)

    answer_all=[]
    for idx, item in enumerate(tqdm(data, desc=f"Processing {args.category} items")):
        image_id = item["image_id"]
        image = get_path(image_id) 
        question = item["query_prompt"]
        answer = item["label"]
        category = args.category
        prompt= question
        # print('prompt:',prompt)
        response, _ = inference(model, template, prompt, images=image,temperature=args.tempeature)
        # print('response:',response)
        item['response']=response
        item['mllm_name']= args.model_type


        answer_all.append(item)
    save_jsonl(answer_all, args.answers_file)
    answers_file = args.answers_file    
    if args.category == 'multichoice':
        from utils import calculate_multichoice_accuracy
        calculate_multichoice_accuracy(answers_file)
    elif args.category == 'yesno':
        from utils import calculate_yes_no_accuracy
        calculate_yes_no_accuracy(answers_file)
    if args.category == 'yesno+':
        from utils import calculate_pairwise_accuracy
        calculate_pairwise_accuracy(args.answers_file)