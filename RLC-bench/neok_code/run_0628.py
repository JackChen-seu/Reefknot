import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from http import HTTPStatus
import json

from tqdm import tqdm

import datetime
import random
import re
import torch

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


def save_filtered_json(data, output_file_path):
    with open(output_file_path, 'w') as f:
        json.dump(data, f, indent=4)


def calculate_consistency(data, filename):
    
    consistent_count = 0
    inconsistent_count = 0
    for item in data:
        response = item['response'].strip().lower()
        true_answer = item['answer'].strip().lower()
    
        if 'yes' in response:
            response = 'yes'
        elif 'no' in response:
            response = 'no'
        
        if response == true_answer:
            consistent_count += 1
        else:
            inconsistent_count += 1
    
    total = consistent_count + inconsistent_count
    consistent_ratio = consistent_count / total
    inconsistent_ratio = inconsistent_count / total
    logging.info(f"Consistent Count: {consistent_count}")
    logging.info(f"Inconsistent Count: {inconsistent_count}")
    logging.info(f"{filename} Consistent Ratio: {consistent_ratio:.2%}")
    logging.info(f"{filename} Inconsistent Ratio: {inconsistent_ratio:.2%}")
    return consistent_ratio, inconsistent_ratio

def cacluate_current_time_save_path(temp,args):
    current_time = datetime.datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{formatted_time}.jsonl"
    save_directory = f"/data2/zkn/neok_code/MiniGPT-4"
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    save_path = f"{save_directory}/{temp}_{filename}"
    return save_path

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_image_path', type=str, default="/data2/zkn/neok_code/MiniGPT-4/MME_test/test")
    parser.add_argument('--model_type', type=str, default='qwen-vl-chat')#qwen-vl-chat 
    parser.add_argument('--tempeature', type=float, default=0.1 )
    args = parser.parse_args()
    
    log_filename = f'{args.model_type}.txt'
    logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if args.model_type == 'llava1_6-yi-34b-instruct' or 'llava1_6-mistral-7b-instruct' or 'cogvlm2-19b-chat' or 'minicpm-v-v2_5-chat' or 'minicpm-v-v2-chat' or 'qwen-vl-chat' or 'glm4v-9b-chat' or 'deepseek-vl-7b-chat':
        model_type = args.model_type
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(42)
    else:
        raise NotImplementedError
    
    try:
        data = load_dataset("lmms-lab/MME")
        val_data = data["test"]
    except Exception as e:
        print(f"Error occurred: {e}. Loading 3000 samples dataset.")

    answer_all=[]
    for idx, item in enumerate(tqdm(val_data, desc="Processing items")):
        index = item["question_id"]
        question = item["question"]
        image = item["image"]
        answer = item["answer"]
        category = item["category"]

        prompt= question
        
        image = os.path.join(args.local_image_path, index)

        response, _ = inference(model, template, prompt, images=image,temperature=args.tempeature)

        answer_data_json = {
            'index': index,
            'question': question,
            'answer' : answer,
            'response': response,
            'category' : category ,
            'prompt': prompt
        }
        print(f'##Prompt : {prompt}')
        print(f'##Response : {response}')
        answer_all.append(answer_data_json)

    save_path = cacluate_current_time_save_path('all', args)
    # save_path = '/data2/zkn/neok_code/MiniGPT-4/test1111.jsonl'
    json.dump(answer_all, open(save_path, 'w', encoding='utf-8'),
                      indent=2, ensure_ascii=False)
    consistent_ratio_all, inconsistent_ratio_all = calculate_consistency(answer_all,'All data-')


    # 第二阶段, 过滤出来true和false的数据
    filtered_data_false_true = []
    filtered_data_true_false = []

    for item in answer_all:
        response = item['response'].strip().lower()
        true_answer = item['answer'].strip().lower()
    
        if 'yes' in response:
            response = 'yes'
        elif 'no' in response:
            response = 'no'
        
        if response == true_answer:
            filtered_data_true_false.append(item)
        elif response != true_answer:
            filtered_data_false_true.append(item)
        else:
            print('ERROR! ERROR!')

    save_path= cacluate_current_time_save_path('all_false',args)
    json.dump(filtered_data_false_true, open(save_path, 'w', encoding='utf-8'),
                        indent=2, ensure_ascii=False)

    save_path= cacluate_current_time_save_path('all_true',args)
    json.dump(filtered_data_true_false, open(save_path, 'w', encoding='utf-8'),
                        indent=2, ensure_ascii=False)
    