import json
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'   
from tqdm import tqdm
import os
import datetime
import random
import re
import torch
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


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, default='minicpm-v-v2-chat')
    parser.add_argument('--api_model', type=str, default='')
    parser.add_argument('--tempeature', type=float, default=0.1 )
    parser.add_argument("--question-file", type=str, default="/data2/zkn/neok_code/Datasets/benchmark-3-percetion/benchmark-3-YesandNo.jsonl") 
    parser.add_argument("--answers-file", type=str, default="TestResult/Perception_yes+no_qwen.jsonl")
    parser.add_argument("--category", type=str, default='yesno')

    args = parser.parse_args()
    
    if args.model_type == 'Qwen-VL-Chat':
        model_id = 'qwen/Qwen-VL-Chat'
        revision = 'v1.0.0'
        model_dir = snapshot_download(model_id, revision=revision)
        torch.manual_seed(1)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        if not hasattr(tokenizer, 'model_dir'):
            tokenizer.model_dir = model_dir
        model = AutoModelForCausalLM.from_pretrained(model_dir, device_map="cuda", trust_remote_code=True).eval()
    if args.model_type == 'closed_model':
        model = None
        template = None
        print(args.api_model)
    elif args.model_type :
        model_type = args.model_type
        template_type = get_default_template_type(model_type)
        model, tokenizer = get_model_tokenizer(model_type, torch.float16, model_kwargs={'device_map': 'auto'})
        model.generation_config.max_new_tokens = 256
        template = get_template(template_type, tokenizer)
        seed_everything(1)
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
        if args.api_model:
            print(args.api_model)
            #from IPython import embed; embed()
            response_temp = get_all_model_api_result(args, prompt, image)
        elif args.model_type == 'phi3-vision-128k-instruct':
            prompt = f'<img>{image}</img>{prompt}'
            response, _ = inference(model, template, prompt, temperature=args.tempeature)
        else :
            response, _ = inference(model, template, prompt, images=image,temperature=args.tempeature)
        print('response:',response)
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