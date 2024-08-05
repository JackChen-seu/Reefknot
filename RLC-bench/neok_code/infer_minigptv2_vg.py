# CUDA_VISIBLE_DEVICES=3 python /data2/zkn/neok_code/MiniGPT-4/infer_minigptv2_vg.py --cfg-path /data2/zkn/neok_code/MiniGPT-4/eval_configs/minigptv2_eval.yaml
# CUDA_VISIBLE_DEVICES=3 python /home/ubuntu/kening/kening/MiniGPT-4/infer_minigptv2_vg.py --cfg-path /home/ubuntu/kening/kening/MiniGPT-4/eval_configs/minigptv2_eval.yaml
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset


# from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config
import os


def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def get_path(image_id):
    Image_path1 = '/home/ubuntu/junkai/VisualGenome/'
    Image_path2 = '/home/ubuntu/hallu_team/junkai/Dataset/COCO/val2014/'
    # if image is not None:
    image_id = str(image_id)
    if image_id.endswith('.jpg'):
        image_id = image_id.split('.')[0]
    if os.path.exists(os.path.join(Image_path1, image_id+'.jpg')):
        # print('Find image in VG100K(small one!) image path is:',os.path.join(Image_path1, image_id+'.jpg'))
        return os.path.join(Image_path1, image_id+'.jpg')
    elif os.path.exists(os.path.join(Image_path2, image_id+'.jpg')):
        return os.path.join(Image_path2, image_id+'.jpg')
    else:
        print('Cannot find image {}.jpg'.format(image_id))
        return None


def list_of_str(arg):
    return list(map(str, arg.split(',')))


def MiniGPT4_infer(model, vis_processor, prompt,image_path):
    conv_temp = CONV_VISION_minigptv2.copy() #聊天类的实例化
    conv_temp.system = ""
    model.eval()
    prompt= f"[vqa] Based on the image, respond to this question with a short answer: {prompt}"
    prompt=[prompt,'']
    prompt = prepare_texts(prompt, conv_temp)  # 必须要输入一个list 应该是之前用于处理batch的 相当于在这里加上了prompt模版里的特殊token
    # prompt=[prompt[0]]
    image = Image.open(image_path).convert('RGB')
    # print(image)
    image = vis_processor(image)
    image=image.unsqueeze(0)
    # print('question :',question)
    response, prob_yes, prob_no  = model.generate(image, prompt, max_new_tokens=30, do_sample=False)
    # print('Response:',response[0])
    return response,  prob_yes, prob_no


def save_jsonl(data, output_file_path):
    with open(output_file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Saved to {output_file_path}")


if __name__=="__main__":
    parser = eval_parser()
    # parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")

    parser.add_argument("--model_version", type=str, default='minigptv2', help="model version")
    parser.add_argument("--question_dir", type=str, default="/home/ubuntu/kening/kening/RLC-bench/RLC-bench/Dataset/release_v0/YESNO.jsonl")
    parser.add_argument("--output_file", type=str, default='', help="output file path")
    parser.add_argument("--category", type=str, default='yesno+', help="category")
    args = parser.parse_args()
    cfg = Config(args)

    model, vis_processor = init_model(args)
    conv_temp = CONV_VISION_minigptv2.copy() #聊天类的实例化
    conv_temp.system = ""
    model.eval()
    vg_json=load_jsonl(args.question_dir)
    result = []
    for i in tqdm(range(len(vg_json)),desc=f"Processing {args.category} items"):
        item = vg_json[i]
        image_path = get_path(vg_json[i]['image_id'])
        question_prompt = vg_json[i]['query_prompt']
        # print(f'prompt: {question_prompt }')
        response,  prob_yes, prob_no = MiniGPT4_infer(model, vis_processor, prompt=question_prompt,image_path=image_path)

        item['response']= response[0]
        item['mllm_name']= args.model_version
        # print(f'response: {response[0]}')
        result.append(item)
    save_jsonl(result, args.output_file)
    if args.category == 'multichoice':
        from utils import calculate_multichoice_accuracy
        calculate_multichoice_accuracy(args.output_file)
    elif args.category == 'yesno':
        from utils import calculate_yes_no_accuracy
        calculate_yes_no_accuracy(args.output_file)
    if args.category == 'yesno+':
        from utils import calculate_pairwise_accuracy
        calculate_pairwise_accuracy(args.output_file)
        
        
