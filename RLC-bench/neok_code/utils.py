import json
import argparse
import torch
import os
import json
from tqdm import tqdm
# import shortuuid
import json


def get_path(image_id):
    Image_path1 = '/home/ubuntu/hallu_team/junkai/Dataset/VisualGenome/'
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
    
    
        category = 'Yes/No'

def read_jsonl(file):
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]
    
    
def save_jsonl(data, output_file_path):
    with open(output_file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Saved to {output_file_path}")
    
def calculate_multichoice_accuracy(file_path):
    total_count = 0
    correct_count = 0

    # 打开并逐行读取 JSONL 文件
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)  # 将每行的 JSON 字符串转换为字典
            total_count += 1
            # 获取 response 的第一个字母
            response_first_letter = data['response'][0].upper()
            # 比较 response 的第一个字母和 label
            if response_first_letter == data['label'].upper():
                correct_count += 1

    # 计算准确率
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f"Total entries: {total_count}")
        print(f"Correct responses: {correct_count}")
        print(f"Accuracy: {accuracy:.2f}")
    else:
        print("No entries to process.")
        
        

def calculate_yes_no_accuracy(file_path):
    total_count = 0
    correct_count = 0
    yes_rate=0
    # 打开并逐行读取 JSONL 文件
    print('file_path:',file_path)
    with open(file_path, 'r') as file:
        for line in file:
            data = json.loads(line)  # 将每行的 JSON 字符串转换为字典
            total_count += 1
            # 获取 response 和 label 并转为小写
            response = data['response'].strip().lower()
            
            label = data['label'].strip().lower()
            # 比较 response 和 label
            response = 'yes' if 'yes' in response else 'no'
            if response=='yes':
                yes_rate+=1
                
            if response==label:
                correct_count += 1

    
    accuracy = correct_count / total_count
    print(f"Total num is : {total_count}")
    print(f"Correct responses: {correct_count}")
    print(f"Accuracy: {accuracy:.4f}")
    hal_rate=1-accuracy
    print(f"Hallucination rate is :{hal_rate:.4f}")
    print(f"Yes rate:{yes_rate/total_count:.2f}")
   
def calculate_pairwise_accuracy(file_path):
    
    correct_count = 0
    yes_right=0
    yes_rate=0
    no_right=0
    pairwise_right=0
    file= read_jsonl(file_path)
    total_count = len(file)
    # 打开并逐行读取 JSONL 文件
    print('file_path:',file_path)

   
    for i in range(0,total_count,2):
        data1 = file[i]
        
        response1 = data1['response'].strip().lower()
        label1 = data1['label'].strip().lower()
        # 比较 response 和 label
        response1 = 'yes' if 'yes' in response1 else 'no'
        if response1==label1 :
            yes_right+=1         
        
        data2 = file[i+1]
        
        response2 = data2['response'].strip().lower()
        label2 = data2['label'].strip().lower()
        # 比较 response 和 label
        response2 = 'no' if 'no' in response2 else 'yes'
        if response2==label2:
            no_right+=1    
        if response1==label1 and response2==label2:
            pairwise_right+=1
        if response1=='yes':
            yes_rate+=1
        if response2=='yes':
            yes_rate+=1
    print(f"Yes right num:{yes_right}")
    print(f"No right num:{no_right}")
    each_count=total_count/2
    print(f"Accuracy for Yes :{yes_right/each_count:.2f}")
    print(f"Accuracy for No :{no_right/each_count:.2f}")
    print(f"Pairwise right num:{pairwise_right}")
    print(f"Pairwise right rate:{pairwise_right/each_count:.2f}")
    print(f"total num :{total_count}")
    
    print(f"saves to {file_path}")


if __name__=='__main__':
    # calculate_multichoice_accuracy('/data2/zkn/neok_code/MiniGPT-4/TestResult/minigpt_Cognition_multichoice_result.jsonl')
    # calculate_pairwise_accuracy('/data2/zkn/neok_code/MiniGPT-4/TestResult/Perception_yes+no_qwen.jsonl')
    
    calculate_yes_no_accuracy(file_path='/home/ubuntu/hallu_team/RLC-bench/Dataset/answer/qwen_Multichoice_POPE.jsonl')
    calculate_yes_no_accuracy(file_path='/home/ubuntu/hallu_team/kening/kening_results/qwen_Multichoice_RLC.jsonl')