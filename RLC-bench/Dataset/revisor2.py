# answer='''Positive: Is the dog beside the tree in this photo? Please answer yes or no.
# Negative: Is the dog behind the tree in this photo? Please answer yes or no.
# '''

# answer='''Is the dog beside the tree in this photo? Please answer yes or no.
# '''
# answer=answer.split(':')[1]
# answer=answer.split('.')[0]
# answer=answer+'.'
# print(answer)


import openai
from tqdm import tqdm
import json
from openai import OpenAI
from tqdm import tqdm
def save_jsonl(data, output_file_path):
    with open(output_file_path, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
    print(f"Saved to {output_file_path}")
def read_jsonl(file):
    with open(file, 'r') as f:
        return [json.loads(line) for line in f]
    
data= read_jsonl('/home/ubuntu/kening/kening/RLC-bench/RLC-bench/Dataset/release_v0/YESNO2.jsonl')
for i in tqdm(range(0,len(data)), desc="Processing data"):
    if '\n' in data[i]['query_prompt']:
        data[i]['query_prompt']=data[i]['query_prompt'].split('\n')[0]
save_jsonl(data,'/home/ubuntu/kening/kening/RLC-bench/RLC-bench/Dataset/release_v0/YESNO3.jsonl')