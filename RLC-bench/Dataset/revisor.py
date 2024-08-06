import openai
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

def get_completion_proxy(prompt, temperature=0): 
    API_SECRET_KEY = "sk-4XNIBlZed4nyH4WYlvuykvtIuDi7qqqdXc32RWgvRsx5ligB";
    BASE_URL = "https://api.chatanywhere.com.cn/v1"
    client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(
                    model="gpt-3.5-turbo-1106",
                    messages=messages,
                    temperature=temperature,
                )
        cost_token=resp.usage.total_tokens
        answer=resp.choices[0].message.content
    except:
        resp = client.chat.completions.create(
                    model="gpt-3.5-turbo-0125",
                    messages=messages,
                    temperature=temperature,
                )
        cost_token=resp.usage.total_tokens
        answer=resp.choices[0].message.content
    return  answer,cost_token

def prompt_wrapper(positive_sentence):
    prompt=f"""You are an assistant who helps humans create questions, please follow my instructions to rephrase the following sentences:
    I will provide some judgment sentences with answers as "yes," and I need you to replace the verbs in these sentences while keeping everything else unchanged, setting a sentence with an answer of "no." I will give you a few examples for reference.
    After replacing the relation word, the sentences should still be reasonable, and it is not allowed to express the same meaning as before change one.
    You can make some minor modifications to ensure the grammatical correctness of the sentence, you can refer to the third example.
    I will provide you with some examples for reference:
    Cases:
    Positive: Is the cat sitting on the table in this photo?
    Negative: Is the cat under the table in this photo? 
    Positive: Are the glasses over pilots in this photo?
    Negative: Are the glasses under pilots in this photo?
    Positive: Is the glass over pilots in this photo?
    Negative: Are the glasses under pilots in this photo?

    Input:
    Positive: {positive_sentence}
    Negative:"""
    return prompt

if __name__ == '__main__':
    data=read_jsonl('/home/ubuntu/kening/kening/RLC-bench/RLC-bench/Dataset/release_v0/YESNO.jsonl')
    cost_token=0
    for idx, item in enumerate(tqdm(data, desc="Processing data")):
        if idx%2==0 and item['label']=='yes':
            while True:
                try:
                    positive_sentence = item["query_prompt"]
                    prompt=prompt_wrapper(positive_sentence)
                    answer,cost_token=get_completion_proxy(prompt)
                    cost_token+=cost_token
                    # print(answer)
                    # print(cost_token)
                    # print(f'idx is {idx}------------------')
                    # print(f'Yes setxxx {positive_sentence}')
                    break
                except Exception as e:
                    print(f"遇到错误: {e}，正在暂停...")
                    time.sleep(30)
                    continue  # 继续循环
        else:
            # answer=answer.split(':')[1]
            # answer=answer.split('.')[0]
            data[idx]['query_prompt']=answer
    print(f'{cost_token} tokens have been used in total')
    save_jsonl(data, '/home/ubuntu/kening/kening/RLC-bench/RLC-bench/Dataset/release_v0/YESNO2.jsonl')
    
 
                
