import json
import torch
import numpy as np
import matplotlib.pyplot as plt

def entropy(p_yes, p_no):
    """Calculate the entropy for a given probability."""
    if p_yes <= 0 or p_no <= 0:
        return 0
    p_yes = torch.tensor([p_yes], dtype=torch.float32)
    p_no = torch.tensor([p_no], dtype=torch.float32)
    entropy_value = -p_yes * torch.log2(p_yes) - p_no * torch.log2(p_no)
    return entropy_value.item()

def calculate_average_probabilities(entries):
    layer_counts = len(entries[0]['yes_layer_prob'])
    yes_prob_sums = {str(i): 0 for i in range(layer_counts)}
    no_prob_sums = {str(i): 0 for i in range(layer_counts)}
    
    for entry in entries:
        for i in range(layer_counts):
            yes_prob_sums[str(i)] += entry['yes_layer_prob'][str(i)]
            no_prob_sums[str(i)] += entry['no_layer_prob'][str(i)]
    
    yes_prob_avgs = {str(i): yes_prob_sums[str(i)] / len(entries) for i in range(layer_counts)}
    no_prob_avgs = {str(i): no_prob_sums[str(i)] / len(entries) for i in range(layer_counts)}
    
    return yes_prob_avgs, no_prob_avgs

# 读取JSONL文件并解析
input_file = '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/test/answers_YesandNo_cognitive.jsonl'
selected_entries = []

labels_responses = {
    'yes_yes': [],
    'yes_no': [],
    'no_yes': [],
    'no_no': []
}

with open(input_file, 'r') as file:
    for line in file:
        data = json.loads(line)
        yes_prob = data['yes_prob']
        no_prob = data['no_prob']
        
        # 计算熵
        ent = entropy(yes_prob, no_prob)
        
        # 选出熵大于0.9的示例
        if ent > 0:
            selected_entries.append(data)
            
            label_first_word = data['label'].split()[0].lower()
            response_first_word = data['response'].split()[0].lower()
            key = f"{label_first_word}_{response_first_word}"
            if key in labels_responses:
                labels_responses[key].append(data)

# 计算每种情况的平均值
averages = {}
for key, entries in labels_responses.items():
    if entries:
        yes_avg, no_avg = calculate_average_probabilities(entries)
        averages[key] = (yes_avg, no_avg)

# 绘制图表
for key, (yes_avg, no_avg) in averages.items():
    layers = list(map(int, yes_avg.keys()))
    yes_probs = list(yes_avg.values())
    no_probs = list(no_avg.values())
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, yes_probs, label='yes_prob', marker='o')
    plt.plot(layers, no_probs, label='no_prob', marker='x')
    plt.xlabel('Layer')
    plt.ylabel('Probability')
    plt.title(f'Average Probabilities for {key}')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/test/average_probabilities_{key}.png')
    plt.close()
