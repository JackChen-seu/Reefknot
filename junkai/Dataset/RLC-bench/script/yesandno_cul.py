import pandas as pd
import json

# 读取jsonl文件
file_path = '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/answer/Relation/Minigpt4/cogintive/YesandNo.jsonl'
data = []

with open(file_path, 'r') as file:
    for line in file:
        data.append(json.loads(line.strip()))

# 将数据转换为DataFrame
df = pd.DataFrame(data)

# 检查response的第一个单词是否与label匹配
df['response_first_word'] = df['response'].apply(lambda x: x.split()[0].strip(',').lower())
df['label_matches_response'] = df['label'] == df['response_first_word']

# 计算准确率
accuracy = df['label_matches_response'].mean()


# 计算四种情况下的yes_prob和no_prob平均值
yes_label_yes_response = df[(df['label'] == 'yes') & (df['response_first_word'] == 'yes')]
yes_label_no_response = df[(df['label'] == 'yes') & (df['response_first_word'] == 'no')]
no_label_yes_response = df[(df['label'] == 'no') & (df['response_first_word'] == 'yes')]
no_label_no_response = df[(df['label'] == 'no') & (df['response_first_word'] == 'no')]

# 计算均值
yes_label_yes_response_yes_prob_mean = yes_label_yes_response['yes_prob'].mean()
yes_label_yes_response_no_prob_mean = yes_label_yes_response['no_prob'].mean()

yes_label_no_response_yes_prob_mean = yes_label_no_response['yes_prob'].mean()
yes_label_no_response_no_prob_mean = yes_label_no_response['no_prob'].mean()

no_label_yes_response_yes_prob_mean = no_label_yes_response['yes_prob'].mean()
no_label_yes_response_no_prob_mean = no_label_yes_response['no_prob'].mean()

no_label_no_response_yes_prob_mean = no_label_no_response['yes_prob'].mean()
no_label_no_response_no_prob_mean = no_label_no_response['no_prob'].mean()

# 输出结果
print(f'准确率: {accuracy * 100:.2f}%')
print(f'当label和response都为yes时:')
print(f'yes_prob 平均值: {yes_label_yes_response_yes_prob_mean}')
print(f'no_prob 平均值: {yes_label_yes_response_no_prob_mean}')

print(f'当label为yes而response为no时:')
print(f'yes_prob 平均值: {yes_label_no_response_yes_prob_mean}')
print(f'no_prob 平均值: {yes_label_no_response_no_prob_mean}')

print(f'当label为no而response为yes时:')
print(f'yes_prob 平均值: {no_label_yes_response_yes_prob_mean}')
print(f'no_prob 平均值: {no_label_yes_response_no_prob_mean}')

print(f'当label和response都为no时:')
print(f'yes_prob 平均值: {no_label_no_response_yes_prob_mean}')
print(f'no_prob 平均值: {no_label_no_response_no_prob_mean}')
