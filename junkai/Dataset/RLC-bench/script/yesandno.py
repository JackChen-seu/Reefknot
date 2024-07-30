import pandas as pd
import json

# 读取jsonl文件
file_path = '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/answer/Object/InstructBLIP/answers_object_random..jsonl'
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

# # 根据response_first_word选择prob列
# df['response_prob'] = df.apply(lambda x: x['yes_prob'] if x['response_first_word'] == 'yes' else x['no_prob'], axis=1)

# # 计算response_prob的平均值
# response_prob_mean = df['response_prob'].mean()

# 输出结果
print(f'准确率: {accuracy * 100:.2f}%')
# print(f'response_prob 平均值: {response_prob_mean:.4f}')
