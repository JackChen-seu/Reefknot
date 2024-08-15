import matplotlib.pyplot as plt
import numpy as np
import json
import torch
import pandas as pd
import debugpy


def read_jsonl(file):
    with open(file, 'r') as f:
        data=[json.loads(line) for line in f]
        print(f'Length of data is {len(data)}')
        return data
    
def filter_and_replace(value):
    value=value.lower()
    if 'yes' in value:
        return 'yes'
    elif 'no' in value:
        return 'no'
    else:
        return value

def entropy(p_yes, p_no):
    """Calculate the entropy for a given probability."""
    if p_yes <= 0 or p_no <= 0:
        return 0
    p_yes = torch.tensor([p_yes], dtype=torch.float32)
    p_no = torch.tensor([p_no], dtype=torch.float32)
    entropy_value = -p_yes * torch.log2(p_yes) - p_no * torch.log2(p_no)
    return entropy_value.item()

rel1_data=read_jsonl('/home/ubuntu/junkai/RLC-bench/test/confidence/MMRel_YesandNo_7b.jsonl')

rel1_pd=pd.DataFrame(rel1_data)

rel1_pd['entropy'] = rel1_pd.apply(lambda row: entropy(row['yes_prob'], row['no_prob']), axis=1)
rel1_pd['decision_prob']= rel1_pd[['yes_prob', 'no_prob']].max(axis=1)
rel1_pd['response']=rel1_pd['response'].apply(filter_and_replace)

rel1_wrong=rel1_pd[rel1_pd['response']!=rel1_pd['label']]
rel1_right=rel1_pd[rel1_pd['response']==rel1_pd['label']]
equal_rel1_right = rel1_right.sample(n=len(rel1_wrong), random_state=1)

print(f'Length of wrong object case is {len(rel1_wrong)}')
print(f'Length of right object case is {len(rel1_right)}')

rel_data=read_jsonl('/home/ubuntu/junkai/RLC-bench/answer/Relation/llava-v1.5/llava-v1.5-7b/YesandNo.jsonl')

rel_pd=pd.DataFrame(rel_data)

rel_pd['entropy'] = rel_pd.apply(lambda row: entropy(row['yes_prob'], row['no_prob']), axis=1)
rel_pd['decision_prob']= rel_pd[['yes_prob', 'no_prob']].max(axis=1)
rel_pd['response']=rel_pd['response'].apply(filter_and_replace)

rel_wrong=rel_pd[rel_pd['response']!=rel_pd['label']]
rel_right=rel_pd[rel_pd['response']==rel_pd['label']]
# equal_rel_right = rel_right.sample(n=len(rel_wrong), random_state=1)

print(f'Length of wrong object case is {len(rel_wrong)}')
print(f'Length of right object case is {len(rel_right)}')


min_entropy = 0  # 为了保持与之前代码相同的设置
max_entropy = 1  # 为了保持与之前代码相同的设置

# 定义区间
bins = np.arange(min_entropy, 0.9, 0.2)
labels = ["0", "0.2", "0.4", "0.6", "0.8"]

# 对第一组数据进行操作
rel_wrong_binned = pd.cut(rel_pd['entropy'], bins=np.append(bins, max_entropy), labels=labels, include_lowest=True).value_counts().sort_index()
equal_rel_right_binned = pd.cut(rel_right['entropy'], bins=np.append(bins, max_entropy), labels=labels, include_lowest=True).value_counts().sort_index()
ratios = rel_wrong_binned / (equal_rel_right_binned + 1e-6)

# 对第二组数据进行操作
rel1_wrong_binned = pd.cut(rel1_pd['entropy'], bins=np.append(bins, max_entropy), labels=labels, include_lowest=True).value_counts().sort_index()
equal_rel1_right_binned = pd.cut(rel1_right['entropy'], bins=np.append(bins, max_entropy), labels=labels, include_lowest=True).value_counts().sort_index()
ratios1 = rel1_wrong_binned / (equal_rel1_right_binned + 1e-6)

# 绘制柱状图
plt.figure(figsize=(6, 4))
width = 0.25  # 柱的宽度
plt.bar(labels, ratios, width=-width, align='edge', color='#EFD496', label='Reefknot')
plt.bar(labels, ratios1, width=width, align='edge', color='#9BB89C', label='MMRel')

plt.xlabel('Entropy')
plt.ylabel('Ratio', fontsize=14)
plt.title('Ratio between hallucination and no hallucination')
plt.legend(loc='upper left')

# 隐藏右侧和上侧的边框线
ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

plt.tight_layout()
plt.savefig('/home/ubuntu/junkai/RLC-bench/photo/combined_test.pdf', dpi=300, bbox_inches='tight')
plt.show()
