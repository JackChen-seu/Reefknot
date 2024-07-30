import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 创建数据框
data = {
    'Model': ['LLaVA 7b', 'LLaVA 13b', 'Minigpt4 7b', 'InstructBLIP 13b', 'Qwen 7b'],
    'Hallucination Rate POPE': [15.79, 14.81, 14.55, 13.51, 13.19],
    'Hallucination Rate RLC': [44.90, 45.43, 42.49, 43.13, 41.02]
}

df = pd.DataFrame(data)

# 设置索引
df.set_index('Model', inplace=True)

# 绘图
fig, ax = plt.subplots(figsize=(8, 3))

# 位置和宽度设置
x = np.arange(len(df))
width = 0.35

# 绘制条形图
rects1 = ax.bar(x - width/2, df['Hallucination Rate POPE'], width, label='Hallucination Rate Of POPE', hatch='*', color='moccasin', edgecolor='orange')
rects2 = ax.bar(x + width/2, df['Hallucination Rate RLC'], width, label='Hallucination Rate Of RLC', hatch='.', color='mistyrose', edgecolor='salmon')

# 设置图表标题和标签
ax.set_ylabel('Hallucination Rates (%)', fontsize=14)  # 调整字体大小
ax.set_xticks(x)
ax.set_xticklabels(df.index)
ax.legend()
ax.legend(loc='lower left', fontsize=12)
# 放大下标字体
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.xticks(rotation=0)
plt.tight_layout()

# 保存图表
plt.savefig('/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/photo/Hallucination_Rates_Chart.png', dpi=300, bbox_inches='tight')
plt.show()
