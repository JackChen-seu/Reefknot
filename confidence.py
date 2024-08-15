import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.colors as mcolors
# 创建数据框
data = {
    'Model': ['LLaVA-7B', 'MiniGPT4v2-7B', 'Qwen-vl-7B', 'InstructBLIP-13B', 'LLaVA-13B'],
    'Hallucination Rate POPE': [15.79, 14.55, 13.19, 13.51, 14.81],
    'Hallucination Rate RLC': [44.90, 42.49, 41.02, 43.13, 45.43]
}

df = pd.DataFrame(data)

# 设置索引
df.set_index('Model', inplace=True)

# 绘图
fig, ax = plt.subplots(figsize=(8, 4))

# 位置和宽度设置
x = np.arange(len(df))
width = 0.35

# 绘制条形图
rects1 = ax.bar(x - width/2, df['Hallucination Rate POPE'], width, label='POPE', hatch='/', color='moccasin', edgecolor='orange')
rects2 = ax.bar(x + width/2, df['Hallucination Rate RLC'], width, label='Reefknot', hatch='\\', color='mistyrose', edgecolor='salmon')

# 设置图表标题和标签
ax.set_ylabel('Hallucination Rates (%)', fontsize=14)  # 调整字体大小
ax.set_xticks(x)
ax.set_xticklabels(df.index)
legend = ax.legend(loc='lower left', fontsize=12, framealpha=0.5)
# 调整图例字体的透明度
# 调整图例字体的透明度
for text in legend.get_texts():
    color = text.get_color()
    rgba_color = mcolors.to_rgba(color, alpha=0.7)  # 设置为50%透明度
    text.set_color(rgba_color)
# 放大下标字体
ax.tick_params(axis='x', labelsize=12)
ax.tick_params(axis='y', labelsize=12)
plt.xticks(rotation=0)
plt.tight_layout()

# 保存图表
plt.savefig('/home/ubuntu/junkai/RLC-bench/photo/test.pdf', dpi=400, bbox_inches='tight')
plt.show()
