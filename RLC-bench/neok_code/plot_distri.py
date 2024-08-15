import re
from numpy import average
import pandas as pd
import matplotlib.pyplot as plt
import pickle


# 读取并处理数据
data_df = pd.read_csv('/home/ubuntu/kening/kening/RLC-bench/RLC-bench/neok_code/0_shot_gpt_4o_mini_label_distribution.csv', index_col=0)
data_df = data_df.div(data_df.sum(0), axis=1) * 100

data_df = data_df.rename(
    index={'YES': 'Fact.', 'NO': 'Non-Fact.', 'NOT SURE ENOUGH': 'NEI'},
    columns={
        'gold_overall': 'Gold Overall',
        'pred_overall': 'Pred Overall',
        'gold_multi-hop reasoning': 'Gold Multi-Hop',
        'pred_multi-hop reasoning': 'Pred Multi-Hop',
        'gold_combining structural and unstructural': 'Gold Struct. & Unstruct.',
        'pred_combining structural and unstructural': 'Pred Struct. & Unstruct.',
        'gold_spatiotemporal cognition': 'Gold Spatiotemporal',
        'pred_spatiotemporal cognition': 'Pred Spatiotemporal',
        'gold_composition understanding': 'Gold Composition',
        'pred_composition understanding': 'Pred Composition',
        'gold_arithmetic calculation': 'Gold Arithmetic',
        'pred_arithmetic calculation': 'Pred Arithmetic',
    }
)

# 分离 Gold 和 Pred 数据
gold_df = data_df.filter(like='Gold', axis=1)
pred_df = data_df.filter(like='Pred', axis=1)
gold_df = gold_df.rename(columns={c: c.replace('Gold ', '') for c in gold_df.columns})
pred_df = pred_df.rename(columns={c: c.replace('Pred ', '') for c in pred_df.columns})

# 创建分组堆叠柱状图并增加画布长度
fig, ax = plt.subplots(figsize=(8, 4))  # 调整画布大小，使其更加紧凑

# 绘制 Gold 数据的柱状图
gold_bars = gold_df.T.plot(kind='barh', stacked=True, ax=ax, color=['#aec7e8', '#ffbb78', '#98df8a'], alpha=0.7, position=1, width=0.3, edgecolor='none', zorder=2)
# 绘制 Pred 数据的柱状图
pred_bars = pred_df.T.plot(kind='barh', stacked=True, ax=ax, color=['#aec7e8', '#ffbb78', '#98df8a'], alpha=0.3, position=0, width=0.3, edgecolor='none', zorder=1)

# 统一外框线颜色为黑色，并覆盖在前面的条形图上
for bars in [gold_bars, pred_bars]:
    for bar in bars.patches:
        ax.add_patch(plt.Rectangle((bar.get_x(), bar.get_y()), bar.get_width(), bar.get_height(),
                                   fill=False, edgecolor='black', linewidth=1.2, zorder=3, alpha=0.4))

# 添加比例虚线并在刻度线上标注百分比说明
for pct in [0, 25, 50, 75, 100]:
    ax.axvline(pct, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.text(pct, ax.get_ylim()[1], f'{pct}%', ha='center', va='bottom', fontsize=10, color='gray')

# 设置图例并标注颜色对应标签
fact_colors = {'Fact.': '#aec7e8', 'Non-Fact.': '#ffbb78', 'NEI': '#98df8a'}
legend_labels = [
    plt.Line2D([0], [0], color=fact_colors['Fact.'], lw=4, alpha=0.3, label='Pred Fact.'),
    plt.Line2D([0], [0], color=fact_colors['Fact.'], lw=4, label='Gold Fact.'),
    plt.Line2D([0], [0], color=fact_colors['Non-Fact.'], lw=4, alpha=0.3, label='Pred Non-Fact.'),
    plt.Line2D([0], [0], color=fact_colors['Non-Fact.'], lw=4, label='Gold Non-Fact.'),
    plt.Line2D([0], [0], color=fact_colors['NEI'], lw=4, alpha=0.3, label='Pred NEI'),
    plt.Line2D([0], [0], color=fact_colors['NEI'], lw=4, label='Gold NEI'),
]

# 调整图例位置，将其放置在图像下方并平铺成长条
ax.legend(handles=legend_labels, bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=3, fontsize=10)

# 去掉图框的上、下、右边框线
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# 去掉X轴刻度
ax.xaxis.set_visible(False)

# 减少不同类别条形图之间的间距，并将左侧标签倾斜
ax.set_yticks(range(len(gold_df.columns)))
ax.set_yticklabels(gold_df.columns, fontsize=12, rotation=45, ha='right')  # 设置倾斜角度

ax.set_ylim(-0.5, len(gold_df.columns) - 0.5)

# 调整布局，减少留白
plt.tight_layout()  # 增加紧凑性
plt.subplots_adjust(left=0.2, right=0.85, top=0.9, bottom=0.2, hspace=0.1)  # 增加 bottom 调整空间

plt.savefig('distribution.pdf', dpi=400, bbox_inches='tight', format='pdf')