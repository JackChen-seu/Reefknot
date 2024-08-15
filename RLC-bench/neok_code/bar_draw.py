import matplotlib.pyplot as plt
import numpy as np

# Data
models = ['Phi-3-4.2B', 'GLM4v-9B', 'Yi-VL-34B', 'MiniCPM', 'GPT-4o']
yn = [63.42, 69.93, 66.67, 70.46, 70.92]
mcq = [67.6, 73.45, 76.07, 73, 80.07]
vqa = [50.22, 42.81, 45.09, 53.31, 53.98]

# Bar positions
bar_width = 0.5
r1 = np.arange(len(models))

# Plotting
plt.figure(figsize=(8, 5))

# Bars
plt.barh(r1, yn, color='linen', edgecolor='black', height=bar_width, label='Y/N')
plt.barh(r1, mcq, color='bisque', edgecolor='black', left=yn, height=bar_width, label='MCQ')
plt.barh(r1, vqa, color='mistyrose', edgecolor='black', left=np.add(yn, mcq), height=bar_width, label='VQA')

# Labels and Titles
plt.xlabel('Scores', fontsize=13)
plt.ylabel('Models', fontsize=13)
plt.yticks(r1, models, fontsize=11)
plt.xticks(fontsize=11)
plt.title('Model Performance Comparison')

# Adding Data Labels
for i in range(len(r1)):
    plt.text(yn[i] / 2, i, str(yn[i]), ha='center', va='center', color='black', fontsize=11)
    plt.text(yn[i] + mcq[i] / 2, i, str(mcq[i]), ha='center', va='center', color='black', fontsize=11)
    plt.text(yn[i] + mcq[i] + vqa[i] / 2, i, str(vqa[i]), ha='center', va='center', color='black', fontsize=11)

# Legend
plt.legend(loc='lower right')
plt.savefig('./bar2.pdf', dpi=400,bbox_inches='tight')
# Show plot
plt.show()