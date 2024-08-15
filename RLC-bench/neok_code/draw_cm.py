import pandas as pd
import json
import pandas as pd
import os
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score
import argparse

def read_jsonl(file_path):
    a=[]
    with open(file_path, 'r') as file:
        
        for line in file:
            json_obj = json.loads(line)
            a.append(json_obj)
    return a

def filter_and_replace(value):
    value=value.lower()
    if 'no' in value:
        return 'no'
    else:
        return 'yes'
    
def MCQ_filter(value):
    if 'A' in value:
        return 'A'
    elif 'B' in value:
        return  'B'
    elif 'C' in value:
        return 'C'
    else:   
        return 'D'
    
def draw(args):
    pwd = os.path.join(args.pwd, args.model_name)
    yesno_path=os.path.join(pwd, 'yesno_result.jsonl')
    # yesno_path=pwd+'yesno_result.jsonl'
    # path='/home/ubuntu/kening/kening/kening_results/minicpm-v-v2_5-chat/yesno_result.jsonl'
    data=read_jsonl(yesno_path)
    df=pd.DataFrame(data)
    df['response'] = df['response'].apply(filter_and_replace)
    cm_binary = confusion_matrix(df['label'], df['response'], labels=['yes', 'no'])

    accuracy = accuracy_score(df['label'], df['response'])
    precision = precision_score(df['label'], df['response'], pos_label='yes')
    f1 = f1_score(df['label'], df['response'], pos_label='yes')

    # 打印结果
    print(f"Yes/No 的值相关值")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    # 创建子图
    fig, axes = plt.subplots(1, 2, figsize=(6 , 3))
    # 绘制二分类混淆矩阵
    sns.heatmap(cm_binary, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                annot_kws={"size": 12}, linewidths=0.5, linecolor='gray',
                xticklabels=[' Yes', ' No'],
                yticklabels=['Yes', 'No'], ax=axes[0])
    for spine in axes[0].spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

    axes[0].set_ylabel('Label', fontsize=10,labelpad=2)
    axes[0].set_xlabel('Predicted', fontsize=10)
    axes[0].set_title('Yes/No Questions', fontsize=10, pad=10)
    axes[0].tick_params(axis='x', rotation=0)
    axes[0].tick_params(axis='y', rotation=0)

    # 绘制多分类混淆矩阵
    mcq_path=os.path.join(pwd, 'Multichoice_result.jsonl')
    # mcq_path=pwd+'Multichoice_result.jsonl'
    # mcq_path='/home/ubuntu/kening/kening/kening_results/minicpm-v-v2_5-chat/Multichoice_result.jsonl'
    data=read_jsonl(mcq_path)
    df=pd.DataFrame(data)
    df['response'] = df['response'].apply(MCQ_filter)
    confusion_mcq = confusion_matrix(df['label'], df['response'], labels=['A', 'B','C','D'])
    print('多选题的confusion matrix:')
    print(confusion_mcq)
    sns.heatmap(confusion_mcq, annot=True, fmt='d', cmap='YlGnBu', cbar=False,
                annot_kws={"size": 12}, linewidths=0.5, linecolor='gray',
                xticklabels=[' A', ' B', ' C', ' D'],
                yticklabels=['A', ' B', 'C',  'D'], ax=axes[1])

    # axes[1].set_ylabel('Label', fontsize=10)
    axes[1].set_xlabel('Predicted', fontsize=10)
    axes[1].set_title('Multiple Choice Questions', fontsize=10, pad=10)
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].tick_params(axis='y', rotation=0)

    # 调整布局以适应所有文本
    plt.tight_layout()

    # 显示图形
    plt.show()

    # 计算和打印准确率、精确率和F1分数
    accuracy = accuracy_score(df['label'], df['response'])
    precision = precision_score(df['label'], df['response'], average='weighted')
    f1 = f1_score(df['label'], df['response'], average='weighted')

    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"F1 Score: {f1:.2f}")
    save_name='CM'+args.model_name+'.pdf'
    plt.savefig(save_name, dpi=400,format='pdf')

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='draw confusion matrix')
    parser.add_argument('--pwd', type=str, default='/home/ubuntu/kening/kening/kening_results/', help='path to the result')
    parser.add_argument('--model_name', type=str, default='qwen-vl-chat', help='model name')
    # parser.add_argument('--save_path', type=str, default='/home/ubuntu/kening/kening/kening_results/minicpm-v-v2_5-chat/cm.pdf', help='path to save the result')
    args = parser.parse_args()
    draw(args)