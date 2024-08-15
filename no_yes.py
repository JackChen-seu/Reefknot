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
    layer_counts = len(entries[0]["yes_layer_prob"])
    yes_prob_sums = {str(i): 0 for i in range(layer_counts)}
    no_prob_sums = {str(i): 0 for i in range(layer_counts)}

    for entry in entries:
        for i in range(layer_counts):
            yes_prob_sums[str(i)] += entry["yes_layer_prob"][str(i)]
            no_prob_sums[str(i)] += entry["no_layer_prob"][str(i)]

    yes_prob_avgs = {
        str(i): yes_prob_sums[str(i)] / len(entries) for i in range(layer_counts)
    }
    no_prob_avgs = {
        str(i): no_prob_sums[str(i)] / len(entries) for i in range(layer_counts)
    }

    return yes_prob_avgs, no_prob_avgs


input_file = "/home/ubuntu/junkai/RLC-bench/test/confidence/minigptv2/yesno_resul.jsonl"
selected_entries = []
labels_responses = {
    "label:yes_response:yes": [],
    "label:yes_response:no": [],
    "label:no_response:yes": [],
    "label:no_response:no": [],
}

with open(input_file, "r") as file:
    for line in file:
        data = json.loads(line)
        yes_prob = data["prob_yes"]
        no_prob = data["prob_no"]

        ent = entropy(yes_prob, no_prob)

        # 选出熵大于0.9的示例
        if ent > 0.8:
            selected_entries.append(data)
            label_first_word = data["label"].lower()
            if "yes" in label_first_word:
                label_first_word = "yes"
            elif "no" in label_first_word:
                label_first_word = "no"
            response_first_word = data["response"].lower()
            if "yes" in response_first_word:
                response_first_word = "yes"
            elif "no" in response_first_word:
                response_first_word = "no"
            key = f"label:{label_first_word}_response:{response_first_word}"
            if key in labels_responses:
                labels_responses[key].append(data)

# 计算 label:no_response:yes 的平均概率
key = "label:no_response:yes"
if labels_responses[key]:
    yes_avg, no_avg = calculate_average_probabilities(labels_responses[key])

    # 创建图表
    fig, ax = plt.subplots(figsize=(7, 5))
    layers = list(map(int, yes_avg.keys()))
    yes_probs = list(yes_avg.values())
    no_probs = list(no_avg.values())

    ax.plot(layers, yes_probs, label="yes_prob", marker="o")
    ax.plot(layers, no_probs, label="no_prob", marker="x")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Probability")
    ax.set_title(f"Average Probabilities for {key}")
    ax.legend()
    ax.grid(True)

    # 保存并显示图表
    plt.tight_layout()
    plt.savefig("/home/ubuntu/junkai/RLC-bench/test/no_yes.png")
    plt.show()
else:
    print(f"No data available for {key}.")