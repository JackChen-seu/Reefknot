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


input_file = "/home/ubuntu/junkai/RLC-bench/test/MMRel/minigptv2/yesno_result.jsonl"
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

# 计算 label:yes_response:no 的平均概率
key = "label:yes_response:no"
if labels_responses[key]:
    yes_avg, no_avg = calculate_average_probabilities(labels_responses[key])
    print(yes_avg)
    print(no_avg)
    # yes_avg = {'0': 3.212690353393555e-05, '1': 3.174565873056088e-05, '2': 3.1606206354105246e-05, '3': 3.3453950342142356e-05, '4': 3.6807555072712446e-05, '5': 3.678168890611181e-05, '6': 4.510497147182249e-05, '7': 5.2183304192884916e-05, '8': 8.423148461107937e-05, '9': 0.00013881246998624983, '10': 0.0001581749826107385, '11': 0.00016971579137838112, '12': 0.00045224855530936765, '13': 0.0006592003804332806, '14': 0.0008655269190950214, '15': 0.001263915367846219, '16': 0.035882481988870875, '17': 0.15908755896226415, '18': 0.1710159013856132, '19': 0.14576577240566038, '20': 0.18745163251768868, '21': 0.24561237839033018, '22': 0.26713705962558965, '23': 0.2508245504127358, '24': 0.3059899672022406, '25': 0.1538120485701651, '26': 0.13227153274248232, '27': 0.17746489902712265, '28': 0.12192685199233722, '29': 0.5346633623231132, '30': 0.6107166218307784, '31': 0.7721753390330188, '32': 0.9103036556603774, '33': 0.9597766804245284, '34': 0.9862175707547169, '35': 0.9964530512971698, '36': 0.9916623673349056, '37': 0.9497577019457547, '38': 0.9998525943396226, '39': 0.22793656475139115, '40': 0.41815761350235847}
    # no_avg = {'0': 3.1888484954833984e-05, '1': 2.8847523455349904e-05, '2': 2.614282212167416e-05, '3': 2.2711618891302146e-05, '4': 2.142618287284419e-05, '5': 1.7869022657286445e-05, '6': 1.9155583291683558e-05, '7': 1.7640725621637308e-05, '8': 2.1148402735872088e-05, '9': 1.9649289688974056e-05, '10': 2.4138756518094044e-05, '11': 3.120359384788657e-05, '12': 4.3804915446155475e-05, '13': 5.940108929040297e-05, '14': 8.30112763170926e-05, '15': 0.000178530531109504, '16': 0.0004966146541091631, '17': 0.0202563427079398677, '18': 0.06016233268773780678, '19': 0.04010740532065337559, '20': 0.041128574587264151, '21': 0.03857657594500847, '22': 0.04566891364331516, '23': 0.02968077893527049, '24': 0.06234005046340655, '25': 0.023, '26': 0.0026, '27': 0.04, '28': 0.038701522395295917, '29': 0.024383792292396977, '30': 0.102470895929156609, '31': 0.07446477251232795, '32': 0.0809151584697219561, '33': 0.062323747805829318, '34': 0.023339022960302965, '35': 0.0017365239701181087, '36': 0.008176454957926049, '37': 0.05024011629932332, '38': 0.00015927710623111365, '39': 0.7720693912146226, '40': 0.5814738723466981}
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

    # 保存并显示图表
    plt.tight_layout()
    plt.savefig("/home/ubuntu/junkai/RLC-bench/test/MMRel_minigpt4_13b_label_yes_response_no.png")
    plt.show()
else:
    print(f"No data available for {key}.")
