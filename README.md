# Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models

> **Update:** This work has been **accepted to ACL 2025 (Finding)**.

This repository contains the source code for **Reefknot**, a multimodal benchmark for relation hallucination evaluation proposed in our paper, [“Reefknot: A Comprehensive Benchmark For Relation Hallucination Evaluation And Mitigation in Multimodal Large Language Models”](https://openreview.net/forum?id=aRQi5gHpcF).

Hallucination issues have persistently plagued current multimodal large language models (MLLMs). Existing research primarily focuses on object-level or attribute-level hallucinations, while sidelining the more sophisticated **relation hallucinations** that require advanced reasoning abilities. Moreover, recent benchmarks addressing relation hallucinations often lack thorough evaluation and effective mitigation strategies. To address these challenges, we introduce **Reefknot**, the first comprehensive benchmark specifically targeting relation hallucinations, consisting of over 20,000 samples drawn from real-world scenarios. We provide a systematic definition of relation hallucinations by integrating perspectives from perceptual and cognitive domains, and construct a relation-based corpus using the representative scene graph dataset **Visual Genome (VG)**.

Our comprehensive evaluation across three distinct tasks reveals substantial shortcomings in current MLLMs’ ability to mitigate relation hallucinations. Finally, we propose a novel **confidence-based mitigation** strategy tailored to this problem.

## Contents

* [Dataset](#dataset)
* [Mitigation](#mitigation)
* [Usage](#usage)
* [Add a New Task](#add-a-new-task)
* [Citation](#citation)

## Dataset

### Construction Method

We first identify relation triplets from the Visual Genome (VG) dataset (Phase a) and conduct triplet filtering (Phase b). Subsequently, we extract semantic triplets (Phase c) and categorize their relations (Phase d). Then, we construct a relation-based question set with three types (Phase e). Finally, we ensure dataset quality through three rounds of expert-based validation (Phase f).

![](img/data_pipeline.png)

### Download

1. Download the images from the [Visual Genome Dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html).
   Then extract the two archives **images1.zip** and **images2.zip** into the **same** directory. The resulting folder structure should look like:

   ```text
   VG_dataset
   ├── VG_100K
   │   ├── 1.jpg
   │   ├── 2.jpg
   │   └── ...
   └── VG_100K_2
       ├── 100001.jpg
       └── ...
   ```

2. Clone this repository and set up the environment:

   ```shell
   git clone https://github.com/JackChen-seu/Reefknot.git
   cd Reefknot
   conda create -yn Reefknot python=3.9
   conda activate Reefknot
   cd LLaVA
   pip install --upgrade pip
   pip install -e .
   ```

3. Our dataset consists of three `.jsonl` files: `YESNO.jsonl`, `Multichoice.jsonl`, and `VQA.jsonl`. Each entry in a JSONL file includes:

   * `image_id`: Image ID in the Visual Genome dataset
   * `query_prompt`: Question
   * `label`: Ground-truth label
   * `relation_type`: Type of relation, including **perception** and **cognition**

## Mitigation

![](img/method_case.png)

### Model Setup

We provide LLaVA-based code as an example for running the mitigation.

1. Download the LLaVA checkpoint and the vision encoder from [LLaVA](https://huggingface.co/liuhaotian/llava-v1.5-13b) and [Vision Encoder](https://huggingface.co/openai/clip-vit-large-patch14-336).

2. In `/LLaVA/infer_LLaVA_yesandno.py`, **replace the file paths at lines 39 and 40** with the paths to your extracted `VG_100K` and `VG_100K_2` directories.

3. Move DTC.py  to `/LLaVA/llava/`.

## Usage

Run `infer.sh`, which contains the following command:

```shell
export CUDA_VISIBLE_DEVICES=0
python LLaVA/infer_LLaVA_yesandno.py \
    --model-path PATH_TO_LLaVA_CHECKPOINT \
    --question-file PATH_TO_QUESTION_FILE \
    --answers-file PATH_TO_ANSWER_FILE \
    --temperature 0 \
    --conv-mode vicuna_v1 \
    --apha APHA \
    --layer LAYER_NUM \
    --threshold ENT_THREAHOLD
```

**Hyperparameters.** We use `apha=0.1`, `layer=38`, `threshold=0.9`.

### Tips & Citation

* We sincerely thank the open-source project [LLaVA](https://github.com/haotian-liu/LLaVA). For convenience in environment setup, we recommend installing by directly integrating LLaVA’s code as described above.


* Additionally, from a technical point, this algorithm is primarily implemented by applying **monkey patches**; the core code is located at `'LLaVA/llava/eval/DTC.py'`. If you are interested in the hidden states of MLLMs/LLMs, you should read this file carefully (GPT can be a great assistant for reading code).

If you find our work useful, please cite:

```
@misc{reefknot,
      title={Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models}, 
      author={Kening Zheng and Junkai Chen and Yibo Yan and Xin Zou and Xuming Hu},
      year={2025},
      eprint={2408.09429},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.09429}, 
}
```
