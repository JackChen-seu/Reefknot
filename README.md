# Reefknot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models

This repository contains the source code for Reefknot, which is a Multimodal Benchmark for Relation Hallucination Evaluation proposed in our paper [ “Reefknot: A Comprehensive Benchmark For Relation Hallucination Evaluation And Mitigation in Multimodal Large Language Models”](https://openreview.net/forum?id=aRQi5gHpcF)

Hallucination issues persistently plagued current multimodal large language models (MLLMs). While existing research primarily focuses on object-level or attribute-level hallucinations, sidelining the more sophisticated relation hallucinations that necessitate advanced reasoning abilities from MLLMs. Besides, recent benchmarks regarding relation hallucinations lack in-depth evaluation and effective mitigation. To handle the aforementioned challenges, we introduce Reefknot, the first comprehensive benchmark specifically targeting relation hallucinations, consisting of over 20,000 samples derived from real-world scenarios. Specifically, we first provide a systematic definition of relation hallucinations, integrating perspectives from perceptive and cognitive domains. Moreover, we construct the relation-based corpus utilizing the representative scene graph dataset Visual Genome (VG).

Our comprehensive evaluation across three distinct tasks revealed a substantial shortcoming in the capabilities of current MLLMs to mitigate issues related to relation hallucinations. Finally, we advance a novel confidence-based mitigation strategy tailored to tackle the relation hallucinations problem.

## Contents
* [Dataset](#dataset)
* [Mitigation](#mitigation)
* [Modules](#modules)
* [Add a New Task](#add-a-new-task)
* [Citation](#citation)

## Dataset
### Contruction Method

We first identify relation triplets from Visual Genome (VG) dataset (Phase a), and conduct triplet filtering (Phase b). Subsequently, we extract the semantic triplets (Phase c) and categorize their relations (Phase d). Then, a relation-based question set can be constructed into three types (Phase e). Finally, the quality of dataset is ensured by three rounds of expert-based validation (Phase f).
![](img/data_pipeline.png)

### Download

1. You need to download the photo from [Visual Genome Dataset](https://homes.cs.washington.edu/~ranjay/visualgenome/api.html) first and merge two image folder to one.

2. Our dataset consists of three jsonl files: YESNO.jsonl, Multichoice.jsonl, VQA.jsonl. Each case in jsonl file includes the following parts:
- `image_id`: Image ID in Visual Genome Dataset
- `query_prompt`: Quetion
- `label`: Ground Truth label
- `relation_type`: Type of relation, including perception and cognition.

## Mitigation

### Model Setup

## Checkpoint Setup
