import json

def count_unique_image_ids(jsonl_files):
    unique_ids = set()  # 使用集合来存储不重复的image_id

    # 遍历所有文件
    for file_name in jsonl_files:
        with open(file_name, 'r', encoding='utf-8') as file:
            for line in file:
                # 解析每行的JSON数据
                data = json.loads(line)
                # 添加image_id到集合中
                unique_ids.add(data["image_id"])

    # 返回不重复的image_id数量
    return len(unique_ids)

# 假设jsonl文件列表如下
jsonl_files = ['/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/bench/Relation/cogintive/benchmark-3-YesandNo.jsonl', '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/bench/Relation/cogintive/benchmark-3-VQA.jsonl', '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/bench/Relation/cogintive/benchmark-3-multichoice.jsonl', '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/bench/Relation/perception/benchmark-3-multichoice.jsonl', '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/bench/Relation/perception/benchmark-3-VQA.jsonl', '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/bench/Relation/perception/benchmark-3-YesandNo.jsonl']

# 调用函数并打印结果
unique_count = count_unique_image_ids(jsonl_files)
print(f"Total number of unique image IDs: {unique_count}")