# 文件路径列表
file_paths = [
    '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/answer/Relation/Minigpt4/cogintive/YesandNo.jsonl',
    '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/answer/Relation/Minigpt4/perception/YesandNo.jsonl'
]

# 新文件路径
output_file_path = '/home/ubuntu/hallu_team/junkai/Dataset/RLC-bench/answer/Relation/Minigpt4/Relation.jsonl'

# 打开新文件以写入
with open(output_file_path, 'w') as outfile:
    # 遍历每个源文件
    for file_path in file_paths:
        # 打开源文件以读取
        with open(file_path, 'r') as infile:
            # 将每行复制到新文件
            for line in infile:
                outfile.write(line)

print("Files have been successfully merged into:", output_file_path)
