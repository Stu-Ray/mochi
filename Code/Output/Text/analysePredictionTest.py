import re
import csv

# 分析预测测试生成的文件，并总结成一个.csv的报告

# 文件路径
model_name = "lstm"     # transformer or lstm
input_file = "./80/" + model_name + "_test_output.txt"
output_file = "./80/" + model_name + "_test_report.csv"

# 正则表达式模式
k_pattern = re.compile(r"----------------- K = (\d+) -----------------")
i_pattern = re.compile(r"\[i=(\d+)\]")
total_num_pattern = re.compile(r"Total Num: ([\d.]+)")
accurate_type_pattern = re.compile(r"Accurate Type Num: ([\d.]+)")
accurate_table_pattern = re.compile(r"Accurate Table Num: ([\d.]+)")
accurate_customer_pattern = re.compile(r"Accurate Customer Num: ([\d.]+)")
accurate_item_pattern = re.compile(r"Accurate Item Num: ([\d.]+)")

# 初始化变量
data = []
current_k = None

# 读取文件并解析
with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()

        # 检测 K 的值
        k_match = k_pattern.match(line)
        if k_match:
            current_k = int(k_match.group(1))
            continue

        # 检测 i 的值
        i_match = i_pattern.match(line)
        if i_match:
            current_i = int(i_match.group(1))
            total_num = 0
            accurate_type = 0
            accurate_table = 0
            accurate_customer = 0
            accurate_item = 0
            continue

        # 检测并提取 Total Num
        total_num_match = total_num_pattern.match(line)
        if total_num_match:
            total_num = float(total_num_match.group(1))

        # 检测并提取 Accurate Type Num
        accurate_type_match = accurate_type_pattern.match(line)
        if accurate_type_match:
            accurate_type = float(accurate_type_match.group(1))

        # 检测并提取 Accurate Table Num
        accurate_table_match = accurate_table_pattern.match(line)
        if accurate_table_match:
            accurate_table = float(accurate_table_match.group(1))

        # 检测并提取 Accurate Customer Num
        accurate_customer_match = accurate_customer_pattern.match(line)
        if accurate_customer_match:
            accurate_customer = float(accurate_customer_match.group(1))

        # 检测并提取 Accurate Item Num
        accurate_item_match = accurate_item_pattern.match(line)
        if accurate_item_match:
            accurate_item = float(accurate_item_match.group(1))
            # 完成当前 i 的记录
            data.append([current_k, current_i, total_num, accurate_type, accurate_table, accurate_customer, accurate_item])

# 保存到 CSV 文件
with open(output_file, 'w+', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # 写入表头
    csvwriter.writerow(['K', 'i', 'Total Num', 'Accurate Type Num', 'Accurate Table Num', 'Accurate Customer Num', 'Accurate Item Num'])
    # 写入数据
    csvwriter.writerows(data)

print(f'Data successfully saved to {output_file}')
