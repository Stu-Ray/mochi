import csv
import random
import ast
import re

# 读取CSV文件
logid = 80
random_factor = 0.6
# input_file = './Text/Data_' + str(logid) + '_output.csv'
input_file = './Text/Data_output.csv'
output_file = './Random_' + str(int(random_factor*100)) + '_output.csv'

# 解析元组字符串为元组列表
def parse_tuple_string(tuple_str):
    tuple_pattern = re.compile(r'\(([^)]+)\)')
    tuples = tuple_pattern.findall(tuple_str)
    return [ast.literal_eval(f'({t})') for t in tuples]

# 将元组列表转回字符串
def tuple_list_to_string(tuple_list):
    return ' '.join(str(t) for t in tuple_list)

with open(input_file, mode='r', newline='') as file:
    reader = csv.DictReader(file)
    rows = list(reader)

# 随机选择20%的行进行修改
num_rows_to_modify = max(1, int(len(rows) * random_factor))
rows_to_modify = random.sample(rows, num_rows_to_modify)

for row in rows_to_modify:
    output_tuples = parse_tuple_string(row['output'])
    # 修改最后一个元组的最后一个值
    random_num1 = random.randint(0, 122)
    random_num2 = random.randint(0, 122)

    sec_last_tuple = list(output_tuples[-2])
    third_last_tuple = list(output_tuples[-3])
    forth_last_tuple = list(output_tuples[-4])

    sec_last_tuple[-1] =    random_num1
    third_last_tuple[-1] =    random_num2
    forth_last_tuple[-1] =    random_num1

    output_tuples[-2] = tuple(sec_last_tuple)
    output_tuples[-3] = tuple(sec_last_tuple)
    output_tuples[-4] = tuple(sec_last_tuple)
    # 将修改后的元组列表转回字符串
    row['output'] = tuple_list_to_string(output_tuples)

# 写回CSV文件
with open(output_file, mode='w+', newline='') as file:
    fieldnames = rows[0].keys()
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print("已写入文件")
