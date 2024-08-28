import re
import csv

# analyze the predicted result and generate a csv report

# basic information about file directory
model_name = "lstm"     # transformer or lstm
input_file = "./80/" + model_name + "_test_output.txt"
output_file = "./80/" + model_name + "_test_report.csv"

# patterns
k_pattern = re.compile(r"----------------- K = (\d+) -----------------")
i_pattern = re.compile(r"\[i=(\d+)\]")
total_num_pattern = re.compile(r"Total Num: ([\d.]+)")
accurate_type_pattern = re.compile(r"Accurate Type Num: ([\d.]+)")
accurate_table_pattern = re.compile(r"Accurate Table Num: ([\d.]+)")
accurate_customer_pattern = re.compile(r"Accurate Customer Num: ([\d.]+)")
accurate_item_pattern = re.compile(r"Accurate Item Num: ([\d.]+)")

# initialize
data = []
current_k = None

# read files and analyze
with open(input_file, 'r') as file:
    for line in file:
        line = line.strip()

        # get K
        k_match = k_pattern.match(line)
        if k_match:
            current_k = int(k_match.group(1))
            continue

        # get i
        i_match = i_pattern.match(line)
        if i_match:
            current_i = int(i_match.group(1))
            total_num = 0
            accurate_type = 0
            accurate_table = 0
            accurate_customer = 0
            accurate_item = 0
            continue

        # get Total Num
        total_num_match = total_num_pattern.match(line)
        if total_num_match:
            total_num = float(total_num_match.group(1))

        # get Accurate Type Num
        accurate_type_match = accurate_type_pattern.match(line)
        if accurate_type_match:
            accurate_type = float(accurate_type_match.group(1))

        # get Accurate Table Num
        accurate_table_match = accurate_table_pattern.match(line)
        if accurate_table_match:
            accurate_table = float(accurate_table_match.group(1))

        # get Accurate Customer Num
        accurate_customer_match = accurate_customer_pattern.match(line)
        if accurate_customer_match:
            accurate_customer = float(accurate_customer_match.group(1))

        # get Accurate Item Num
        accurate_item_match = accurate_item_pattern.match(line)
        if accurate_item_match:
            accurate_item = float(accurate_item_match.group(1))
            data.append([current_k, current_i, total_num, accurate_type, accurate_table, accurate_customer, accurate_item])

# save to a csv file
with open(output_file, 'w+', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['K', 'i', 'Total Num', 'Accurate Type Num', 'Accurate Table Num', 'Accurate Customer Num', 'Accurate Item Num'])
    csvwriter.writerows(data)

print(f'Data successfully saved to {output_file}')
