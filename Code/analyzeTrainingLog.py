import re
import csv

# 用来分析训练后生成的txt文件（loss\ACC等数据），并整理成一个csv文件

# 指定输入和输出文件路径
input_filename = "Transformer"

input_file = "./Dataset/" + input_filename + ".txt"
output_file = "./Output/training/" + input_filename +"_metrics.csv"

# 正则表达式匹配模式
epoch_pattern = r"Epoch\s+(\d+)/\d+"
step_metrics_pattern = r"(\d+)/\d+\s+\[.*?\]\s+-\s+ETA:.*?\s+-\s+loss:\s+([0-9\.eE\-]+)\s+-\s+accuracy:\s+([\d\.]+)"
final_step_metrics_pattern = r"(\d+)/\d+\s+\[.*?\]\s+-\s+(\d+s\s+\d+ms/step|\d+s\s+\d+s/step)\s+-\s+loss:\s+([0-9\.eE\-]+)\s+-\s+accuracy:\s+([\d\.]+)"

# 初始化数据列表
data = []

# 定义一个函数来转换时间为纯数字
def convert_time_to_seconds(time_str):
    # 提取秒和毫秒
    time_match = re.match(r"(\d+)s\s+(\d+)(ms|s)/step", time_str)
    if time_match:
        seconds = int(time_match.group(1))
        ms_or_s_value = int(time_match.group(2))
        unit = time_match.group(3)
        # 如果是ms，则将毫秒转化为秒的形式
        if unit == "ms":
            ms_or_s_value = ms_or_s_value / 1000
        # 返回总的秒数
        return seconds + ms_or_s_value
    else:
        return 0


# 打开txt文件并逐行读取
with open(input_file, "r") as file:
    for line in file:
        # 提取 epoch 信息
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            current_epoch = epoch_match.group(1)

        # 提取每步的时间、loss 和 accuracy 信息
        step_metrics_match = re.search(step_metrics_pattern, line)
        if step_metrics_match:
            step = step_metrics_match.group(1)
            loss_str = step_metrics_match.group(2)
            accuracy = step_metrics_match.group(3)
            # 无法提取时间，因此设为 None
            time_in_seconds = None
            # 将 Loss 转换为浮点数
            loss = float(loss_str)
            # 将数据存入列表
            data.append([current_epoch, step, time_in_seconds, loss, accuracy])

        # 提取最后一步的时间、loss 和 accuracy 信息
        final_step_metrics_match = re.search(final_step_metrics_pattern, line)
        if final_step_metrics_match:
            step = final_step_metrics_match.group(1)
            time_str = final_step_metrics_match.group(2)
            loss_str = final_step_metrics_match.group(3)
            accuracy = final_step_metrics_match.group(4)
            # 转换时间为纯数字
            time_in_seconds = convert_time_to_seconds(time_str)
            # 将 Loss 转换为浮点数
            loss = float(loss_str)
            # 将数据存入列表
            data.append([current_epoch, step, time_in_seconds, loss, accuracy])

# 确保输出目录存在
import os

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# 将数据写入 CSV 文件
with open(output_file, "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    # 写入表头
    csvwriter.writerow(["Epoch", "Step", "Time (seconds)", "Loss", "Accuracy"])
    # 写入数据
    csvwriter.writerows(data)

print(f"数据已成功写入 {output_file}")

# # 正则表达式匹配模式
# epoch_pattern = r"Epoch\s+(\d+)/\d+"
# final_metrics_pattern = r"(\d+)/\d+\s+\[.*?\]\s+-\s+(\d+s\s+\d+ms/step|\d+s\s+\d+s/step)\s+-\s+loss:\s+([0-9\.eE\-]+)\s+-\s+accuracy:\s+([\d\.]+)"
#
# # 初始化数据列表
# data = []
#
# # 定义一个函数来转换时间为纯数字
# def convert_time_to_seconds(time_str):
#     # 提取秒和毫秒
#     time_match = re.match(r"(\d+)s\s+(\d+)(ms|s)/step", time_str)
#     if time_match:
#         seconds = int(time_match.group(1))
#         ms_or_s_value = int(time_match.group(2))
#         unit = time_match.group(3)
#         # 如果是ms，则将毫秒转化为秒的形式
#         if unit == "ms":
#             ms_or_s_value = ms_or_s_value / 1000
#         # 返回总的秒数
#         return seconds + ms_or_s_value
#     else:
#         return 0
#
#
# # 打开txt文件并逐行读取
# with open(input_file, "r") as file:
#     for line in file:
#         # 提取 epoch 信息
#         epoch_match = re.search(epoch_pattern, line)
#         if epoch_match:
#             current_epoch = epoch_match.group(1)
#
#         # 提取每轮最后一步的时间、loss 和 accuracy 信息
#         final_metrics_match = re.search(final_metrics_pattern, line)
#         if final_metrics_match:
#             time_str = final_metrics_match.group(2)
#             loss_str = final_metrics_match.group(3)
#             accuracy = final_metrics_match.group(4)
#             # 转换时间为纯数字
#             time_in_seconds = convert_time_to_seconds(time_str)
#             # 将 Loss 转换为浮点数
#             loss = float(loss_str)
#             # 将数据存入列表
#             data.append([current_epoch, time_in_seconds, loss, accuracy])
#
# # 确保输出目录存在
# import os
#
# os.makedirs(os.path.dirname(output_file), exist_ok=True)
#
# # 将数据写入 CSV 文件
# with open(output_file, "w", newline='') as csvfile:
#     csvwriter = csv.writer(csvfile)
#     # 写入表头
#     csvwriter.writerow(["Epoch", "Time (seconds)", "Loss", "Accuracy"])
#     # 写入数据
#     csvwriter.writerows(data)
#
# print(f"数据已成功写入 {output_file}")