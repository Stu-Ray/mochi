import re
import csv

# Analyze the txt file generated when training models (includes training history) and generate a csv file to record loss and accuracy

# input and output file name and directory
input_filename = "Transformer"
input_file = "./Dataset/" + input_filename + ".txt"
output_file = "./Output/training/" + input_filename +"_metrics.csv"

epoch_pattern = r"Epoch\s+(\d+)/\d+"
step_metrics_pattern = r"(\d+)/\d+\s+\[.*?\]\s+-\s+ETA:.*?\s+-\s+loss:\s+([0-9\.eE\-]+)\s+-\s+accuracy:\s+([\d\.]+)"
final_step_metrics_pattern = r"(\d+)/\d+\s+\[.*?\]\s+-\s+(\d+s\s+\d+ms/step|\d+s\s+\d+s/step)\s+-\s+loss:\s+([0-9\.eE\-]+)\s+-\s+accuracy:\s+([\d\.]+)"

# all the data is stored here
data = []

# convert time string to seconds and milliseconds
def convert_time_to_seconds(time_str):
    time_match = re.match(r"(\d+)s\s+(\d+)(ms|s)/step", time_str)
    if time_match:
        seconds = int(time_match.group(1))
        ms_or_s_value = int(time_match.group(2))
        unit = time_match.group(3)
        if unit == "ms":
            ms_or_s_value = ms_or_s_value / 1000
        return seconds + ms_or_s_value
    else:
        return 0


# open txt file
with open(input_file, "r") as file:
    for line in file:
        # get the epoch count
        epoch_match = re.search(epoch_pattern, line)
        if epoch_match:
            current_epoch = epoch_match.group(1)

        # get loss, accuracy
        step_metrics_match = re.search(step_metrics_pattern, line)
        if step_metrics_match:
            step = step_metrics_match.group(1)
            loss_str = step_metrics_match.group(2)
            accuracy = step_metrics_match.group(3)
            time_in_seconds = None
            loss = float(loss_str)
            data.append([current_epoch, step, time_in_seconds, loss, accuracy])

        final_step_metrics_match = re.search(final_step_metrics_pattern, line)
        if final_step_metrics_match:
            step = final_step_metrics_match.group(1)
            time_str = final_step_metrics_match.group(2)
            loss_str = final_step_metrics_match.group(3)
            accuracy = final_step_metrics_match.group(4)
            time_in_seconds = convert_time_to_seconds(time_str)
            loss = float(loss_str)
            data.append([current_epoch, step, time_in_seconds, loss, accuracy])

# make sure the file exists
import os

os.makedirs(os.path.dirname(output_file), exist_ok=True)

# write into the csv file
with open(output_file, "w", newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Step", "Time (seconds)", "Loss", "Accuracy"])
    csvwriter.writerows(data)