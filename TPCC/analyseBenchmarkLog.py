import re
import csv

# 定义日志文件和输出CSV文件路径
LOG_FILE = '/opt/pg/tpcc/benchmark.log'
OUTPUT_CSV = '/opt/pg/tpcc/benchmark_results.csv'

# 定义正则表达式模式以匹配所需的日志信息
patterns = {
    "runfilepath": re.compile(r"runfilepath=(\/opt\/pg\/tpcc\/data-\d{2}\.csv)"), 
    "concurrencyControl": re.compile(r"concurrencyControl=([a-zA-Z]+)"),
    "kValue": re.compile(r"kValue=(\d+)"),  
    "needsRetry": re.compile(r"needsRetry=([a-zA-Z]+)"),
    "terminals": re.compile(r"terminals=(\d+)"),
    "Session Time": re.compile(r"Session Time\(s\)\s+=\s+([\d.Ee+-]+)"),
    "Predict Time": re.compile(r"Predict Time\(s\)\s+=\s+([\d.Ee+-]+)"),
    "Lock Time": re.compile(r"Lock Time\(s\)\s+=\s+([\d.Ee+-]+)"),
    "Abort Time": re.compile(r"Abort Time\(s\)\s+=\s+([\d.Ee+-]+)"),
    "Execute Time": re.compile(r"Execute Time\(s\)\s+=\s+([\d.Ee+-]+)"),
    "Transaction Count": re.compile(r"Transaction Count\s+=\s+(\d+)"),
    "Retry Number": re.compile(r"Retry Number\s+=\s+(\d+)"),
    "Aborted Count": re.compile(r"Aborted Count\s+=\s+(\d+)"),
    "Commit Count": re.compile(r"Commit Count\s+=\s+(\d+)"),
    "Waited Count": re.compile(r"Waited Count\s+=\s+(\d+)"),
    "Timeout Count": re.compile(r"Timeout Count\s+=\s+(\d+)"),
    "Error Count": re.compile(r"Error Count\s+=\s+(\d+)")
}

# 定义表头，将runfilepath放在最前面
headers = [
    "runfilepath", "concurrencyControl", "kValue", "needsRetry", "terminals",  # 新增kValue表头
    "Session Time", "Predict Time", "Lock Time", "Abort Time", 
    "Execute Time", "Transaction Count", "Retry Number", "Aborted Count", 
    "Commit Count", "Waited Count", "Timeout Count", "Error Count"
]

# 初始化结果列表
results = []

# 初始化记录上一个runfilepath, concurrencyControl, kValue 和 terminals 的变量
last_runfilepath = None
last_concurrencyControl = None
last_kValue = None  
last_terminals = None

# 解析日志文件
with open(LOG_FILE, 'r') as log_file:
    current_result = {key: None for key in headers}
    for line in log_file:
        for key, pattern in patterns.items():
            match = pattern.search(line)
            if match:
                current_result[key] = match.group(1)
        
        # 检查是否已收集到所有所需的信息
        if current_result["concurrencyControl"] and current_result["Error Count"]:
            # 如果 Predict Time 为空，设置为 0
            if current_result["Predict Time"] is None:
                current_result["Predict Time"] = '0'
            
            # 检查runfilepath, concurrencyControl, kValue 或 terminals 是否改变
            if (current_result["runfilepath"] != last_runfilepath or 
                current_result["concurrencyControl"] != last_concurrencyControl or 
                current_result["kValue"] != last_kValue or  # 添加kValue的比较条件
                current_result["terminals"] != last_terminals):
                
                # 添加一个空行
                results.append({key: '' for key in headers})
                last_runfilepath = current_result["runfilepath"]
                last_concurrencyControl = current_result["concurrencyControl"]
                last_kValue = current_result["kValue"]  # 更新last_kValue
                last_terminals = current_result["terminals"]
            
            # 将当前结果添加到结果列表
            results.append(current_result.copy())
            
            # 重置当前结果
            current_result = {key: None for key in headers}

# 将结果写入CSV文件
with open(OUTPUT_CSV, 'w', newline='') as csv_file:
    writer = csv.DictWriter(csv_file, fieldnames=headers)
    writer.writeheader()
    for result in results:
        writer.writerow(result)

print(f"结果已保存到 {OUTPUT_CSV}")
