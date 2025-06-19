import re
import os
import csv
import random
from collections import defaultdict
import time
import numpy as np
import pandas as pd
import word2Vec
import dataProcessor as dp
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout
from keras.models import Model
import Transformer

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], 'GPU')

tf.config.optimizer.set_jit(False)

# 模型参数设置
v_size  =   5
w_size  =   20

in_dim      = 12
maxlen      = 13
embed_dim   = 16
num_heads   = 4
ff_dim      = 16
num_transformer_blocks = 4

b_size          = 32
old_epoch_num   = 1000
epoch_num       = 3000

# 其他参数
new_order       =   0
payment         =   0
k_value         =   2
logid           =   80
model_path      =   "../Model/"
log_file        =   "../Dataset/DATA-LOG-" + str(logid) + ".csv"
finetune_file   =   "../Dataset/Finetune/executed_sql.csv"
model_name      =   model_path + "Transformer/TRANSFORMER_TEST_" + str(logid) + "_"  + str(k_value) + "_" + str(old_epoch_num) + ".keras"
new_model_name  =   model_path + "Finetuned/TRANSFORMER_" + str(logid) + "_"  + str(k_value) + "_" + str(epoch_num) + "_" + str(new_order) + "_" + str(payment) + ".keras"
history_output_csv   = "../Output/finetuning/Transformer_finetune_" + str(new_order) + "_" + str(payment) + ".csv"
predict_cache_output = "../Output/finetuning/Transformer_Data_" + str(logid) + "_cache.csv"

# 映射表名和表ID
table_id_dict ={"bmsql_district":1, "bmsql_customer":2, "bmsql_item":3, "bmsql_stock":4, "bmsql_warehouse":5}

# 映射字段到标准名称
field_aliases = {
    'w_id': 'WID', 'c_w_id': 'WID', 's_w_id': 'WID', 'd_w_id': 'WID',
    'd_id': 'DID', 'c_d_id': 'DID', 's_d_id': 'DID',
    'c_id': 'CID',
    'i_id': 'IID', 's_i_id': 'IID', 'ol_i_id': 'IID'
}

# 类型ID
type_id_dict = {"SELECT":1, "UPDATE":2, "DELETE":3, "INSERT":4}

# 打乱顺序
def shuffle_transactions(transactions, transaction_cache):
    # 获取所有 VXID 键
    vxid_list = list(transactions.keys())
    random.shuffle(vxid_list)  # 就地打乱顺序

    # 按打乱后的顺序构造新字典
    shuffled_transactions = {vxid: transactions[vxid] for vxid in vxid_list}
    shuffled_cache = {vxid: transaction_cache[vxid] for vxid in vxid_list}

    return shuffled_transactions, shuffled_cache


# 截取比例
def trim_by_ratio(transactions, transaction_cache, ratio=0.8):
    vxid_list = list(transactions.keys())
    total = len(vxid_list)
    keep_n = int(total * ratio)

    random.shuffle(vxid_list)

    # 截取前 keep_n 个事务
    vxid_keep = set(vxid_list[:keep_n])

    # 构造新字典（或就地修改）
    transactions_trimmed = {vxid: transactions[vxid] for vxid in vxid_keep}
    cache_trimmed = {vxid: transaction_cache[vxid] for vxid in vxid_keep}

    return transactions_trimmed, cache_trimmed

# 分析SQL语句参数值
def extract_ids(sql: str) -> dict:
    id_map = {}

    result = {'TYPE_ID': 0, 'WID': 0, 'DID': 0, 'CID': 0, 'IID': 0, 'TABLE': '', 'TABLE_ID': 0}

    # 将 SQL 转小写以统一处理
    sql_lower = sql.lower()

    # 获取类型
    type = sql.upper().split()[0].strip()
    type_id =  type_id_dict[type] if type in type_id_dict else 0
    result['TYPE_ID'] = type_id

    # 提取表名（第一个出现的 FROM / UPDATE / INTO 后的单词）
    table_match = re.search(r'\b(?:from|update|into)\b\s+([a-zA-Z0-9_]+)', sql_lower)
    if table_match:
        table = table_match.group(1)
        result['TABLE'] = table
        result['TABLE_ID'] = table_id_dict.get(table, 0)

    # 匹配 where 子句内容（简单处理）
    where_match = re.search(r'\bwhere\b(.+?)(?:\border\b|\bgroup\b|\blimit\b|$)', sql_lower, re.IGNORECASE)
    if where_match:
        where_clause = where_match.group(1)
        pattern = re.compile(r'(\b\w+\b)\s*=\s*(\d+)')
        for field, value in pattern.findall(where_clause):
            if field in field_aliases:
                std_key = field_aliases[field]
                result[std_key] = int(value)
    return result

cids = []
iids = []

# 获取Transformer训练和测试所需的数据集
def getTransformeratasets_finetune(log_file, finetune_file, modelPath="../Model/", new_order = 0, payment = 0, k_value = 2):
    global v_size
    # WV模型
    model1, model2 = word2Vec.loadTwoModels(modelPath)
    # 事务计数
    new_order_count = new_order-1
    payment_count = payment-1
    # 数据集事务
    transactions, transaction_cache = Transformer.getTransformeratasets(log_file, modelPath, k_value)
    transactions, transaction_cache = trim_by_ratio(transactions, transaction_cache, 0.1)

    # 初始数据集
    with open(finetune_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        results = defaultdict(list)
        for row in reader:
            if len(row) < 3:
                continue  # 跳过非法行
            txn_type = str(row[0])
            if "NEW_ORDER" in txn_type:
                if new_order_count == 0:
                    continue
                else:
                    new_order_count -= 1
            elif "PAYMENT" in txn_type:
                if payment_count == 0:
                    continue
                else:
                    payment_count -= 1
            task_id = row[1]
            sql = row[3].strip('"').strip()  # 去掉双引号
            sql_dataitem = extract_ids(sql)
            results[task_id].append(sql_dataitem)

    # 转为普通 dict（可选）
    dict_result = dict(results)

    for task_id, sql_list in dict_result.items():
        for dataitem in sql_list:
            if task_id not in transactions:
                transactions[task_id] = []
                transaction_cache[task_id] = {}
                transaction_cache[task_id]["X"] = []
                transaction_cache[task_id]["y"] = []
                transaction_cache[task_id]["count"] = 0
            temp_row = []
            if dataitem["TYPE_ID"] != -1:
                transaction_cache[task_id]["count"] += 1
                if transaction_cache[task_id]["count"] <= k_value:
                    tempx = []
                    tempx.append(int(dataitem["TYPE_ID"]))
                    tempx.append(int(dataitem["TABLE_ID"]))
                    tempx.append(int(dataitem["WID"]))
                    tempx.append(int(dataitem["DID"]))
                    tempx.append(int(dataitem["CID"]))
                    tempx.append(int(dataitem["IID"]))
                    transaction_cache[task_id]["X"].append(tuple(tempx))
                if transaction_cache[task_id]["count"] > k_value:
                    tempy = []
                    tempy.append(int(dataitem["TYPE_ID"]))
                    tempy.append(int(dataitem["TABLE_ID"]))
                    tempy.append(int(dataitem["WID"]))
                    tempy.append(int(dataitem["DID"]))
                    tempy.append(int(dataitem["CID"]))
                    tempy.append(int(dataitem["IID"]))
                    transaction_cache[task_id]["y"].append(tuple(tempy))
                temp_row.append(float(dataitem["TYPE_ID"])/4)
                temp_row.append(float(dataitem["TABLE_ID"])/4)
                if dataitem["CID"] == 0:
                    temp_row.extend(np.zeros(v_size))
                else:
                    cids.append(int(dataitem["CID"]))
                    temp_row.extend(model1.wv.get_vector(str(dataitem["CID"])))
                if dataitem["IID"] == 0:
                    temp_row.extend(np.zeros(v_size))
                else:
                    iids.append(int(dataitem["IID"]))
                    temp_row.extend(model2.wv.get_vector(str(dataitem["IID"])))
                transactions[task_id].append(temp_row)
    return shuffle_transactions(transactions, transaction_cache)



if __name__ == '__main__':
    # WV模型
    model1, model2 = word2Vec.loadTwoModels(model_path)
    default_vector = np.zeros(v_size)
    model1.wv['0'] = default_vector
    model2.wv['0'] = default_vector

    # 原有模型
    model = tf.keras.models.load_model(model_name)

    # 假设新数据路径为 new_log_file0
    new_transactions, new_transactions_cache = getTransformeratasets_finetune(log_file, finetune_file, model_path, new_order, payment, k_value)

    print(len(new_transactions))

    with open('../test_wv.txt', mode='a+') as f:
        for cid in cids:
            f.write(str(cid) + " ")

        f.write("\n")

        for iid in iids:
            f.write(str(iid) + " ")

    X_new = []
    y_new = []

    count = {}

    for vxid in new_transactions:
        if len(new_transactions[vxid]) >= maxlen:
            X_new.append([new_transactions[vxid][i] for i in range(k_value)])
            y_new.append([new_transactions[vxid][i] for i in range(k_value, maxlen)])

    X_new = np.array(X_new)
    y_new = np.array(y_new)

    loss, acc = model.evaluate(X_new, y_new, batch_size=b_size)
    print(f"Loss: {loss:.4f}, Accuracy: {acc:.4f}")

    model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])

    history = model.fit(X_new, y_new, batch_size=b_size, epochs=epoch_num, validation_split=0.2)

    model.save(new_model_name)

    # 保存每轮的 loss 和 acc 数据
    df = pd.DataFrame({
        'epoch': list(range(1, epoch_num + 1)),
        'loss': history.history['loss'],
        'accuracy': history.history['accuracy'],
        'val_loss': history.history.get('val_loss', [None] * epoch_num),
        'val_accuracy': history.history.get('val_accuracy', [None] * epoch_num)
    })
    df.to_csv(history_output_csv, index=False)

    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Model Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("../Output/" + str(epoch_num)  + "_" + str(new_order) + "_" + str(payment) + "_finetune_accuracy.png")


