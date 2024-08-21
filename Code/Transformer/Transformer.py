import os
import csv
import sys
import time
import numpy as np
import pandas as pd
import word2Vec
import dataProcessor as dp
import tensorflow as tf
from datetime import datetime
from keras.layers import Embedding, MultiHeadAttention, Dense, LayerNormalization, Dropout
from keras.models import Model
from collections import defaultdict

# tf.config.set_visible_devices([tf.config.get_visible_devices('CPU')[0]])

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
tf.config.set_visible_devices(gpus[0], 'GPU')

# 配置 TensorFlow 的线程池
num_threads = 20
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

tf.config.optimizer.set_jit(True)

# 模型参数设置
v_size  =   5
w_size  =   20

in_dim      = 12
maxlen      = 13
embed_dim   = 16
num_heads   = 4
ff_dim      = 16
num_transformer_blocks = 4

b_size      = 64
epoch_num   = 5000

# 其他参数
logid       =   80
print_log   =   False
model_path  =   "../Model/"
log_file    =   "../Dataset/DATA-LOG-" + str(logid) + ".csv"

# 获取Transformer训练和测试所需的数据集
def getTransformeratasets(logFile, modelPath="./Model/", k_value = 2):
    global v_size
    # 初始数据集
    dataset = dp.read_csv_all_data(logFile)
    # WV模型
    model1, model2 = word2Vec.loadTwoModels(modelPath)
    # 数据集事务
    transactions = {}
    transaction_cache = {}
    for index, row in dataset.iterrows():
        if row["VXID"] not in transactions:
            transactions[row["VXID"]] = []
            transaction_cache[row["VXID"]] = {}
            transaction_cache[row["VXID"]]["X"] = []
            transaction_cache[row["VXID"]]["y"] = []
            transaction_cache[row["VXID"]]["count"] = 0
        temp_row = []
        if row["TYPEID"] != -1:
            transaction_cache[row["VXID"]]["count"] += 1
            if transaction_cache[row["VXID"]]["count"] <= k_value:
                tempx = []
                tempx.append(row["TYPEID"])
                tempx.append(row["TABLEID"])
                tempx.append(row["WID"])
                tempx.append(row["DID"])
                tempx.append(row["CID"])
                tempx.append(row["IID"])
                transaction_cache[row["VXID"]]["X"].append(tuple(tempx))
            if transaction_cache[row["VXID"]]["count"] > k_value:
                tempy = []
                tempy.append(row["TYPEID"])
                tempy.append(row["TABLEID"])
                tempy.append(row["WID"])
                tempy.append(row["DID"])
                tempy.append(row["CID"])
                tempy.append(row["IID"])
                transaction_cache[row["VXID"]]["y"].append(tempy)
            temp_row.append(float(row["TYPEID"])/4)
            temp_row.append(float(row["TABLEID"])/4)
            if row["CID"] == 0:
                temp_row.extend(np.zeros(v_size))
            else:
                temp_row.extend(model1.wv.get_vector(str(row["CID"])))
            if row["IID"] == 0:
                temp_row.extend(np.zeros(v_size))
            else:
                temp_row.extend(model2.wv.get_vector(str(row["IID"])))
            transactions[row["VXID"]].append(temp_row)
    return transactions, transaction_cache

# 定义Transformer块
@tf.keras.utils.register_keras_serializable("TransformerBlock")
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.001, **kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="tanh"),
            Dense(embed_dim)
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "rate": self.rate,
        })
        return config

# 定义Token和位置嵌入层
@tf.keras.utils.register_keras_serializable("TokenAndPositionEmbedding")
class TokenAndPositionEmbedding(tf.keras.layers.Layer):
    def __init__(self, maxlen, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        self.token_emb = Dense(embed_dim)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-2]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config()
        config.update({
            "maxlen": self.maxlen,
            "embed_dim": self.embed_dim,
        })
        return config

# 创建模型
def create_model(in_dim, maxlen, embed_dim, num_heads, ff_dim, K, num_transformer_blocks):
    inputs = tf.keras.Input(shape=(K, in_dim))
    embedding_layer = TokenAndPositionEmbedding(maxlen, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    for _ in range(num_transformer_blocks):
        transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
        x = transformer_block(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(64, activation="tanh")(x)
    x = tf.keras.layers.Dense(in_dim * (maxlen - K), activation="linear")(x)
    outputs = tf.keras.layers.Reshape((maxlen - K, in_dim))(x)
    model = Model(inputs=inputs, outputs=outputs)
    return model

predict_times = []

if __name__ == '__main__':
    # WV模型
    model1, model2 = word2Vec.loadTwoModels(model_path)
    default_vector = np.zeros(v_size)
    model1.wv['0'] = default_vector
    model2.wv['0'] = default_vector

    # 划分训练测试集
    for k_value in range(1,6):
        transactions, transaction_cache = getTransformeratasets(log_file, model_path, k_value)
        if print_log:
            print("------------------- K = " + str(k_value) + " -------------------")
        X_data = []
        y_data = []

        # for vxid in transactions:
        txn_size    =   0
        for vxid in transactions:
            if len(transactions[vxid]) >= maxlen:
                txn_size    =   max(txn_size, len(transactions[vxid]))
                temp_X = []
                temp_y = []
                for i in range(0, k_value):
                    temp_X.append(transactions[vxid][i])
                for i in range(k_value, maxlen):
                    temp_y.append(transactions[vxid][i])
                X_data.append(temp_X)
                y_data.append(temp_y)

        split_index = int(len(X_data) * 0.8)

        X_train_data = np.array(X_data[:split_index])
        y_train_data = np.array(y_data[:split_index])

        # 模型保存位置
        model_name = "../Model/Transformer/TRANSFORMER_TEST_" + str(logid) + "_"  + str(k_value) + "_" + str(epoch_num) + ".keras"

        # 创建、编译、训练、保存模型
        training_log_file_path = "../Dataset/Transformer.txt"

        # with open(training_log_file_path, 'w+') as f:
        #     original_stdout = sys.stdout
        #     sys.stdout = f  # 将标准输出重定向到文件
        #     model = create_model(in_dim, maxlen, embed_dim, num_heads, ff_dim, k_value, num_transformer_blocks)
        #     model.compile(optimizer="adam", loss="mse", metrics=["accuracy"])
        #     model.fit(X_train_data, y_train_data, batch_size=b_size, epochs=epoch_num)
        #     tf.keras.models.save_model(model, model_name)
        #     sys.stdout = original_stdout    # 恢复标准输出

        # 加载模型
        model = tf.keras.models.load_model(model_name)

        # 预测
        total_num = list(np.zeros(txn_size))
        customer_accurate_num = list(np.zeros(txn_size))
        item_accurate_num = list(np.zeros(txn_size))
        type_accurate_num = list(np.zeros(txn_size))
        table_accurate_num = list(np.zeros(txn_size))

        cache_keys = list(transaction_cache.keys())

        for m in range(0, len(X_data)):
            start_time = time.time()
            pred = model.predict(np.array(X_data[m:m+1]))  # 对测试数据进行预测
            end_time = time.time()
            predict_time_seconds = (end_time-start_time)
            predict_times.append(predict_time_seconds)

            for i in range(0, len(pred[0])):
                predict_type = round(pred[0][i][0] * 4)
                predict_table = round(pred[0][i][1] * 4)
                customer_v = np.array(pred[0][i][2:7])
                item_v = np.array(pred[0][i][7:12])
                if np.linalg.norm(customer_v) < 1e-2:
                    predict_customer = '0'
                else:
                    predict_customer = model1.wv.similar_by_vector(customer_v, topn = 1)[0][0]
                if np.linalg.norm(item_v) < 1e-2:
                    predict_item = '0'
                else:
                    predict_item = model2.wv.similar_by_vector(item_v, topn = 1)[0][0]

                real_type = round(y_data[m][i][0] * 4)
                real_table = round(y_data[m][i][1] * 4)
                customer_v = np.array(y_data[m][i][2:7])
                item_v = np.array(y_data[m][i][7:12])
                if np.linalg.norm(customer_v) < 1e-2:
                    real_customer = '0'
                else:
                    real_customer = model1.wv.similar_by_vector(customer_v, topn=1)[0][0]
                if np.linalg.norm(item_v) < 1e-2:
                    real_item = '0'
                else:
                    real_item = model2.wv.similar_by_vector(item_v, topn=1)[0][0]

                # transaction_cache[cache_keys[m]]["y"][i][0]  =    predict_type
                # transaction_cache[cache_keys[m]]["y"][i][1]  =    predict_table
                # transaction_cache[cache_keys[m]]["y"][i][4]  =    predict_customer
                # transaction_cache[cache_keys[m]]["y"][i][5]  =    predict_item

                # 准确数计算
                total_num[0] += 1
                total_num[i + k_value] += 1
                if real_type == predict_type:
                    type_accurate_num[0] += 1
                    type_accurate_num[i + k_value] += 1
                if real_table == predict_table:
                    table_accurate_num[0] += 1
                    table_accurate_num[i + k_value] += 1
                if real_customer == predict_customer:
                    customer_accurate_num[0] += 1
                    customer_accurate_num[i + k_value] += 1
                if real_item == predict_item:
                    item_accurate_num[0] += 1
                    item_accurate_num[i + k_value] += 1

                if print_log:
                    print("\t" + str(i) + ": " + str(
                        (predict_type, predict_table, predict_customer, predict_item)) + "\t  " + str(
                        np.linalg.norm(customer_v)) + "\t  " + str(np.linalg.norm(item_v)) + " \t---\t " + str(
                        str(i) + ": " + str((real_type, real_table, real_customer, real_item))))


        # 预测效果记录
        with open("../Output/Text/" + str(logid) + "/transformer_test_output.txt", 'a+',encoding='utf-8') as file:
            file.write("----------------- K = " + str(k_value) + " -----------------\n")
            file.write(str(datetime.now()) + "\n")
            for i in range(0, len(total_num)):
                file.write("[i=" + str(i) + "]" + " Total Num: " + str(total_num[i]) + "\n")
                if total_num[i] > 0:
                    file.write("    Accurate Type Num: " + str(type_accurate_num[i]) + "  " + str(
                        type_accurate_num[i] / total_num[i]) + "\n")
                    file.write("    Accurate Table Num: " + str(table_accurate_num[i]) + "  " + str(
                        table_accurate_num[i] / total_num[i]) + "\n")
                    file.write("    Accurate Customer Num: " + str(customer_accurate_num[i]) + "  " + str(
                        customer_accurate_num[i] / total_num[i]) + "\n")
                    file.write("    Accurate Item Num: " + str(item_accurate_num[i]) + "  " + str(
                        item_accurate_num[i] / total_num[i]) + "\n")

        df = pd.DataFrame(predict_times)
        df.to_csv("../Output/predict/Transformer_preTime.csv", index=False, header=None)

        # for key in transaction_cache:
        #     for j in range(0, len(transaction_cache[key]["X"])):
        #         transaction_cache[key]["X"][j] = [int(item) for item in transaction_cache[key]["X"][j]]
        #         transaction_cache[key]["X"][j] = tuple(transaction_cache[key]["X"][j])
        #     for i in range(0, len(transaction_cache[key]["y"])):
        #         transaction_cache[key]["y"][i] = [int(item) for item in transaction_cache[key]["y"][i]]
        #         transaction_cache[key]["y"][i] = tuple(transaction_cache[key]["y"][i])
        #
        # X_cache =  []
        # y_cache =  []
        #
        # count_cache = defaultdict(int)  # 计数字典，记录每个 input-output 组合出现的次数
        #
        # for key in transaction_cache:
        #     X_cache.append(transaction_cache[key]["X"])
        #     y_cache.append(transaction_cache[key]["y"])
        # output_filename    =   "../Output/Text/Transformer_Data_" + str(logid) + "_cache.csv"
        # if not os.path.exists(output_filename):
        #     with open(output_filename, 'w+', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         writer.writerow(['input', 'output'])  # 写入表头
        #         # 遍历每组数据
        #         for X_group, y_group in zip(X_cache, y_cache):
        #             # 将输入数据和输出数据转换为字符串，写入CSV文件
        #             input_str = ' '.join(map(str, X_group))
        #             output_str = ' '.join(map(str, y_group))
        #             writer.writerow([input_str, output_str])
        # else:
        #     with open(output_filename, 'a+', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         # 遍历每组数据
        #         for X_group, y_group in zip(X_cache, y_cache):
        #             # 将输入数据和输出数据转换为字符串，写入CSV文件
        #             input_str = ' '.join(map(str, X_group))
        #             output_str = ' '.join(map(str, y_group))
        #             writer.writerow([input_str, output_str])
        # print(f"CSV 文件 '{output_filename}' 已写入")

