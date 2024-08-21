import os
import sys
import csv
import time
import numpy as np
import pandas as pd
import word2Vec
import dataProcessor as dp
import tensorflow as tf
from datetime import datetime
from sklearn.decomposition import PCA
from collections import defaultdict

# 配置 TensorFlow 的线程池
num_threads = 20
tf.config.threading.set_intra_op_parallelism_threads(num_threads)
tf.config.threading.set_inter_op_parallelism_threads(num_threads)

# tf.config.set_visible_devices([tf.config.get_visible_devices('CPU')[0]])

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # 设置TensorFlow使用指定数量的GPU
tf.config.experimental.set_visible_devices(gpus[0], 'GPU') # 使用第一个GPU


# WV模型相关参数
v_size          =       5
w_size          =       10

# 训练数据文件
logid = 80
logFile = "../Dataset/DATA-LOG-" + str(logid) + ".csv"
# logFile         =       "../Dataset/LOG_TPCC4.csv"

# 是否输出Log
print_log       =       True

# 获取LSTM训练和测试所需的数据集
def getLSTMDatasets(logFile, k_num = 2):
    global v_size
    # 初始数据集
    dataset = dp.read_csv_all_data(logFile)
    # 初始化PCA
    pca = PCA(n_components=1)
    pca_type = PCA(n_components=1)
    # WV模型
    model1, model2 = word2Vec.loadTwoModels("../Model/")
    # 训练和测试的数据集
    X_data, y_customer_data, y_item_data, X_type_data, y_type_data = [], [], [], [], []
    # 数据集事务
    transactions = {}
    for index, row in dataset.iterrows():
        if row["VXID"] not in transactions:
            transactions[row["VXID"]] = []
        temp_row = []
        if row["TYPEID"] != -1:
            temp_row.append(row["TYPEID"])
            temp_row.append(row["TABLEID"])
            if int(row["CID"]) == 0:
                temp_row.extend(np.zeros(v_size))
            else:
                temp_row.extend(model1.wv.get_vector(str(row["CID"])))
            if int(row["IID"]) == 0:
                temp_row.extend(np.zeros(v_size))
            else:
                temp_row.extend(model2.wv.get_vector(str(row["IID"])))
            transactions[row["VXID"]].append(temp_row)
    for vxid in transactions:
        for k in range(0, len(transactions[vxid]) - k_num):
            temp_matrix = []
            temp_type_matrix = []
            for i in range(0, k+k_num):
                temp_matrix.append(transactions[vxid][i])
                temp_type_matrix.append(transactions[vxid][i][0:2])
            temp_matrix = np.array(temp_matrix).T
            temp_type_matrix = np.array(temp_type_matrix).T
            pca.fit(temp_matrix)
            pca_type.fit(temp_type_matrix)
            X_type_data.append(pca_type.transform(temp_type_matrix).flatten().tolist())
            X_data.append(pca.transform(temp_matrix).flatten().tolist())
            y_type_data.append(transactions[vxid][k + k_num][0:2])
            y_customer_data.append(transactions[vxid][k+k_num][2:v_size+2])
            y_item_data.append(transactions[vxid][k+k_num][v_size+2:2*v_size+2])
    return transactions, np.array(X_data), np.array(y_customer_data), np.array(y_item_data), np.array(X_type_data), np.array(y_type_data)

def get_txn_cache(logFile, k_num = 2):
    dataset = dp.read_csv_all_data(logFile)
    transaction_cache ={}
    for index, row in dataset.iterrows():
        if row["VXID"] not in transaction_cache:
            transaction_cache[row["VXID"]] = {}
            transaction_cache[row["VXID"]]["X"] = []
            transaction_cache[row["VXID"]]["y"] = []
            transaction_cache[row["VXID"]]["count"] = 0
        if row["TYPEID"] != -1:
            transaction_cache[row["VXID"]]["count"] += 1
            if transaction_cache[row["VXID"]]["count"] <= k_num:
                tempx = []
                tempx.append(row["TYPEID"])
                tempx.append(row["TABLEID"])
                tempx.append(row["WID"])
                tempx.append(row["DID"])
                tempx.append(row["CID"])
                tempx.append(row["IID"])
                transaction_cache[row["VXID"]]["X"].append(tuple(tempx))
            if transaction_cache[row["VXID"]]["count"] > k_num:
                tempy = []
                tempy.append(row["TYPEID"])
                tempy.append(row["TABLEID"])
                tempy.append(row["WID"])
                tempy.append(row["DID"])
                tempy.append(row["CID"])
                tempy.append(row["IID"])
                transaction_cache[row["VXID"]]["y"].append(tempy)
    return transaction_cache

# 建立LSTM模型
def build_lstm_model(input_dim, output_dim, hidden_units):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Reshape((input_dim, 1)),
        # tf.keras.layers.LSTM(hidden_units, activation='tanh', return_sequences=True),
        tf.keras.layers.LSTM(hidden_units, activation='tanh'),
        tf.keras.layers.Dense(output_dim)
    ])
    learning_rate = 0.0001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,loss='mse', metrics=['accuracy'])
    return model

predict_times = []

# 在指定的K值下训练和测试LSTM模型
def train_with_K(logFile, k_value, input_dim, output_dim, hidden_units, hidden_units2, b_size, epoch_num1, epoch_num2):
    model_name1 = "../Model/LSTM/LSTM_TEST_CUST_" + str(logid) + "_" + str(k_value) + "_" + str(epoch_num1) + ".keras"
    model_name2 = "../Model/LSTM/LSTM_TEST_ITEM_" + str(logid) + "_" + str(k_value) + "_" + str(epoch_num1) + ".keras"
    model_name3 = "../Model/LSTM/LSTM_TEST_TYPE_" + str(logid) + "_" + str(k_value) + "_" + str(epoch_num2) + ".keras"

    # 获取LSTM数据集
    transaction_dict, X_data, y_customer_data, y_item_data, X_type_data, y_type_data = getLSTMDatasets(logFile, k_value)
    if print_log:
        print(X_data.shape)
        print(y_item_data.shape)
        print(y_customer_data.shape)
        print(X_type_data.shape)
        print(y_type_data.shape)

    # 划分LSTM数据集
    split_index = int(len(X_data) * 0.8)
    X_train = X_data[:split_index]
    y_item_train = y_item_data[:split_index]
    y_customer_train = y_customer_data[:split_index]
    X_test = X_data[split_index:]
    y_item_test = y_item_data[split_index:]
    y_customer_test = y_customer_data[split_index:]
    X_type_train = X_type_data[:split_index]
    y_type_train = y_type_data[:split_index]
    X_type_test = X_type_data[split_index:]
    y_type_test = y_type_data[split_index:]

    # 加载WV模型
    model1, model2 = word2Vec.loadTwoModels("../Model/")
    default_vector = np.ones(v_size)
    model1.wv['0'] = default_vector
    model2.wv['0'] = default_vector

    # 训练并保存LSTM模型
    # training_log_file_path = '../Dataset/LSTM.txt'
    # with open(training_log_file_path, 'w+') as f:
    #     original_stdout = sys.stdout
    #     sys.stdout = f    # 将标准输出重定向到文件
    # customer
    model_customer = build_lstm_model(input_dim, output_dim, hidden_units)
    history_customer = model_customer.fit(X_train, y_customer_train, epochs=epoch_num1, batch_size=b_size)
    # item
    model_item = build_lstm_model(input_dim, output_dim, hidden_units)
    history_item = model_item.fit(X_train, y_item_train, epochs=epoch_num1, batch_size=b_size)
    # type
    model_type = build_lstm_model(2, 2, hidden_units2)
    history_type = model_type.fit(X_type_train, y_type_train, epochs=epoch_num2, batch_size=b_size)
    #     sys.stdout = original_stdout  # 恢复标准输出
    # # customer
    model_customer.evaluate(X_test, y_customer_test)
    # tf.keras.models.save_model(model_customer, model_name1)
    # # item
    model_item.evaluate(X_test, y_item_test)
    # tf.keras.models.save_model(model_item, model_name2)
    # # type
    model_type.evaluate(X_type_test, y_type_test)
    # tf.keras.models.save_model(model_type, model_name3)

    # 从文件加载LSTM模型
    # model_customer = tf.keras.models.load_model(model_name1)
    # model_item = tf.keras.models.load_model(model_name2)
    # model_type = tf.keras.models.load_model(model_name3)

    # 输出训练历史
    # customer_history_df = pd.DataFrame(history_customer.history)
    # customer_history_df.to_csv('../Output/logs/customer_training_log.csv', index=False)
    #
    # item_history_df = pd.DataFrame(history_item.history)
    # item_history_df.to_csv('../Output/logs/item_training_log.csv', index=False)
    #
    # type_history_df = pd.DataFrame(history_type.history)
    # type_history_df.to_csv('../Output/logs/type_training_log.csv', index=False)

    # 进行预测测试
    txn_size =  0   # 初始化计数量
    for vxid in transaction_dict:
        txn_size    =   max(txn_size, len(transaction_dict[vxid]))

    total_num = list(np.zeros(txn_size))
    customer_accurate_num = list(np.zeros(txn_size))
    item_accurate_num = list(np.zeros(txn_size))
    type_accurate_num = list(np.zeros(txn_size))
    table_accurate_num = list(np.zeros(txn_size))

    # 取后20%的事务测试
    txn_num = len(transaction_dict)
    txn_split_index = int(txn_num * 0)
    txn_vxids = list(transaction_dict.keys())
    transactions = {key: transaction_dict[key] for key in txn_vxids[txn_split_index:]}

    transaction_cache = get_txn_cache(logFile, k_value)

    # 开始进行预测的测试
    for vxid in transactions:
        pca         =   PCA(n_components=1)
        pca_type    =   PCA(n_components=1)
        current_dataset         =   []
        current_dataset_type    =   []
        i = 0
        # 先加载前K个事务操作作为初始的输入数据
        while i < k_value:
            current_dataset.append(transactions[vxid][i])
            current_dataset_type.append(transactions[vxid][i][0:2])
            i += 1
        loop = 0
        latest_type = current_dataset_type[len(current_dataset_type)-1][0]
        latest_table = current_dataset_type[len(current_dataset_type)-1][1]

        # 循环预测，每次预测结果加入下一轮的输入数据，直到预测到Commit或达到最大轮数
        while (i + loop) < len(transactions[vxid]): # latest_type != 0 and latest_table != 0
            input = np.array(current_dataset).T
            input_type = np.array(current_dataset_type).T
            pca.fit(input)
            pca_type.fit(input_type)
            X = [pca.transform(input).flatten().tolist()]
            X_type = [pca_type.transform(input_type).flatten().tolist()]

            start_time = time.time()

            predict_customer = model_customer.predict(np.array(X))[0]

            predict_item = model_item.predict(np.array(X))[0]
            predict_type = model_type.predict(np.array(X_type))[0]

            end_time = time.time()
            predict_time_seconds = (end_time - start_time)
            predict_times.append(predict_time_seconds)

            latest_type    =   int(predict_type[0])
            latest_table   =   int(predict_type[1])

            if np.linalg.norm(np.array(predict_customer)) < 1e-2:
                predicted_customer = '0'
            else:
                predicted_customer = model1.wv.similar_by_vector(np.array(predict_customer), topn=1)[0][0]
            if np.linalg.norm(np.array(predict_item)) < 1e-2:
                predicted_item = '0'
            else:
                predicted_item = model2.wv.similar_by_vector(np.array(predict_item), topn=1)[0][0]

            real_type = transactions[vxid][i+loop][0]
            real_table = transactions[vxid][i+loop][1]

            if np.linalg.norm(np.array(transactions[vxid][i+loop][2:7])) < 1e-2:
                real_customer = '0'
            else:
                real_customer = model1.wv.similar_by_vector(np.array(transactions[vxid][i+loop][2:7]), topn=1)[0][0]
            if np.linalg.norm(np.array(transactions[vxid][i+loop][7:12])) < 1e-2:
                real_item = '0'
            else:
                real_item = model2.wv.similar_by_vector(np.array(transactions[vxid][i+loop][7:12]), topn=1)[0][0]

            transaction_cache[vxid]["y"][loop][0]   =   latest_type
            transaction_cache[vxid]["y"][loop][1]   =   latest_table
            transaction_cache[vxid]["y"][loop][4]   =   predicted_customer
            transaction_cache[vxid]["y"][loop][5]   =   predicted_item

            # 对于值为0的客户或商品ID，设置全一向量作为默认值
            if predicted_customer == '0':
                closest_customer_vector = np.zeros(v_size).tolist()
            else:
                closest_customer_vector = model1.wv.get_vector(predicted_customer)
            if predicted_item == '0':
                closest_item_vector = np.zeros(v_size).tolist()
            else:
                closest_item_vector = model2.wv.get_vector(predicted_item)

            # 获取下一轮预测需要添加的输入值（即本轮的完整预测结果）
            new_input = []
            new_input.append(latest_type)
            new_input.append(latest_table)
            new_input.extend(closest_customer_vector)
            new_input.extend(closest_item_vector)
            current_dataset.append(new_input)
            current_dataset_type.append(predict_type)
            loop += 1

            # 准确数计算
            total_num[0] += 1
            print(loop - 1)
            print(i + loop - 1)
            total_num[i + loop - 1] += 1
            if real_type == latest_type:
                type_accurate_num[0] += 1
                type_accurate_num[i + loop - 1] += 1
                if print_log:
                    print("predicted type: " + str(latest_type) + " (正确)")
            elif print_log:
                print("predicted type: " + str(latest_type) + " (错误，应为" + str(real_type) + ")")
            if real_table == latest_table:
                table_accurate_num[0] += 1
                table_accurate_num[i + loop - 1] += 1
                if print_log:
                    print("predicted table: " + str(latest_table) + " (正确)")
            elif print_log:
                print("predicted table: " + str(latest_table) + " (错误，应为" + str(real_table) + ")")
            if real_customer == predicted_customer:
                customer_accurate_num[0] += 1
                customer_accurate_num[i + loop - 1] += 1
                if print_log:
                    print("predicted customer: " + str(predicted_customer) + " (正确)")
            elif print_log:
                print("predicted customer: " + str(predicted_customer) + " (错误，应为" + str(real_customer) + ")")
            if real_item == predicted_item:
                item_accurate_num[0] += 1
                item_accurate_num[i + loop - 1] += 1
                if print_log:
                    print("predicted item: " + str(predicted_item) + " (正确)")
            elif print_log:
                print("predicted item: " + str(predicted_item) + " (错误，应为" + str(real_item) + ")")

            # 预测效果记录
        with open("../Output/Text/" + str(logid) + "/lstm_test_output.txt", 'a+', encoding='utf-8') as file:
            file.write("----------------- K = " + str(k_value) + " -----------------\n")
            file.write(str(datetime.now()) + "\n")
            for i in range(0, len(total_num)):
                file.write("[i=" + str(i) + "]" + " Total Num: " + str(int(total_num[i])) + "\n")
                if total_num[i] > 0:
                    file.write("    Accurate Type Num: " + str(int(type_accurate_num[i])) + "  " + str(
                        type_accurate_num[i] / total_num[i]) + "\n")
                    file.write("    Accurate Table Num: " + str(int(table_accurate_num[i])) + "  " + str(
                        table_accurate_num[i] / total_num[i]) + "\n")
                    file.write("    Accurate Customer Num: " + str(int(customer_accurate_num[i])) + "  " + str(
                        customer_accurate_num[i] / total_num[i]) + "\n")
                    file.write("    Accurate Item Num: " + str(int(item_accurate_num[i])) + "  " + str(
                        item_accurate_num[i] / total_num[i]) + "\n")

    # df = pd.DataFrame(predict_times)
    # df.to_csv("../Output/predict/LSTM_preTime.csv", index=False, header=None)

    for key in transaction_cache:
        for j in range(0, len(transaction_cache[key]["X"])):
            transaction_cache[key]["X"][j] = [int(item) for item in transaction_cache[key]["X"][j]]
            transaction_cache[key]["X"][j] = tuple(transaction_cache[key]["X"][j])
        for i in range(0, len(transaction_cache[key]["y"])):
            transaction_cache[key]["y"][i] = [int(item) for item in transaction_cache[key]["y"][i]]
            transaction_cache[key]["y"][i] = tuple(transaction_cache[key]["y"][i])

    X_cache = []
    y_cache = []
    count_cache = defaultdict(int)  # 计数字典，记录每个 input-output 组合出现的次数
    # 提取 transaction_cache 中的数据
    for key in transaction_cache:
        X_cache.append(transaction_cache[key]["X"])
        y_cache.append(transaction_cache[key]["y"])

    # 统计每个 input-output 组合出现的次数
    for X_group, y_group in zip(X_cache, y_cache):
        input_str = ' '.join(map(str, X_group))
        output_str = ' '.join(map(str, y_group))
        combination = (input_str, output_str)
        count_cache[combination] += 1

    output_filename = "../Output/Text/LSTM_Data_" + str(logid) + "_cache.csv"

    # 写入 CSV 文件
    # if not os.path.exists(output_filename):
    #     with open(output_filename, 'w+', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(['input', 'output', 'count'])
    #         # 遍历每个组合并写入
    #         for (input_str, output_str), count in count_cache.items():
    #             writer.writerow([input_str, output_str, count])
    # else:
    #     with open(output_filename, 'a+', newline='') as csvfile:
    #         writer = csv.writer(csvfile)
    #         # 遍历每个组合并写入
    #         for (input_str, output_str), count in count_cache.items():
    #             writer.writerow([input_str, output_str, count])
    #
    # print(f"CSV 文件 '{output_filename}' 已写入")

            # if print_log:
            #     print("--------------- K = " + str(k_value) + " ---------------")
            #     print("共有" + str(len(transactions[vxid])) + "条SQL语句")
            #     print("预测第" + str(i + loop + 1) + "条SQL语句")
            #     print("输入值", str(transactions[vxid][i + loop-1]))
            #     print("真值", str(transactions[vxid][i + loop]))



if __name__=="__main__":
    # 模型参数
    input_dim       =   2 + 2 * v_size              # 决定了customer/item模型的输入维度
    output_dim      =   v_size                      # 决定了customer/item模型的输出维度
    hidden_units    =   32                          # customer/item - LSTM层的单元数
    hidden_units2    =   32                         # type/table - LSTM层的单元数
    epoch_num1      =   1000                        # customer/item - 训练轮数
    epoch_num2      =   1000                        # type/table - 训练轮数
    batch_size      =   64

    for k in range(2,3):
        train_with_K(logFile, k, input_dim, output_dim, hidden_units, hidden_units2, batch_size, epoch_num1, epoch_num2)

