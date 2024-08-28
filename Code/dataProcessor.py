import re
import os
import csv
import numpy as np
import pandas as pd

# pre-process dataset and generate usable training and testing data for models

pd.set_option('display.max_columns', 10000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 10000)

bool_print = False           # whether to print logs

logid = 80  # which file to process

# data item related IDs and records
data_total = 1
data_dict = {}
table_total = 1
table_id = {}

# write some training or testing data into a csv file
def write_data_to_csv(X_data, y_data, filename='./Output/Text/Data_output.csv'):
    if not os.path.exists(filename):
        with open(filename, 'w+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['input', 'output'])
            for X_group, y_group in zip(X_data, y_data):
                input_str = ' '.join(map(str, X_group))
                output_str = ' '.join(map(str, y_group))
                writer.writerow([input_str, output_str])
    else:
        with open(filename, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for X_group, y_group in zip(X_data, y_data):
                input_str = ' '.join(map(str, X_group))
                output_str = ' '.join(map(str, y_group))
                writer.writerow([input_str, output_str])

# the most important function
def read_csv_all_data(logFile):
    global data_total
    # get all columns
    colNames = ['TIME', 'UNAME', 'DBNAME', 'PID', 'HOSTPORT', 'SID', 'SLNUM', 'TAG', 'TIME2', 'VXID', 'XID', 'LTYPE', 'STATE', 'LOG', 'DETAIL', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'AMAME', 'BACKEND']
    dataset = pd.read_csv(logFile, header=None, names=colNames, dtype='str', encoding="ANSI")

    # remove invalid rows and columns
    dataset.drop(dataset[dataset['VXID'] == ''].index, inplace=True)
    dataset.dropna(subset="VXID" ,axis=0, inplace=True)
    dataset.drop(['UNAME', 'DBNAME', 'PID', 'HOSTPORT', 'SID', 'SLNUM', 'TIME2', 'XID', 'STATE', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',  'AMAME', 'BACKEND'], axis=1, inplace=True)

    # new column to be added
    sql_Type = []
    sql_statement = []
    sql_data = []
    sql_Table = []
    sql_TypeId = []
    sql_TableId = []

    sql_KeyValues = {}
    sql_KeyValues["w_id"] = []
    sql_KeyValues["d_id"] = []
    sql_KeyValues["c_id"] = []
    sql_KeyValues["i_id"] = []
    sql_KeyValues["other"] = []

    table_total_num = 0
    table_id_dict ={}

    aborted_txns = []
    for index, row in dataset.iterrows():
        if row["LTYPE"] == "ERROR" or row["TAG"] == "ROLLBACK":
            aborted_txns.append(row["VXID"])

    # get new column data
    for index, row in dataset.iterrows():
        query_type = ""
        table = ""
        query_condition = ""
        data_index = -1
        statement = ""

        type_id     =   0
        table_id    =   0
        w_id = 0
        d_id = 0
        c_id = 0
        i_id = 0
        other = ""

        # remove invalid rows
        if str(row["VXID"]) == ""  or (str(row["VXID"]).startswith("0")) or (str(row["VXID"]).endswith("/0")):
            dataset.drop(index, axis=0, inplace=True)
            continue

        # errors regarded as rollbacks(aborts)
        if row["VXID"] in aborted_txns:
            # dataset.drop(index, axis=0, inplace=True)
            query_type = "ROLLBACK"
            sql_statement.append(query_condition)
            sql_Type.append(query_type)
            sql_Table.append(table)
            sql_TypeId.append(type_id)
            sql_TableId.append(table_id)
            sql_data.append(data_index)
            sql_KeyValues["w_id"].append(w_id)
            sql_KeyValues["d_id"].append(d_id)
            sql_KeyValues["c_id"].append(c_id)
            sql_KeyValues["i_id"].append(i_id)
            sql_KeyValues["other"].append(other)
            continue

        if str(row["LOG"]).find(":") != -1:
            statement = str(row["LOG"])[(str(row["LOG"]).index(":")+2):].strip('/"').rstrip(';').rstrip()
        else:
            dataset.drop(index, axis=0, inplace=True)
            continue

        statement = re.sub(r'\s+', ' ', statement)

        # SELECT
        if statement.upper().startswith("SELECT"):
            query_type = "SELECT"
            if(statement.upper().find("FROM") != -1):
                if (statement.upper().find("WHERE") != -1):
                    table = statement[statement.upper().index("FROM")+4: statement.upper().find("WHERE")].strip()
            if statement.find("WHERE") != -1:
                query_condition = statement[statement.index("WHERE") + 6:].strip()
                if query_condition.find("FOR UPDATE") != -1:
                    query_condition = query_condition[:  query_condition.index("FOR UPDATE")].strip()

            else:
                query_condition = "ALL"

        # UPDATE
        elif statement.upper().startswith("UPDATE"):
            query_type = "UPDATE"
            table = re.search(r'update (\w+) set', statement, re.I).group(1).strip()
            if statement.find("WHERE") != -1:
                query_condition = statement[statement.index("WHERE") + 6:].strip()
            else:
                query_condition = "ALL"

        # DELETE
        elif statement.upper().startswith("DELETE"):
            query_type = "DELETE"
            table = re.search(r'delete from (\w+) where', statement, re.I).group(1).strip()
            if statement.find("WHERE") != -1:
                query_condition = statement[statement.index("WHERE") + 6:].lstrip()
            else:
                query_condition = "ALL"

        # INSERT
        elif statement.upper().startswith("INSERT"):
            query_type = "INSERT"
            insert = re.sub('\"', '', re.search(r'INTO (.*?) VALUES', statement, re.I).group(1)).strip()
            if insert.find("(") == -1:
                table = insert.split()[0].rstrip()
                insert_key = "ALL"
            else:
                table = insert.split("(")[0].strip()
                insert_key = insert.split("(")[1].split(")")[0].strip()
                insert_value = re.search(r"VALUES\((.*?)\)", statement).group(1).strip()
            query_condition = insert_key + "=" + insert_value

        # BEGIN
        elif (statement.upper().startswith("BEGIN")) or (statement.upper().startswith("START")):
            query_type = "BEGIN"
        # COMMIT
        elif (statement.upper().startswith("END")) or (statement.upper().startswith("COMMIT")):
            query_type = "COMMIT"
        # ROLLBACK
        elif (statement.upper().startswith("ROLLBACK")) or (statement.upper().startswith("ABORT")):
            query_type = "ROLLBACK"
        # OTHERS
        else:
            query_type = "OTHERS"

        if (row["DETAIL"] is not np.nan) and (row["DETAIL"] != "")  and (row["DETAIL"].split() != ""):
            temp_parm = row["DETAIL"].strip("\"")[row["DETAIL"].index(":", 7) + 2:].strip().strip(';').strip()
            temp_parm_dict = {}
            for pair in temp_parm.split(","):
                key, value = pair.strip().split("=")
                temp_parm_dict[key.strip()] = value.strip()

            for key, value in temp_parm_dict.items():
                query_condition = query_condition.replace(key, value)
        else:
            temp_parm = ""

        # get data item ID
        if (query_condition not in data_dict) and query_condition != "":
            data_dict[query_condition] = data_total
            data_index = data_total
            data_total = data_total + 1
        elif query_condition in data_dict:
            data_index = data_dict[query_condition]
        elif query_condition == "ALL":
            data_index = 0      # 0 for all the data
        else:
            data_index = -1     # -1 for no data

        # get table ID
        if table == "":
            table_id = 0
        elif table in table_id_dict:
            table_id = table_id_dict[table]
        else:
            table_total_num += 1
            table_id = table_total_num
            table_id_dict[table] = table_id

        # get type ID
        if query_type == "SELECT":
            type_id = 1
        elif query_type == "UPDATE":
            type_id = 2
        elif query_type == "DELETE":
            type_id = 3
        elif query_type == "INSERT":
            type_id = 4
        elif query_type == "BEGIN":
            type_id = -1
        else:
            type_id = 0

        keyValueStrs = query_condition.split("AND")
        for keyValueStr in keyValueStrs:
            if(keyValueStr.find("=") != -1):
                keyValue = keyValueStr.split("=")
                if(len(keyValue) != 2):
                    print("[ERROR] key-value: " + str(keyValue))
                else:
                    key = keyValue[0]
                    value = keyValue[1].strip().strip("\'")
                    if "w_id" in key:
                        w_id = value
                    elif "d_id" in key:
                        d_id = value
                    elif "c_id" in key:
                        c_id = value
                    elif "i_id" in key:
                        i_id = value
                    else:
                        if other == "":
                            other = query_condition
                        else:
                            other = other + " AND " + query_condition

        sql_statement.append(query_condition)
        sql_data.append(data_index)
        sql_Type.append(query_type)
        sql_TypeId.append(type_id)
        sql_Table.append(table)
        sql_TableId.append(table_id)
        sql_KeyValues["w_id"].append(w_id)
        sql_KeyValues["d_id"].append(d_id)
        sql_KeyValues["c_id"].append(c_id)
        sql_KeyValues["i_id"].append(i_id)
        sql_KeyValues["other"].append(other)

    dataset["TYPE"] = sql_Type
    dataset["TABLE"] = sql_Table
    dataset["STATEMENT"] = sql_statement
    dataset["TYPEID"] = sql_TypeId
    dataset["TABLEID"] = sql_TableId
    dataset["WID"] = sql_KeyValues["w_id"]
    dataset["DID"] = sql_KeyValues["d_id"]
    dataset["CID"] = sql_KeyValues["c_id"]
    dataset["IID"] = sql_KeyValues["i_id"]
    dataset["DATA"] = sql_data
    dataset["OTHER"] = sql_KeyValues["other"]
    dataset.drop(['LOG', 'DETAIL', 'LTYPE', 'TAG'], axis=1, inplace=True)
    dataset.drop(dataset[dataset['TYPE'] == 'OTHERS'].index, inplace=True)
    dataset.drop(dataset[dataset['TYPE'] == 'INSERT'].index, inplace=True)

    dataset.reset_index(drop=True, inplace=True)

    return dataset

def getWorkingsets(dataset):
    transactions = {}
    temp_data = []
    for index, row in dataset.iterrows():
        if row["VXID"] not in transactions:
            transactions[row["VXID"]] = []
        if row["TYPEID"] != -1:
            temp_data.append(int(row["TYPEID"]))
            temp_data.append(int(row["TABLEID"]))
            temp_data.append(int(row["WID"]))
            temp_data.append(int(row["DID"]))
            temp_data.append(int(row["CID"]))
            temp_data.append(int(row["IID"]))
            transactions[row["VXID"]].append(temp_data)
            temp_data = []
    return transactions

def get_sentences(logFile, onlySQL=False): # onlySQL option removes statements such as BEGIN, COMMIT and so on
    # get columns in the logFile
    colNames = ['TIME', 'UNAME', 'DBNAME', 'PID', 'HOSTPORT', 'SID', 'SLNUM', 'TAG', 'TIME2', 'VXID', 'XID', 'LTYPE',
                'STATE', 'LOG', 'DETAIL', 'P', 'Q', 'R',
                'S', 'T', 'U', 'V', 'AMAME', 'BACKEND']
    dataset = pd.read_csv(logFile, header=None, names=colNames, dtype='str', encoding="ANSI")

    # remove invalid rows and columns
    dataset.drop(dataset[dataset['VXID'] == ''].index, inplace=True)
    dataset.dropna(subset="VXID", axis=0, inplace=True)
    dataset.drop(
        ['UNAME', 'DBNAME', 'PID', 'HOSTPORT', 'SID', 'SLNUM', 'TAG', 'TIME2', 'XID', 'STATE', 'P', 'Q', 'R', 'S', 'T',
         'U', 'V', 'AMAME', 'BACKEND'], axis=1, inplace=True)

    # new column data to be added
    sql_statement = []
    sql_types = []

    # get new column data
    for index, row in dataset.iterrows():
        query_type  = ""
        statement   = ""    # SQL statement
        data_index  = -1    # the ID for data to be accessed

        # delete invalid rows
        if str(row["VXID"]) == "" or (str(row["VXID"]).startswith("0")) or (str(row["VXID"]).endswith("/0")):
            dataset.drop(index, axis=0, inplace=True)
            continue

        # errors regarded as rollbacks(aborts)
        if row["LTYPE"] == "ERROR":
            query_type = "ROLLBACK"
            sql_statement.append(statement)
            continue

        # get SQL statements
        if str(row["LOG"]).find(":") != -1:
            statement = str(row["LOG"])[(str(row["LOG"]).index(":") + 2):].strip('/"').rstrip(';').rstrip()
        else:
            dataset.drop(index, axis=0, inplace=True)
            continue

        # get other data
        statement = re.sub(r'\s+', ' ', statement)
        # SELECT
        if statement.upper().startswith("SELECT"):
            query_type = "SELECT"
            if statement.find("FOR UPDATE") != -1:
                statement = statement[:  statement.index("FOR UPDATE")].strip()

        # UPDATE
        elif statement.upper().startswith("UPDATE"):
            query_type = "UPDATE"

        # DELETE
        elif statement.upper().startswith("DELETE"):
            query_type = "DELETE"

        # INSERT
        elif statement.upper().startswith("INSERT"):
            query_type = "INSERT"

        # BEGIN
        elif (statement.upper().startswith("BEGIN")) or (statement.upper().startswith("START")):
            query_type = "BEGIN"

        # COMMIT
        elif (statement.upper().startswith("END")) or (statement.upper().startswith("COMMIT")):
            query_type = "COMMIT"

        # ROLLBACK
        elif (statement.upper().startswith("ROLLBACK")) or (statement.upper().startswith("ABORT")):
            query_type = "ROLLBACK"

        # OTHERS
        else:
            query_type = "OTHERS"
            if (onlySQL):
                dataset.drop(index, axis=0, inplace=True)
                continue

        if (row["DETAIL"] is not np.nan) and (row["DETAIL"] != "") and (row["DETAIL"].split() != ""):
            temp_parm = row["DETAIL"].strip("\"")[row["DETAIL"].index(":", 7) + 2:].strip().strip(
                ';').strip()
            temp_parm_dict = {}
            for pair in temp_parm.split(","):
                key, value = pair.strip().split("=")
                temp_parm_dict[key.strip()] = value.strip()

            for key, value in temp_parm_dict.items():
                statement = statement.replace(key, value)
        else:
            temp_parm = ""

        sql_types.append(query_type)
        sql_statement.append(statement)

    # new rows added
    dataset["TYPE"] = sql_types
    dataset["SENTENCE"] = sql_statement
    # dataset.sort_values(by=["TIME", "TYPE"], ascending=[True, False], inplace=True)

    if(onlySQL):
        dataset.drop(dataset[dataset['TYPE'] == 'BEGIN'].index, inplace=True)
        dataset.drop(dataset[dataset['TYPE'] == 'COMMIT'].index, inplace=True)
        dataset.drop(dataset[dataset['TYPE'] == 'ROLLBACK'].index, inplace=True)
    dataset.drop(['DETAIL', 'LOG', 'LTYPE', 'TIME', 'TYPE'], axis=1, inplace=True)

    new_LogFile = logFile.rstrip(".csv") + "_sentences.csv"
    dataset.to_csv(new_LogFile)

    return dataset

def get_transaction_info(logFile):
    dataset = read_csv_all_data(logFile)
    transaction_info = {}
    for index, row in dataset.iterrows():
        if row["VXID"] not in transaction_info:
            transaction_info[row["VXID"]] = {}
            transaction_info[row["VXID"]]["CID"] = 0
            transaction_info[row["VXID"]]["IID"] = 0
        if row["CID"] != -1 and row["CID"] != 0:
            transaction_info[row["VXID"]]["CID"] = row["CID"]
        if row["IID"] != -1 and row["IID"] != 0:
            transaction_info[row["VXID"]]["IID"] = row["IID"]
    with open("./Dataset/w2v1.txt", 'a+', encoding='utf-8') as file:
        for vxid in transaction_info:
            file.write(str(transaction_info[vxid]["CID"]) + " ")
    with open("./Dataset/w2v2.txt", 'a+', encoding='utf-8') as file:
        for vxid in transaction_info:
            file.write(str(transaction_info[vxid]["IID"]) + " ")

def get_csv_data(log_file):
    dataset =   read_csv_all_data(log_file)
    workingsets = getWorkingsets(dataset)
    X_data, y_data = [], []
    for k in range(1,6):
        for vxid in workingsets:
            temp_X, temp_y = [], []
            for i in range(0, k):
                temp_X.append(tuple(workingsets[vxid][i]))
            for j in range(k, len(workingsets[vxid])):
                temp_y.append(tuple(workingsets[vxid][j]))
            X_data.append(temp_X)
            y_data.append(temp_y)
    return X_data, y_data

if __name__ == '__main__':
    # --------------------- result analysis ---------------------
    for term_num in [40]:
        # For accumulating the conflict errors or other errors still existing in concurrency
        log_file    =   "./Dataset/Result/Transformer-" + str(term_num) + "-" + str(logid) + ".csv"
        dataset = read_csv_all_data(log_file)

        error_txn = []
        total_txn = {}
        abort_txn = []
        data_occupy = {}
        txn_occupy = {}
        error_num   =   0

        # Just for counting the number of transactions
        raw_log_file = "./Dataset/DATA-LOG-" + str(logid) + ".csv"
        raw_dataset = read_csv_all_data(raw_log_file)
        all_txn = []
        for index, row in raw_dataset.iterrows():
            if row["VXID"] not in all_txn:
                all_txn.append(row["VXID"])

        for index, row in dataset.iterrows():
            if row["VXID"] not in total_txn:
                total_txn[row["VXID"]] = 0
            if row["VXID"] not in txn_occupy:
                txn_occupy[row["VXID"]] = 0
            if row["TYPE"] == "ROLLBACK":
                if row["VXID"] not in abort_txn:
                    abort_txn.append(row["VXID"])
            if row["TYPE"] == "COMMIT" or row["TYPE"] == "ROLLBACK":
                if txn_occupy[row["VXID"]] != 0 and txn_occupy[row["VXID"]] in data_occupy:
                    data_occupy.pop(txn_occupy[row["VXID"]])
                    # print("Transaction " + str(row["VXID"]) + " pops out Data " + str(txn_occupy[row["VXID"]]) + ".")
                txn_occupy.pop(row["VXID"])
                # print("Transaction " + str(row["VXID"]) + " Ends!")
            if row["IID"] != 0:
                if row["IID"] in data_occupy and data_occupy[row["IID"]] != row["VXID"] and row["VXID"] not in error_txn:
                    error_num += 1
                    error_txn.append(row["VXID"])
                    # print("[Error] Transaction " + str(row["VXID"]) + " requires Data " + str(row["IID"]) + ", but Transaction " + str(data_occupy[row["IID"]]) + " Occupying!")
                elif row["IID"] not in data_occupy:
                    txn_occupy[row["VXID"]] = row["IID"]
                    data_occupy[row["IID"]] = row["VXID"]
                    # print("Transaction " + str(row["VXID"]) + " occupies Data " + str(row["IID"]) + ".")

        dataset['TIME'] = pd.to_datetime(dataset['TIME'])

        # calculate time
        max_time = dataset['TIME'].max()
        min_time = dataset['TIME'].min()
        execution_time = max_time - min_time

        error_num   += max((len(all_txn)-len(total_txn)), 0)
        error_num   += len(abort_txn)

        print("=================" + str(log_file) + "=================")
        print("Execution Time: " + str(execution_time))
        print("ALL Num: " + str(len(all_txn)))
        print("Total Num: " + str(len(total_txn)))
        print("Abort Num: " + str(len(abort_txn)))
        print("Total Error Num: " + str(error_num))

        if(len(total_txn) > 0):
            print("Error Rate: " + str((error_num/len(total_txn))))
    # --------------------- result analysis ---------------------

    # --------------------- generate cache file ---------------------

    # X_data, y_data = get_csv_data("./Dataset/DATA-LOG-80.csv")
    # write_data_to_csv(X_data, y_data)

    # for logid in [20, 40, 60, 80]:
    #     filename = './Output/Text/Data_' + str(logid) + '_output.csv'
    #     log_file = "./Dataset/DATA-LOG-" + str(logid) + ".csv"
    #     X_data, y_data = get_csv_data(log_file)
    #     write_data_to_csv(X_data, y_data, filename)

        # dataset = read_csv_all_data(log_file)
        # dataset.drop(['STATEMENT'], axis=1, inplace=True)
        # new_LogFile = log_file.rstrip(".csv") + "_new.csv"
        # dataset.to_csv(new_LogFile)
        # print(dataset.head(20))
        # get_transaction_info(log_file)

        # model_name = "./Model/Transformer/TRANSFORMER_TEST_20_2_5000.keras"
        # model = tf.keras.models.load_model(model_name)

    # --------------------- generate cache file ---------------------

    # setences = get_sentences(log_file)
    # print(setences.head(30))


