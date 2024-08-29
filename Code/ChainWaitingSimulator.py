import threading
import psycopg2
import random
import time
import string
import csv

# Simulate a chain-waiting situation for chain-splitting ablation experiments

# Configurations for PostgreSQL
myhost = "127.0.0.1"        # host
mydb = "postgres"           # database name
myrole = "Administrator"    # user role name
mypwd = ""                  # password

max_k = 5  # how many repeatitive tests are needed to produce an average output data
file_name = './Output/chain_waiting_simulation.csv'  # where to put the output data

TIMEOUT_TIME = 0.06     # seconds, the time to be regarded as timeout
DATABASE_SIZE = 20000   # how many data items inserted into the database
TEST_SIZE = 10000       # how many data items are used in testing
TXN_SIZE = 20           # how many operations inside each transaction
ORDER_NUM = 2           # the k value of the prediction model (just for simulation, not so useful)

# ---------------------- no need to change ---------------------------

# Configs for chain waiting simulator
bool_chain = False          # whether to use chain waiting detection or not
bool_confidence = False

TXN_NUM = 400       # how many transactions in total
THREAD_NUM = 40     # how many threads(transactions, one thread for one concurrent transaction)

chain_percentage        = 0.2   # the percentage of transactions to form a long chain
chain_possibility       = 0.2   # the possibility to form a chain
false_chain_possibility = 0.2   # the possibility for a chain to be a false positive chain

# Others
count_commit = 0    # count the number of successfully committed transactions
count_rollback = 0  # count the number of unsuccessfully committed transactions (due to update error or something else)
count_timeout = 0   # count the number of timeout transactions (not counted into the previous two)
execute_time = 0.0  # total time for all the threads to execute given transactions

current_txn_num = 0
lock = threading.Lock()         # global lock for current_txn_num

transaction_level = {}          # concurrent transaction seeds and their levels in the chain
transaction_seeds = {}          # contrary to the transaction_level
transaction_confidence = {}     # how confident we are to prediction this transaction (default 0.9)
to_wake_txn = []                # all the transactions to be waken
                                # A transaction is removed from waiting_txn only under 2 circumstances:
                                #    1. The transaction has been waiting for too long and reached the timeout limit (50 milliseconds in this simulator)
                                #    2. It is added into to_wake_txn list, which means it no longer needs waiting

waiting_chain = {}              # The waiting chain (only records transactions that are waiting for other transactions)
waited_chain = {}               # The waited chain (only records transactions that are being waited by other transactions)

backup_waiting_chain = {}       # The backup of waiting chain
backup_waited_chain = {}        # The backup of waited chain

# ---------------------- no need to change ---------------------------

def generate_random_string(length=10):
    # string.ascii_letters
    characters = string.ascii_letters + string.digits
    random_string = ''.join(random.choices(characters, k=length))
    return random_string


# With chain waiting detection simulation
def chain_waiting_detect_txn(conn, useConfidence):
    global count_commit
    global count_rollback
    global count_timeout
    global execute_time
    global current_txn_num

    conn.autocommit = False

    while True:
        lock.acquire()
        if current_txn_num >= TXN_NUM:
            lock.release()
            break
        try:
            seed = transaction_seeds[current_txn_num]
            current_txn_num += 1
        finally:
            lock.release()

        try:
            cursor = conn.cursor()
            # every transaction start from a seed
            for i in range(0, TXN_SIZE):
                id = seed % TEST_SIZE + i
                real_id = seed % DATABASE_SIZE + i
                if i == ORDER_NUM - 1:
                    time.sleep(0.08)  # prediction  time
                    random.seed(random.random())
                    if useConfidence:
                        random_value = random.random()
                        if transaction_confidence[seed] > random_value:
                            if (seed in waiting_chain and transaction_level[seed] % 2 == 1):
                                count = 0
                                while (seed not in to_wake_txn):
                                    time.sleep(TIMEOUT_TIME / 10)
                                    count = count + 1
                                    if count >= 10:
                                        count_timeout = count_timeout + 1
                                        raise Exception("Timeout Error")
                    else:
                        random_value = random.random()
                        if random_value < transaction_confidence[seed]:
                            pass
                        if (seed in waiting_chain and transaction_level[seed] % 2 == 1):
                            count = 0
                            while (seed not in to_wake_txn):
                                time.sleep(TIMEOUT_TIME / 10)
                                count = count + 1
                                if count >= 10:
                                    count_timeout = count_timeout + 1
                                    raise Exception("Timeout Error")
                if i % 5 == 4:
                    random_str = generate_random_string(10)
                    statement = "UPDATE test SET name = %s WHERE id = %s"
                    cursor.execute(statement, (random_str, real_id))
                    time.sleep(0.02)
                else:
                    statement = "SELECT * FROM test WHERE id = %s"
                    cursor.execute(statement, (real_id,))
                    time.sleep(0.01)
            # commit transaction
            if seed in waited_chain:
                to_wake_txn.append(waited_chain[seed])
                if waited_chain[seed] in waiting_chain:
                    waiting_chain.pop(waited_chain[seed])
                waited_chain.pop(seed)
            if seed in waiting_chain:
                waiting_chain.pop(seed)
            count_commit = count_commit + 1
            conn.commit()
        except (Exception, psycopg2.DatabaseError) as error:
            # roll back
            if seed in waited_chain:
                to_wake_txn.append(waited_chain[seed])
                if waited_chain[seed] in waiting_chain:
                    waiting_chain.pop(waited_chain[seed])
                waited_chain.pop(seed)
            if seed in waiting_chain:
                waiting_chain.pop(seed)
            conn.rollback()
            count_rollback = count_rollback + 1
        finally:
            cursor.close()
    conn.close()


# Without chain waiting detection simulation
def non_chain_waiting_detect_txn(conn):
    global count_commit
    global count_rollback
    global count_timeout
    global execute_time
    global current_txn_num

    conn.autocommit = False
    while True:
        seed = 0
        lock.acquire()
        if current_txn_num >= TXN_NUM:
            lock.release()
            break
        try:
            seed = transaction_seeds[current_txn_num]
            current_txn_num += 1
        finally:
            lock.release()

        try:
            cursor = conn.cursor()
            for i in range(0, TXN_SIZE):
                id = seed % TEST_SIZE + i
                real_id = seed % DATABASE_SIZE + i
                if i == ORDER_NUM - 1:
                    time.sleep(0.08)  # prediction  time
                    random.seed(random.random())
                    random_value = random.random()
                    if (seed in waiting_chain):
                        count = 0
                        while (seed not in to_wake_txn):
                            time.sleep(TIMEOUT_TIME / 5)
                            count = count + 1
                            if count >= 5:
                                count_timeout = count_timeout + 1
                                time.sleep(0.01 * TXN_SIZE)
                                raise Exception("Timeout Error")
                if i % 5 == 3 or i % 5 == 4:
                    random_str = generate_random_string(10)
                    statement = "UPDATE test SET name = %s WHERE id = %s"
                    cursor.execute(statement, (random_str, real_id))
                    time.sleep(0.02)
                else:
                    statement = "SELECT * FROM test WHERE id = %s"
                    cursor.execute(statement, (real_id,))
                    time.sleep(0.01)
            # commit
            if seed in waited_chain:
                to_wake_txn.append(waited_chain[seed])
                if waited_chain[seed] in waiting_chain:
                    waiting_chain.pop(waited_chain[seed])
                waited_chain.pop(seed)
            if seed in waiting_chain:
                waiting_chain.pop(seed)
            conn.commit()
            count_commit = count_commit + 1
        except (Exception, psycopg2.DatabaseError) as error:
            if seed in waited_chain:
                to_wake_txn.append(waited_chain[seed])
                if waited_chain[seed] in waiting_chain:
                    waiting_chain.pop(waited_chain[seed])
                waited_chain.pop(seed)
            if seed in waiting_chain:
                waiting_chain.pop(seed)
            conn.rollback()
            count_rollback = count_rollback + 1
        finally:
            cursor.close()
    conn.close()

# generate transactions and wait chains with chain percentage, and initialize parameters for testing
def generate_transactions_with_chainPercentage():
    global transaction_level
    global transaction_seeds
    global transaction_confidence
    global backup_waiting_chain
    global backup_waited_chain
    global chain_percentage
    global false_chain_possibility

    transaction_level = {}  # concurrent transaction seeds and their levels in the chain
    transaction_seeds = {}  # contrary to the transaction_level
    transaction_confidence = {}  # how confident we are to prediction this transaction (default 0.9)

    # The seed is the first data accessed by the transaction, we use it to represent different transactions in our simulator (similar to a transaction ID)
    # When a seed is set, the working set of the transaction is the consecutive TXN_SIZE data items starting from the seed
    # Thus we can use a seed to generate a transaction and conduct conflict detection easily
    initial_seed = TEST_SIZE - TXN_SIZE
    last_seed = initial_seed
    transaction_level[initial_seed] = 0
    transaction_confidence[initial_seed] = 0.9
    transaction_seeds[0] = initial_seed

    for i in range(1, TXN_NUM):
        if i < TXN_NUM * chain_percentage:
            # First, pick up some random transactions as "False Positive",
            # which means they don't conflict with other transactions but still detected as conflicted
            random.seed(random.random())
            random_num = random.random()
            if (transaction_level[last_seed] + 1 % 2 == 1) and (random_num < false_chain_possibility):
                seed = last_seed - TXN_SIZE
                transaction_confidence[seed] = 0.1
            else:
                seed = last_seed - TXN_SIZE + 1
                transaction_confidence[seed] = 0.9
            transaction_level[seed] = transaction_level[last_seed] + 1
            transaction_seeds[i] = seed
            backup_waiting_chain[seed] = last_seed
            backup_waited_chain[last_seed] = seed
            last_seed = seed
        else:
            seed = last_seed - TXN_SIZE
            transaction_level[seed] = 0
            transaction_seeds[i] = seed
            transaction_confidence[seed] = 0.9
            last_seed = seed

# generate transactions and wait chains with chain possibility, and initialize parameters for testing
def generate_transactions_with_chainPossibility():
    global transaction_level
    global transaction_seeds
    global transaction_confidence
    global backup_waiting_chain
    global backup_waited_chain
    global chain_possibility
    global false_chain_possibility

    transaction_level = {}  # concurrent transaction seeds and their levels in the chain
    transaction_seeds = {}  # contrary to the transaction_level
    transaction_confidence = {}  # how confident we are to prediction this transaction (default 0.9)

    # The seed is the first data accessed by the transaction, we use it to represent different transactions in our simulator (similar to a transaction ID)
    # When a seed is set, the working set of the transaction is the consecutive TXN_SIZE data items starting from the seed
    # Thus we can use a seed to generate a transaction and conduct conflict detection easily
    initial_seed = TEST_SIZE - TXN_SIZE
    last_seed = initial_seed
    transaction_level[initial_seed] = 0
    transaction_confidence[initial_seed] = 0.9
    transaction_seeds[0] = initial_seed

    for i in range(1, TXN_NUM):
        random.seed(random.random())
        random_num = random.random()

        if random_num < chain_possibility:
            # First, pick up some random transactions as "False Positive",
            # which means they don't conflict with other transactions but still detected as conflicted
            random.seed(random.random())
            random_num2 = random.random()
            if (transaction_level[last_seed] + 1 % 2 == 1) and (random_num2 < false_chain_possibility):
                seed = last_seed - TXN_SIZE
                transaction_confidence[seed] = 0.1
            else:
                seed = last_seed - TXN_SIZE + 1
                transaction_confidence[seed] = 0.9
            transaction_level[seed] = transaction_level[last_seed] + 1
            transaction_seeds[i] = seed
            backup_waiting_chain[seed] = last_seed
            backup_waited_chain[last_seed] = seed
            last_seed = seed
        else:
            seed = last_seed - TXN_SIZE
            transaction_level[seed] = 0
            transaction_seeds[i] = seed
            transaction_confidence[seed] = 0.9
            last_seed = seed


# Initialize the waiting chains
def restore_wait_chains():
    global to_wake_txn
    global waited_chain
    global waiting_chain
    to_wake_txn = []
    waiting_chain = {}
    waited_chain = {}
    for seed in backup_waiting_chain:
        waiting_chain[seed] = backup_waiting_chain[seed]
    for seed in backup_waited_chain:
        waited_chain[seed] = backup_waited_chain[seed]


# Test concurrent transactions in multi-threads
def concurrent_transactions():
    global bool_confidence
    global bool_chain
    global count_commit
    global count_rollback
    global count_timeout
    global execute_time
    global current_txn_num

    count_timeout   = 0
    count_commit    = 0
    count_rollback  = 0
    execute_time    = 0

    try:
        # Connect to PostgreSQL and run tests
        threads = []

        for m in range(0, THREAD_NUM):
            temp_conn = psycopg2.connect(
                host=myhost,
                database=mydb,
                user=myrole,
                password=mypwd
            )
            if not bool_chain:
                temp_thread = threading.Thread(target=non_chain_waiting_detect_txn, args=(temp_conn,))
            else:
                temp_thread = threading.Thread(target=chain_waiting_detect_txn, args=(temp_conn, bool_confidence))
            threads.append(temp_thread)

        current_txn_num = 0

        time1 = time.time()
        for j in range(0, THREAD_NUM):
            threads[j].start()

        for k in range(0, THREAD_NUM):
            threads[k].join()
        time2 = time.time()
        execute_time = time2 - time1

        print("Excution Time：" + str(execute_time))
        print("Commit Transactions：" + str(count_commit))
        print("Error Transactions：" + str(count_rollback))
        print("Timeout Transactions：" + str(count_timeout))

    except (Exception, psycopg2.DatabaseError) as error:
        print("Error connecting to PostgreSQL:", error)


# Insert the initial values into the `test` table for testing, only needed for the first time using the program
def insert_values():
    conn = psycopg2.connect(
        host=myhost,
        database=mydb,
        user=myrole,
        password=mypwd
    )
    cursor = conn.cursor()
    for seed in range(0, DATABASE_SIZE - TXN_SIZE):
        name_seed = str((7 * seed + 11) % DATABASE_SIZE)  # Randomly give each seed an initial name, no specific meaning
        insert_statement = "INSERT INTO test VALUES(%s, %s)"
        cursor.execute(insert_statement, (seed, name_seed))
    conn.commit()
    cursor.close()
    conn.close()


# Conduct tests for experiments and save the output in the given file
if __name__ == '__main__':
    # insert_values() # to insert data into the test table
    # start testing
    for TXN_NUM in [100, 200, 300, 400]:
        for chain_possibility in [0.2, 0.8]:
            for false_chain_possibility in [0.2, 0.8]:
                for THREAD_NUM in [5, 10, 15, 20, 25, 30, 35, 40]:
                    for bool_chain in [False, True]:
                        generate_transactions_with_chainPossibility()  # generate the needed transactions
                        if bool_chain:
                            for bool_confidence in [False, True]:
                                avg_timeout = 0
                                avg_commit = 0
                                avg_time = 0.0
                                avg_throughput = 0.0
                                for k in range(0, max_k):
                                    print("---------------------- txn_num = " + str(TXN_NUM) + ", thread_num = " + str(
                                        THREAD_NUM) + ", chain_possibility = " + str(
                                        chain_possibility) + ", chain_percentage = " + str(
                                        chain_percentage) + ", false_chain = " + str(
                                        false_chain_possibility) + ", bool_chain = " + str(
                                        bool_chain) + ", bool_confidence = " + str(
                                        bool_confidence) + "----------------------")
                                    restore_wait_chains()  # initialize the waiting chains
                                    concurrent_transactions()  # start concurrent execution
                                    avg_timeout += count_timeout
                                    avg_commit += count_commit
                                    avg_time += execute_time
                                avg_timeout = avg_timeout / max_k
                                avg_commit = avg_commit / max_k
                                avg_time = avg_time / max_k
                                avg_throughput = avg_commit / avg_time
                                data = {
                                    "txn_num": TXN_NUM,
                                    "thread_num": THREAD_NUM,
                                    "chain_possibility": chain_possibility,
                                    "chain_percentage": chain_percentage,
                                    "false_chain_possibility":false_chain_possibility,
                                    "bool_chain": bool_chain,
                                    "bool_confidence": bool_confidence,
                                    "avg_time": avg_time,
                                    "avg_timeout": avg_timeout,
                                    "avg_commit": avg_commit,
                                    "avg_throughput": avg_throughput
                                }
                                # check whether the file exists
                                file_exists = False
                                try:
                                    with open(file_name, mode='r') as file:
                                        file_exists = True
                                except FileNotFoundError:
                                    pass
                                with open(file_name, mode='a+', newline='') as file:
                                    writer = csv.writer(file)
                                    if not file_exists:
                                        writer.writerow(data.keys())
                                    writer.writerow(data.values())
                        else:
                            avg_timeout = 0
                            avg_commit = 0
                            avg_time = 0.0
                            avg_throughput = 0.0
                            for k in range(0, max_k):
                                print("---------------------- txn_num = " + str(TXN_NUM) + ", thread_num = " + str(
                                    THREAD_NUM) + ", chain_possibility = " + str(chain_possibility) + ", chain_percentage = " + str(
                                    chain_percentage) + ", false_chain = " + str(false_chain_possibility) + ", bool_chain = " + str(
                                    bool_chain) + ", bool_confidence = " + str(bool_confidence) + "----------------------")
                                restore_wait_chains()  # initialize the waiting chains
                                concurrent_transactions()  # start concurrent execution
                                avg_timeout += count_timeout
                                avg_commit += count_commit
                                avg_time += execute_time
                            avg_timeout = avg_timeout / max_k
                            avg_commit = avg_commit / max_k
                            avg_time = avg_time / max_k
                            avg_throughput = avg_commit / avg_time
                            data = {
                                "txn_num": TXN_NUM,
                                "thread_num": THREAD_NUM,
                                "chain_possibility": chain_possibility,
                                "chain_percentage": chain_percentage,
                                "false_chain_possibility": false_chain_possibility,
                                "bool_chain": bool_chain,
                                "bool_confidence": bool_confidence,
                                "avg_time": avg_time,
                                "avg_timeout": avg_timeout,
                                "avg_commit": avg_commit,
                                "avg_throughput": avg_throughput
                            }
                            # check whether the file exists
                            file_exists = False
                            try:
                                with open(file_name, mode='r') as file:
                                    file_exists = True
                            except FileNotFoundError:
                                pass
                            with open(file_name, mode='a+', newline='') as file:
                                writer = csv.writer(file)
                                if not file_exists:
                                    writer.writerow(data.keys())
                                writer.writerow(data.values())