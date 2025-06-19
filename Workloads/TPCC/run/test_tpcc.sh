#!/bin/bash

# 定义配置文件路径
CONFIG_FILE="/opt/pg/tpcc/benchmarksql-5.0/benchmarksql-5.0/run/props.pg"
# 定义日志文件路径
LOG_FILE="/opt/pg/tpcc/benchmark.log"

# 定义需要测试的 concurrencyControl 值
concurrency_controls=("mogi")

# 定义数据文件路径选项
data_files=(
    "/opt/pg/tpcc/data-80.csv"
    # "/opt/pg/tpcc/data-60.csv"
    # "/opt/pg/tpcc/data-40.csv"
    # "/opt/pg/tpcc/data-20.csv"
)

# 备份原始配置文件
cp $CONFIG_FILE "${CONFIG_FILE}.bak"

# 清空日志文件
# echo "" > $LOG_FILE

# 定义一个函数用于运行基准测试
run_benchmark() {
    local control=$1
    local needsRetry=$2
    local terminals=$3
    local filepath=$4  # 新增参数 filepath
    local kValue=$5    # 新增参数 kValue

    # echo "============================== 设置 concurrencyControl 为 $control, kValue 为 $kValue, needsRetry 为 $needsRetry, terminals 为 $terminals, filepath 为 $filepath ==============================" | tee -a $LOG_FILE

    # 更改配置文件中的 concurrencyControl, kValue, needsRetry, terminals 和 filepath 值
    sed -i "s/^concurrencyControl=.*/concurrencyControl=$control/" $CONFIG_FILE
    sed -i "s/^kValue=.*/kValue=$kValue/" $CONFIG_FILE
    sed -i "s/^needsRetry=.*/needsRetry=$needsRetry/" $CONFIG_FILE
    sed -i "s/^terminals=.*/terminals=$terminals/" $CONFIG_FILE
    sed -i "s|^loadfilepath=.*|loadfilepath=$filepath|" $CONFIG_FILE
    sed -i "s|^runfilepath=.*|runfilepath=$filepath|" $CONFIG_FILE

    # 设置 conflictDetection 配置项
    if [[ "$control" == "aria" || "$control" == "tictoc" || "$control" == "mogi" ]]; then
        sed -i "s/^conflictDetection=.*/conflictDetection=on/" $CONFIG_FILE
    else
        sed -i "s/^conflictDetection=.*/conflictDetection=off/" $CONFIG_FILE
    fi

    # 每个配置项运行 TPCC 5 遍
    for run in {1..5}
    do
        # echo "----------------------------- 运行 TPCC 基准测试 - 轮次 $run -----------------------------" | tee -a $LOG_FILE

        # 运行 TPCC 基准测试并将输出追加到日志文件
        # cd /opt/pg/tpcc/benchmarksql-5.0/benchmarksql-5.0/run && ./runBenchmark.sh props.pg | tee -a $LOG_FILE

         # 运行 TPCC 基准测试
        cd /opt/pg/tpcc/benchmarksql-5.0/benchmarksql-5.0/run && ./runBenchmark.sh props.pg

        # 等待几秒钟以确保基准测试完成（可根据需要调整）
        sleep 5
    done
}

# 按不同的文件路径和 kValue 进行测试
for filepath in "${data_files[@]}"
do
    for control in "calvin" "s2pl"
    do
        for terminals in 40
        do
            run_benchmark $control "off" $terminals $filepath
        done
    done

    # 设置 needsRetry 为 on 后重新测试 tictoc, mogi
    for control in "tictoc" "aria"
    do
        for terminals in 40
        do
            for kValue in 2
            do
                run_benchmark $control "on" $terminals $filepath $kValue
            done
        done
    done
done

# 恢复原始配置文件
mv "${CONFIG_FILE}.bak" $CONFIG_FILE

# sed -i '/ERROR/d' $LOG_FILE
# sed -i '/FATAL/d' $LOG_FILE

# python3 /opt/pg/tpcc/analyseBenchmarkLog.py

echo "================================== 所有测试已完成 ==================================" | tee -a $LOG_FILE
