#!/bin/bash

# 源码安装Postgresql源码，需要设置好PG目录和llvm-config目录

postgresql_dir="/opt/pg/pgsql"

cd "$postgresql_dir"

make clean

./configure --prefix=`pwd`/debug

make -j4 && make install