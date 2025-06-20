#!/bin/bash

make clean

./configure --prefix=`pwd`/debug

make -j4 && make install

cd debug/bin

./initdb -D ../data