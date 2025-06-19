This directory contains the source codes for the experimental workloads, including YCSB, TPC-C benchmarksql and PostgreSQL-13.9. All above are derived from existing projects:

- [YCSB: Yahoo! Cloud Serving Benchmark](https://github.com/brianfrankcooper/YCSB): We removed unrelated parts of the project and only use the JDBC part in PostgreSQL. We also modified a few settings such as making `autocommit` false.
- [TPC-C benchmarksql](https://github.com/brvlant/benchmarksql-1): We applied new_order and payment transactions in the experiments. Settings are also modified as above.
- [PostgreSQL Download](https://www.postgresql.org/ftp/source/v13.9/)