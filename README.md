# Mochi: Predictive and Semi-deterministic Concurrency Control for High-performance OLTP Databases

This repository contains the scripts and experimental workload for the article:

**Predictive and Semi-deterministic Concurrency Control for High-performance OLTP Databases**

## Overview

![image-20250403000451613](https://my-typora-image-host.oss-cn-hangzhou.aliyuncs.com//img/image-20250403000451613.png) 

We propose **Mochi**, a semi-deterministic concurrency control method that proactively predicts, detects, and avoids potential conflicts among concurrent transactions during execution. This repository provides the codebase used to implement, analyze, and evaluate the proposed approach. 

## Repository Structure

- **Code/**  
  Contains Python scripts for data analysis and model training related to conflict prediction and scheduling.
  
- **Workloads/**  
  Contains the Java implementation of the YCSB and TPC-C benchmark workload used in our experiments.

## Experimental Environment

All experiments were conducted on **PostgreSQL 13.9** with modified concurrency control mechanisms integrated to support Mochi in **Ubuntu 20.04**. 

