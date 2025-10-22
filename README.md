# CUDA and CPU Example Codes

This repository contains example programs for **matrix multiplication** and **vector addition** that are written to run on **CPU-only setups** but can also run on **GPU with CUDA** without any modification.  

The goal is to demonstrate how CUDA code can be structured so it works on both CPU and GPU environments.

## Programs Included

1. **matrix_mul.cpp** – Performs multiplication of two 3x3 matrices.
2. **vectors_mul.cpp** – Performs addition of two vectors.

## How to Run on CPU

1. Make sure you have a C++ compiler installed (Visual Studio, g++, etc.).
2. Open a terminal or command prompt.
3. Compile the code:

   ```bash
   cl /EHsc matrix_mul.cpp
   cl /EHsc vectors_mul.cpp





