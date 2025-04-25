# Lock-Free Linked-List Stack using CAS for GPU

## Project Report

This repository contains the implementation and report for a lock-free concurrent stack designed for GPUs. The implementation utilizes a linked-list structure built upon a pre-allocated node pool and employs Compare-and-Swap (CAS) atomic operations for thread safety.

**Authors:**
* Vraj Patel (241110080)
* Aditya Azad (241110008)
* Sparsh Sehgal (24111071)


**(The full implementation report can be found in `report.pdf` or `main.tex`)**

## Abstract

This project implements a lock-free concurrent stack for GPUs based on a linked-list structure and Compare-and-Swap (CAS) atomic operations. This approach uses dynamically linked nodes allocated from a pre-defined memory pool. Key features include a tagged pointer scheme using 64-bit atomics to prevent the ABA problem and a concurrent free list for efficient node reclamation and reuse. We analyze the structure, core operations (push and pop), memory management strategy, and performance characteristics, including scalability results from experimental tests with varying pool capacities.

## Features

* **Lock-Free Operations:** Push and pop operations are implemented without locks using atomic CAS primitives.
* **Linked-List Structure:** Uses nodes allocated from a pre-defined pool in GPU global memory.
* **ABA Problem Prevention:** Employs a 64-bit tagged pointer (32-bit index + 32-bit tag) mechanism, validated using a single atomic CAS operation.
* **Concurrent Free List:** Includes a lock-free free list, also using tagged pointers, for efficient recycling of nodes without global synchronization.
* **Verification Suite:** Includes tests for correctness, empty stack pops, and pool exhaustion.
* **Scalability Testing:** Measures throughput under varying operation counts and pool capacities.

## Implementation Details

* **Tagged Pointers:** A `uint64_t` atomic variable holds both a 32-bit index (relative to the node pool start) and a 32-bit tag. The tag is incremented upon node recycling.
* **CAS Operations:** `cuda::atomic<uint64_t>::compare_exchange_weak` is used to atomically update the `stack_top` and `free_list_top` pointers, checking both index and tag simultaneously.
* **Memory Orders:** Uses CUDA C++ memory orders (e.g., `memory_order_acquire`, `memory_order_release`, `memory_order_relaxed`) for correct synchronization.
* **Node Pool:** Nodes are pre-allocated in an array (`node_pool`). Initial allocation uses `next_free_node_idx`.

## Performance Summary

* **Contention Bottleneck:** Performance is primarily limited by contention on the atomic `stack_top` and `free_list_top` pointers, especially under high concurrency.
* **Scalability:**
    * With a large node pool (\~100k nodes), throughput decreases monotonically as thread count increases.
    * With a smaller node pool (\~10k nodes), throughput showed non-monotonic behavior, initially increasing significantly before dropping, possibly due to cache effects and free list dynamics.
* **Pool Capacity:** Performance characteristics are sensitive to the size of the pre-allocated node pool.
* **Warp Scheduling:** Internal tests indicated a potential performance benefit (30-40% speedup) when scheduling operations per warp rather than per thread, although the final tests focused on thread-level assignment.

## Requirements

* NVIDIA GPU with CUDA support (tested with compute capability 8.6 - Ampere)
* CUDA Toolkit (tested with versions compatible with C++17 atomics)
* C++17 compliant compiler (e.g., g++ version 9 or later)

## Compilation and Usage

The primary implementation is in `CAS_stack.cu`.

To compile and run:

```bash
# Adjust -ccbin and -arch=sm_XX as needed for your compiler and GPU architecture
# For NVIDIA A40 GPU, use -arch=sm_86
nvcc -O2 --allow-unsupported-compiler -ccbin=g++-9 -std=c++17 -arch=sm_86 -lineinfo -res-usage CAS_stack.cu -o cas_output

./cas_output
