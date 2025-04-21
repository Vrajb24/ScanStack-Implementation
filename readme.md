# CUDA Concurrent Stack Implementations: CAS vs. Scan Stack

## Project Goal

This project implements and compares two concurrent stack data structures using CUDA:

1.  A standard lock-free stack using Compare-and-Swap (CAS) operations[cite: 3].
2.  An optimized, search-based "Scan Stack" designed for GPU architectures[cite: 1, 10].

The primary goal is to evaluate the performance optimizations offered by the Scan Stack approach in terms of throughput and scalability compared to the traditional CAS-based implementation on a GPU[cite: 12, 13, 43, 44].

## Implementations

* `CAS_stack.cu`: Contains the implementation of a standard lock-free concurrent stack using atomic Compare-and-Swap (CAS) primitives for thread synchronization[cite: 3]. It includes basic push, pop, and peek operations with overflow and underflow checks.
* `Scan_stack.cu`: Contains the implementation of the Scan Stack[cite: 1]. This approach avoids a single atomic 'top' pointer and instead uses a search-based method to find the stack top, aiming to leverage GPU memory access patterns and reduce contention[cite: 98, 102, 104]. It incorporates optimizations like Lowest-Area searching and Elimination techniques[cite: 105, 203].

## Comparison

The project aims to benchmark these two implementations under various workloads (e.g., random mix of push/pop, push-only, pop-only) to compare their performance characteristics, particularly focusing on the effectiveness of the Scan Stack optimizations[cite: 230, 231, 232, 235].

## Reference Paper

The Scan Stack implementation is based on the concepts presented in the following paper:

* South, Noah Brennen, "Scan Stack: A Search-based Concurrent Stack for GPU" (2022). Electronic Theses and Dissertations. 2459.
    [https://egrove.olemiss.edu/etd/2459](https://egrove.olemiss.edu/etd/2459) [cite: 2]