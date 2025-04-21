# CUDA Concurrent Stack Implementations: Scan Stack vs. CAS Stack

## Project Goal

This project implements and compares two concurrent stack data structures using CUDA:

1.  **CAS Stack**: A standard lock-free stack using Compare-and-Swap (CAS) operations and tagged pointers for memory reclamation, based on common lock-free linked-list techniques.
2.  **Scan Stack**: An optimized, search-based concurrent stack designed for GPU architectures, as proposed by Noah South. This approach avoids a single atomic 'top' pointer, instead using a search-based method combined with optimizations to leverage GPU memory patterns and reduce contention.

The primary goal is to evaluate the performance optimizations offered by the Scan Stack approach, including Lowest-Area searching and Elimination techniques, in terms of throughput and scalability compared to the CAS-based implementation on a GPU.

## Implementations

* `CAS_stack.cu`: Contains the implementation of a lock-free concurrent stack using atomic Compare-and-Swap (CAS) primitives and tagged pointers with a free list for node management and ABA problem prevention. It provides push, pop, and peek operations.
    * **⚠️ Compatibility Warning**: This implementation requires GPU hardware and compiler support for 128-bit atomic operations (`unsigned __int128`). This may not be available on older GPU architectures. Ensure your target architecture (e.g., `sm_86` used in compile commands) supports this feature. Lack of support could lead to silent errors.
* `Scan_stack.cu`: Contains the implementation of the Scan Stack. This uses an array-based structure and implements push and pop operations that search for the stack top. Key optimizations include:
    * **Lowest-Area (LA) Searching**: Reduces the scan area by estimating the top's location.
    * **Warp/Block-Level Elimination**: Lightweight pairing of push/pop operations within a block using shared memory.
    * **Grid-Wide Elimination-Backoff**: A more robust elimination strategy using global memory arrays and atomic operations to match operations across the grid, adapting to contention levels.
* `temp.cpp`: Appears to be a host-side C++ file potentially used for testing or managing the CUDA kernels (like the CAS stack), including verification logic and performance measurement.

## Contention Handling & Challenges

* **CAS Stack**: Relies on atomic CAS for synchronization and tagged pointers to handle the ABA problem inherent in linked-list structures.
* **Scan Stack**: Distributes contention via scanning. It uses invalidation cells (-2) and retracing logic during pops and pushes to handle race conditions like the "step over" problem. Elimination techniques (both local and grid-wide) further reduce contention on the central array structure. Balancing elimination overhead versus direct access is a key consideration.

## Performance Comparison

The project aims to benchmark these implementations under various workloads (e.g., random mix, push-only, pop-only). The expectation, based on the reference paper and the report, is that the Scan Stack, particularly with Elimination and Lowest-Area (EL+LA) optimizations, will demonstrate significantly higher throughput compared to the non-optimized or CAS versions, especially under mixed workloads where elimination is effective. Memory coalescing is also a factor considered in the Scan Stack design.

## Reference Paper

The Scan Stack implementation is based on the concepts presented in:

* South, Noah Brennen, "Scan Stack: A Search-based Concurrent Stack for GPU" (2022). Electronic Theses and Dissertations. 2459.
    [https://egrove.olemiss.edu/etd/2459](https://egrove.olemiss.edu/etd/2459)

## How to Compile and Run

*(Compilation commands based on Report.pdf)*

* **Scan Stack**:
    ```bash
    nvcc -O2 --allow-unsupported-compiler -ccbin g++-9 -std=c++17 -arch=sm_86 -lineinfo -res-usage -src-in-ptx Scan_stack.cu -o output && ./output
    ```
* **CAS Stack**:
    ```bash
    nvcc -O2 --allow-unsupported-compiler -ccbin g++-9 -std=c++17 -arch=sm_86 -lineinfo -res-usage -src-in-ptx CAS_stack.cu -o output && ./output
    ```

## Project Repository

* [https://github.com/Vrajb24/ScanStack-Implementation.git](https://github.com/Vrajb24/ScanStack-Implementation.git)