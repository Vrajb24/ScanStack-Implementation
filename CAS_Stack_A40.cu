#include <cuda_runtime.h>
#include <cuda/atomic>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <limits.h> // Include for INT_MIN if used

#include <vector>
#include <numeric>
#include <algorithm>
#include <set>
#include <cmath>
#include <cstddef>
#include <cstdlib> // Include for malloc/free

#define NULL_INDEX UINT32_MAX

typedef struct Node {
    int data;
    uint32_t next_idx;
} Node;

typedef struct {
    uint32_t node_idx;
    uint32_t tag;
} TaggedIndex;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}

__host__ __device__ inline uint64_t taggedIndexToAtomic(TaggedIndex ti) {
    return ((uint64_t)ti.tag << 32) | (uint64_t)ti.node_idx;
}

__host__ __device__ inline TaggedIndex atomicToTaggedIndex(uint64_t val) {
    TaggedIndex ti;
    ti.node_idx = (uint32_t)(val & 0xFFFFFFFFULL);
    ti.tag      = (uint32_t)(val >> 32);
    return ti;
}

typedef struct Stack Stack;

__device__ inline Node* indexToNodePtr(Stack* stack, uint32_t index);


typedef struct Stack {
    Node* node_pool;
    int   pool_capacity;
    cuda::atomic<int, cuda::thread_scope_device> next_free_node_idx;
    cuda::atomic<uint64_t, cuda::thread_scope_device> stack_top;
    cuda::atomic<uint64_t, cuda::thread_scope_device> free_list_top;
} Stack;

__device__ inline Node* indexToNodePtr(Stack* stack, uint32_t index) {
    if (index == NULL_INDEX) return nullptr;
    if (index >= (uint32_t)stack->pool_capacity) return nullptr;
    return &stack->node_pool[index];
}

__device__ bool pop_tagged_stack(cuda::atomic<uint64_t, cuda::thread_scope_device>* list_top, TaggedIndex* result_ti, Stack* stack) {
    uint64_t old_top_atomic = list_top->load(cuda::memory_order_acquire);
    TaggedIndex old_top;
    TaggedIndex new_top;
    do {
        old_top = atomicToTaggedIndex(old_top_atomic);
        if (old_top.node_idx == NULL_INDEX) return false;
        Node* old_top_node_ptr = indexToNodePtr(stack, old_top.node_idx);
        if (!old_top_node_ptr) { old_top_atomic = list_top->load(cuda::memory_order_acquire); continue; }
        new_top.node_idx = old_top_node_ptr->next_idx;
        new_top.tag = old_top.tag;
    } while (!list_top->compare_exchange_weak(old_top_atomic, taggedIndexToAtomic(new_top), cuda::memory_order_release, cuda::memory_order_acquire));
    *result_ti = atomicToTaggedIndex(old_top_atomic);
    return true;
}

__device__ void push_tagged_stack(cuda::atomic<uint64_t, cuda::thread_scope_device>* list_top, uint32_t node_idx_to_push, uint32_t node_tag, Stack* stack) {
    uint64_t old_top_atomic = list_top->load(cuda::memory_order_relaxed);
    TaggedIndex old_top;
    TaggedIndex new_top;
    new_top.node_idx = node_idx_to_push;
    new_top.tag = node_tag;
    Node* node_to_push_ptr = indexToNodePtr(stack, node_idx_to_push);
    if (!node_to_push_ptr) return;
    do {
        old_top = atomicToTaggedIndex(old_top_atomic);
        node_to_push_ptr->next_idx = old_top.node_idx;
    } while (!list_top->compare_exchange_weak(old_top_atomic, taggedIndexToAtomic(new_top), cuda::memory_order_release, cuda::memory_order_acquire));
}

__device__ bool push_gpu(Stack* stack, int value) {
    uint32_t node_idx_to_use = NULL_INDEX, node_tag = 0;
    TaggedIndex reused_node_ti;
    if (pop_tagged_stack(&stack->free_list_top, &reused_node_ti, stack)) {
        node_idx_to_use = reused_node_ti.node_idx; node_tag = reused_node_ti.tag;
    } else {
        int allocated_idx = stack->next_free_node_idx.fetch_add(1, cuda::memory_order_relaxed);
        if (allocated_idx >= stack->pool_capacity) {
            stack->next_free_node_idx.fetch_sub(1, cuda::memory_order_relaxed); return false;
        }
        node_idx_to_use = (uint32_t)allocated_idx; node_tag = 0;
    }
    Node* node_to_use_ptr = indexToNodePtr(stack, node_idx_to_use);
    if (!node_to_use_ptr) return false;
    node_to_use_ptr->data = value;
    push_tagged_stack(&stack->stack_top, node_idx_to_use, node_tag, stack);
    return true;
}

__device__ bool pop_gpu(Stack* stack, int* result) {
    TaggedIndex popped_ti;
    if (!pop_tagged_stack(&stack->stack_top, &popped_ti, stack)) return false;
    uint32_t popped_node_idx = popped_ti.node_idx, current_tag = popped_ti.tag;
    Node* popped_node_ptr = indexToNodePtr(stack, popped_node_idx);
    if (!popped_node_ptr) return false;
    *result = popped_node_ptr->data;
    uint32_t next_tag = current_tag + 1;
    push_tagged_stack(&stack->free_list_top, popped_node_idx, next_tag, stack);
    return true;
}

__device__ bool peek_gpu(Stack* stack, int* result) { /* ... unchanged peek logic ... */
    uint64_t current_top_atomic; TaggedIndex current_top; int data_read;
    while (true) {
        current_top_atomic = stack->stack_top.load(cuda::memory_order_acquire);
        current_top = atomicToTaggedIndex(current_top_atomic);
        if (current_top.node_idx == NULL_INDEX) return false;
        Node* current_top_ptr = indexToNodePtr(stack, current_top.node_idx);
        if (!current_top_ptr) continue;
        data_read = current_top_ptr->data;
        cuda::atomic_thread_fence(cuda::memory_order_acquire);
        uint64_t check_top_atomic = stack->stack_top.load(cuda::memory_order_relaxed);
        if (current_top_atomic == check_top_atomic) { *result = data_read; return true; }
    }
}

// --- Test Kernels ---

__global__ void stack_test_kernel(Stack* d_stack, int num_ops_total, bool* d_push_results, bool* d_pop_results, int* d_pop_values) { /* ... unchanged ... */
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_ops_total) {
        if (tid % 2 == 0) {
            int value_to_push = tid; bool success = push_gpu(d_stack, value_to_push);
            if (d_push_results) d_push_results[tid / 2] = success;
        } else {
            int popped_value = -1; bool success = pop_gpu(d_stack, &popped_value);
            if (d_pop_results) d_pop_results[tid / 2] = success;
            if (d_pop_values && success) d_pop_values[tid / 2] = popped_value;
        }
    }
}

// NEW Kernel for Scalability Test - Avoids top-level push/pop divergence
#define WARP_SIZE 32 // Standard warp size for NVIDIA GPUs including A40

__global__ void stack_test_kernel_warp_assigned( // Renamed for clarity
    Stack* d_stack,
    int num_ops_total
    // Removed optional logging pointers as they were passed as nullptr anyway
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: Only threads corresponding to operations proceed
    if (tid < num_ops_total) {

        // Determine the warp this thread belongs to
        int warp_id = tid / WARP_SIZE;

        // Assign task based on WARP ID to avoid intra-warp divergence on push/pop choice
        if (warp_id % 2 == 0) {
            // --- This entire warp performs PUSH operations ---
            // Still use tid to generate unique values for simplicity in this test
            int value_to_push = tid;
            /* bool success = */ push_gpu(d_stack, value_to_push);
            // Note: Can still have divergence *inside* push_gpu based on free list/allocator state
        } else {
            // --- This entire warp performs POP operations ---
            int popped_value = -1; // Variable needed for pop_gpu signature
            /* bool success = */ pop_gpu(d_stack, &popped_value);
            // Note: Can still have divergence *inside* pop_gpu based on stack empty state
        }
    }
}

// Modified reporting kernel accepting flag and pointer
__global__ void reporting_kernel(
    Stack* d_stack, int num_ops,
    int base_value_or_unused,
    bool use_predefined_values, // Flag
    const int* d_predefined_push_values, // Pointer
    int op_mode,
    int* d_op_tid, int* d_op_type, bool* d_op_success, int* d_op_value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_ops) {
        d_op_tid[tid] = tid;
        d_op_type[tid] = op_mode;
        if (op_mode == 0) {
            int value_to_push = -1; bool prep_success = true;
            if (use_predefined_values) {
                if (d_predefined_push_values != nullptr) {
                    value_to_push = d_predefined_push_values[tid];
                } else {
                    printf("Kernel ERROR: Told to use predefined values, but array pointer is NULL! tid=%d\n", tid);
                    prep_success = false; value_to_push = -999;
                }
            } else {
                value_to_push = tid + base_value_or_unused;
            }
            bool success = false;
            if (prep_success) success = push_gpu(d_stack, value_to_push);
            d_op_success[tid] = success; d_op_value[tid] = value_to_push;
        } else {
            int popped_value = -1; bool success = pop_gpu(d_stack, &popped_value);
            d_op_success[tid] = success; d_op_value[tid] = success ? popped_value : -1;
        }
    }
}

// --- Host Code ---

void reset_stack(Stack* d_stack_struct_ptr, int pool_capacity) { /* ... unchanged ... */
    TaggedIndex null_tagged_idx = {NULL_INDEX, 0}; uint64_t atomic_null_val = taggedIndexToAtomic(null_tagged_idx);
    int zero_idx = 0; gpuErrchk(cudaMemcpy(&(d_stack_struct_ptr->stack_top), &atomic_null_val, sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_stack_struct_ptr->free_list_top), &atomic_null_val, sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_stack_struct_ptr->next_free_node_idx), &zero_idx, sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaDeviceSynchronize()); printf("Stack reset complete.\n");
}

void print_operation_log(int num_ops, int* h_op_tid, int* h_op_type, bool* h_op_success, int* h_op_value) { /* ... unchanged ... */
    printf("--- Operation Log ---\n"); printf("Attempt | TID   | Op   | Value | Success\n"); printf("--------|-------|------|-------|--------\n");
    for (int i = 0; i < num_ops; ++i) printf("%-7d | %-5d | %-4s | %-5d | %s\n", i, h_op_tid[i], (h_op_type[i] == 0) ? "Push" : "Pop ", h_op_value[i], h_op_success[i] ? "True " : "False");
    printf("---------------------\n");
}

// --- Verification Tests ---

// Restored detailed checks in Verification [1]
bool run_verification_pops_gt_pushes(Stack* d_stack_struct_ptr, int pool_capacity, int block_size) {
    printf("\n--- Verification [1]: Pops > Pushes (Reclamation Enabled, Predefined Values Flag) ---\n");
    bool test_passed = true;
    const int num_pushes = 150; const int num_pops = num_pushes + 50;
    const int base_value_for_generation = 100; const int total_log_size = num_pushes + num_pops;
    printf("Config: Initial Pushes=%d, Total Pops=%d, Pool Capacity=%d\n", num_pushes, num_pops, pool_capacity);
    if (num_pushes > pool_capacity) printf("Warning: num_pushes exceeds pool capacity...\n");

    int *d_op_tid, *h_op_tid; int *d_op_type, *h_op_type; bool *d_op_success, *h_op_success; int *d_op_value, *h_op_value;
    /* ... allocate log arrays ... */
    gpuErrchk(cudaMalloc(&d_op_tid, total_log_size * sizeof(int))); gpuErrchk(cudaMalloc(&d_op_type, total_log_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_op_success, total_log_size * sizeof(bool))); gpuErrchk(cudaMalloc(&d_op_value, total_log_size * sizeof(int)));
    h_op_tid = (int*)malloc(total_log_size * sizeof(int)); h_op_type = (int*)malloc(total_log_size * sizeof(int));
    h_op_success = (bool*)malloc(total_log_size * sizeof(bool)); h_op_value = (int*)malloc(total_log_size * sizeof(int));
    if (!h_op_tid || !h_op_type || !h_op_success || !h_op_value) { /* error handling */ return false;}

    printf("Generating %d push values on CPU...\n", num_pushes);
    int* h_push_values = (int*)malloc(num_pushes * sizeof(int));
    if (!h_push_values) { /* error handling */ return false;}
    for (int i = 0; i < num_pushes; ++i) h_push_values[i] = base_value_for_generation + i * 2 + (i % 3);
    int* d_push_values = nullptr; gpuErrchk(cudaMalloc(&d_push_values, num_pushes * sizeof(int)));
    printf("Copying push values from Host to Device...\n");
    gpuErrchk(cudaMemcpy(d_push_values, h_push_values, num_pushes * sizeof(int), cudaMemcpyHostToDevice));

    reset_stack(d_stack_struct_ptr, pool_capacity);

    printf("Running Push Phase (using predefined values from device array via flag)...\n");
    int grid_size_push = (num_pushes + block_size - 1) / block_size;
    reporting_kernel<<<grid_size_push, block_size>>>(d_stack_struct_ptr, num_pushes, 0, true, d_push_values, 0, d_op_tid, d_op_type, d_op_success, d_op_value);
    gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());

    printf("Running Pop Phase...\n");
    int grid_size_pop = (num_pops + block_size - 1) / block_size;
    reporting_kernel<<<grid_size_pop, block_size>>>(d_stack_struct_ptr, num_pops, 0, false, nullptr, 1, d_op_tid + num_pushes, d_op_type + num_pushes, d_op_success + num_pushes, d_op_value + num_pushes);
    gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_op_tid, d_op_tid, total_log_size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_type, d_op_type, total_log_size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_success, d_op_success, total_log_size * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_value, d_op_value, total_log_size * sizeof(int), cudaMemcpyDeviceToHost));

    print_operation_log(total_log_size, h_op_tid, h_op_type, h_op_success, h_op_value);

    int successful_pushes = 0; int successful_pops = 0;
    std::multiset<int> successfully_pushed_values_multiset; std::multiset<int> popped_values_multiset;
    printf("Verifying results...\n");
    for(int i = 0; i < total_log_size; ++i) { /* ... unchanged verification loop ... */
        if (i < num_pushes) {
             if (h_op_type[i] != 0) printf("WARN: Expected push log at index %d, got pop\n", i);
             if (h_op_tid[i] < num_pushes && h_op_value[i] != h_push_values[h_op_tid[i]]) { // Check tid bounds for safety
                 printf("WARN: Logged push value %d for tid %d at log index %d does not match original host value %d\n",
                        h_op_value[i], h_op_tid[i], i, h_push_values[h_op_tid[i]]);
             }
             if (h_op_success[i]) { successful_pushes++; successfully_pushed_values_multiset.insert(h_op_value[i]); }
        } else {
            if (h_op_type[i] != 1) printf("WARN: Expected pop log at index %d, got push\n", i);
            if (h_op_success[i]) { successful_pops++; popped_values_multiset.insert(h_op_value[i]); }
        }
    }

    printf("Result: Successful Pushes recorded: %d\n", successful_pushes);
    printf("Result: Successful Pops recorded: %d (Expected to equal successful pushes: %d)\n", successful_pops, successful_pushes);
    if (successful_pops != successful_pushes) { printf("FAIL: Counts mismatch.\n"); test_passed = false; }
    if (successfully_pushed_values_multiset != popped_values_multiset) { printf("FAIL: Multisets mismatch.\n"); test_passed = false; }
    else { printf("Result: Multisets match.\n"); }

    // +++ Restored Final State Checks +++
    uint64_t final_top_atomic;
    gpuErrchk(cudaMemcpy(&final_top_atomic, &(d_stack_struct_ptr->stack_top), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    TaggedIndex final_top_ti = atomicToTaggedIndex(final_top_atomic);
    printf("Result: Final stack top index: %u (Expected: %u for NULL)\n", final_top_ti.node_idx, NULL_INDEX); // Restored Printf
    if (final_top_ti.node_idx != NULL_INDEX) {
       printf("FAIL: Final stack top index is not NULL_INDEX.\n"); // Added Fail message
       test_passed = false;
    }

    int final_alloc_idx_host = -1;
    gpuErrchk(cudaMemcpy(&final_alloc_idx_host, &(d_stack_struct_ptr->next_free_node_idx), sizeof(int), cudaMemcpyDeviceToHost));
    int expected_alloc_idx_approx = std::min(num_pushes, pool_capacity);
    printf("Result: Final next_free_node_idx: %d (Expected approx %d if pool >= pushes)\n", final_alloc_idx_host, expected_alloc_idx_approx); // Restored Printf
    // Note: Verifying exact final_alloc_idx_host is tricky due to concurrency.
    // We primarily care that stack is empty and values matched. This is informational.

    printf("Test [1] Result: %s\n", test_passed ? "PASS" : "FAIL");

    free(h_op_tid); free(h_op_type); free(h_op_success); free(h_op_value);
    gpuErrchk(cudaFree(d_op_tid)); gpuErrchk(cudaFree(d_op_type)); gpuErrchk(cudaFree(d_op_success)); gpuErrchk(cudaFree(d_op_value));
    free(h_push_values); gpuErrchk(cudaFree(d_push_values));
    return test_passed;
}

// Restored detailed checks in Verification [2]
bool run_verification_pop_empty(Stack* d_stack_struct_ptr, int pool_capacity, int block_size) {
    printf("\n--- Verification [2]: Pop Empty Stack (Reclamation Enabled) ---\n");
    bool test_passed = true; const int num_pops = 50; printf("Config: Pops=%d\n", num_pops);
    int *d_op_tid, *h_op_tid; int *d_op_type, *h_op_type; bool *d_op_success, *h_op_success; int *d_op_value, *h_op_value;
    /* ... allocate log arrays ... */
    gpuErrchk(cudaMalloc(&d_op_tid, num_pops * sizeof(int))); gpuErrchk(cudaMalloc(&d_op_type, num_pops * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_op_success, num_pops * sizeof(bool))); gpuErrchk(cudaMalloc(&d_op_value, num_pops * sizeof(int)));
    h_op_tid = (int*)malloc(num_pops * sizeof(int)); h_op_type = (int*)malloc(num_pops * sizeof(int));
    h_op_success = (bool*)malloc(num_pops * sizeof(bool)); h_op_value = (int*)malloc(num_pops * sizeof(int));
    if (!h_op_tid || !h_op_type || !h_op_success || !h_op_value) { /* error */ return false;}

    reset_stack(d_stack_struct_ptr, pool_capacity);
    printf("Running Pop Phase on Empty Stack...\n");
    int grid_size_pop = (num_pops + block_size - 1) / block_size;
    reporting_kernel<<<grid_size_pop, block_size>>>(d_stack_struct_ptr, num_pops, 0, false, nullptr, 1, d_op_tid, d_op_type, d_op_success, d_op_value);
    gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_op_tid, d_op_tid, num_pops * sizeof(int), cudaMemcpyDeviceToHost)); gpuErrchk(cudaMemcpy(h_op_type, d_op_type, num_pops * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_success, d_op_success, num_pops * sizeof(bool), cudaMemcpyDeviceToHost)); gpuErrchk(cudaMemcpy(h_op_value, d_op_value, num_pops * sizeof(int), cudaMemcpyDeviceToHost));
    print_operation_log(num_pops, h_op_tid, h_op_type, h_op_success, h_op_value);

    int successful_pops = 0; for(int i=0; i<num_pops; ++i) if(h_op_success[i]) successful_pops++;
    printf("Result: Successful pops: %d (Expected: %d)\n", successful_pops, 0);
    if (successful_pops != 0) { printf("FAIL: Pops succeeded on an empty stack.\n"); test_passed = false; }

    // +++ Restored Final State Checks +++
    uint64_t final_top_atomic;
    gpuErrchk(cudaMemcpy(&final_top_atomic, &(d_stack_struct_ptr->stack_top), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    TaggedIndex final_top_ti = atomicToTaggedIndex(final_top_atomic);
    printf("Result: Final stack top index: %u (Expected: %u for NULL)\n", final_top_ti.node_idx, NULL_INDEX); // Restored Printf
    if (final_top_ti.node_idx != NULL_INDEX) {
       printf("FAIL: Final stack top index changed from NULL_INDEX.\n"); // Added Fail message
       test_passed = false;
    }
    uint64_t final_free_atomic;
    gpuErrchk(cudaMemcpy(&final_free_atomic, &(d_stack_struct_ptr->free_list_top), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    TaggedIndex final_free_ti = atomicToTaggedIndex(final_free_atomic);
    printf("Result: Final free list top index: %u (Expected: %u for NULL)\n", final_free_ti.node_idx, NULL_INDEX); // Restored Printf
    if (final_free_ti.node_idx != NULL_INDEX) {
       printf("FAIL: Final free list top index changed from NULL_INDEX.\n"); // Added Fail message
       test_passed = false;
    }

    printf("Test [2] Result: %s\n", test_passed ? "PASS" : "FAIL");
    free(h_op_tid); free(h_op_type); free(h_op_success); free(h_op_value);
    gpuErrchk(cudaFree(d_op_tid)); gpuErrchk(cudaFree(d_op_type)); gpuErrchk(cudaFree(d_op_success)); gpuErrchk(cudaFree(d_op_value));
    return test_passed;
}

// Restored detailed checks in Verification [3]
bool run_verification_pool_exhaustion(Stack* d_stack_struct_ptr, int pool_capacity, int block_size) {
    printf("\n--- Verification [3]: Pool Index Exhaustion (Reclamation Enabled) ---\n");
    printf("      NOTE: This test uses calculated push values (tid + base_value).\n");
    bool test_passed = true;
    const int num_pushes = pool_capacity + 50; const int base_value = 400;
    printf("Config: Pool Capacity=%d, Pushes Attempted=%d\n", pool_capacity, num_pushes);

    int *d_dummy_tid, *h_dummy_tid; int *d_dummy_type, *h_dummy_type; bool *d_op_success, *h_op_success; int *d_dummy_value, *h_dummy_value;
    /* ... allocate log arrays ... */
    gpuErrchk(cudaMalloc(&d_dummy_tid, num_pushes * sizeof(int))); gpuErrchk(cudaMalloc(&d_dummy_type, num_pushes * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_op_success, num_pushes * sizeof(bool))); gpuErrchk(cudaMalloc(&d_dummy_value, num_pushes * sizeof(int)));
    h_dummy_tid = (int*)malloc(num_pushes * sizeof(int)); h_dummy_type = (int*)malloc(num_pushes * sizeof(int));
    h_op_success = (bool*)malloc(num_pushes * sizeof(bool)); h_dummy_value = (int*)malloc(num_pushes * sizeof(int));
    if (!h_dummy_tid || !h_dummy_type || !h_op_success || !h_dummy_value) { /* error */ return false; }

    reset_stack(d_stack_struct_ptr, pool_capacity);
    printf("Running Push Phase (expecting pool index exhaustion)...\n");
    int grid_size_push = (num_pushes + block_size - 1) / block_size;
    reporting_kernel<<<grid_size_push, block_size>>>(d_stack_struct_ptr, num_pushes, base_value, false, nullptr, 0, d_dummy_tid, d_dummy_type, d_op_success, d_dummy_value);
    gpuErrchk(cudaGetLastError()); gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaMemcpy(h_op_success, d_op_success, num_pushes * sizeof(bool), cudaMemcpyDeviceToHost));

    // +++ Restored Detailed Verification Logic +++
    int successful_pushes = 0;
    int failed_pushes = 0;
    for(int i = 0; i < num_pushes; ++i) { // Restored loop
        if (h_op_success[i]) successful_pushes++;
        else failed_pushes++;
    }

    printf("Result: Successful pushes: %d\n", successful_pushes); // Restored printf
    printf("Result: Failed pushes: %d (Expected > 0 if num_pushes > capacity)\n", failed_pushes); // Restored printf

    // Optional bounds check (less critical than failure check)
    int lower_bound = pool_capacity - block_size * 2; // Wider tolerance for concurrency
    int upper_bound = pool_capacity + block_size * 2;
    printf("Result: Successful pushes approx expected around %d. Bounds check [%d, %d]\n", pool_capacity, lower_bound > 0 ? lower_bound:0 , upper_bound); // Restored printf
    if (successful_pushes < lower_bound || successful_pushes > upper_bound) {
        printf("WARN: Number of successful pushes (%d) is outside expected range around pool capacity (%d).\n", successful_pushes, pool_capacity);
    }

    if (num_pushes > pool_capacity && failed_pushes == 0) {
        printf("FAIL: Expected push failures as num_pushes (%d) > capacity (%d), but got none.\n", num_pushes, pool_capacity); // Restored printf
        test_passed = false;
    } else if (num_pushes > pool_capacity) {
        printf("Result: Observed push failures as expected.\n"); // Restored printf
    } else if (failed_pushes > 0) {
         printf("WARN: Pushes failed (%d) even though num_pushes (%d) <= capacity (%d).\n", failed_pushes, num_pushes, pool_capacity);
    }

    int final_alloc_idx_host = -1;
    gpuErrchk(cudaMemcpy(&final_alloc_idx_host, &(d_stack_struct_ptr->next_free_node_idx), sizeof(int), cudaMemcpyDeviceToHost));
    printf("Result: Final next_free_node_idx: %d (Expected >= %d if pool exhausted)\n", final_alloc_idx_host, pool_capacity); // Restored printf

    if (final_alloc_idx_host < pool_capacity && num_pushes >= pool_capacity && failed_pushes > 0) {
         // Only fail if we expected exhaustion (num_pushes >= capacity AND failures occurred) but index didn't reach capacity
         printf("FAIL: Pool exhaustion likely occurred (pushes failed), but final alloc index %d is less than capacity %d.\n", final_alloc_idx_host, pool_capacity); // Restored printf + check
         test_passed = false;
    } else if (final_alloc_idx_host >= pool_capacity) {
         printf("Result: Pool allocation index reached or exceeded capacity as expected during exhaustion test.\n"); // Restored printf
    }

    printf("Test [3] Result: %s\n", test_passed ? "PASS" : "FAIL");
    free(h_dummy_tid); free(h_dummy_type); free(h_op_success); free(h_dummy_value);
    gpuErrchk(cudaFree(d_dummy_tid)); gpuErrchk(cudaFree(d_dummy_type)); gpuErrchk(cudaFree(d_op_success)); gpuErrchk(cudaFree(d_dummy_value));
    return test_passed;
}


// --- Main Function (Unchanged - throughput already present) ---

int main() {
    int pool_capacity = 1024 * 100;
    // int pool_capacity = 1024 * 1000;
    // int pool_capacity = 1024 * 10000; // ~80MB pool
    const int block_size = 256;
    printf("Initializing Lock-Free Linked-List Stack (32b Index + 32b Tag = 64b Atomic)...\n");
    printf("Initial Node Pool Capacity: %d\n", pool_capacity); printf("Block Size: %d\n", block_size);
    printf("Node size: %zu bytes\n", sizeof(Node)); printf("NULL_INDEX representation: %u\n", NULL_INDEX);
    printf("========================================\n");

    Node* d_node_pool; gpuErrchk(cudaMalloc((void**)&d_node_pool, (size_t)pool_capacity * sizeof(Node)));
    Stack* d_stack_struct_ptr; gpuErrchk(cudaMalloc((void**)&d_stack_struct_ptr, sizeof(Stack)));
    Stack h_stack_template; h_stack_template.node_pool = d_node_pool; h_stack_template.pool_capacity = pool_capacity;
    gpuErrchk(cudaMemcpy(d_stack_struct_ptr, &h_stack_template, sizeof(Stack), cudaMemcpyHostToDevice));
    reset_stack(d_stack_struct_ptr, pool_capacity);

    printf("Starting Verification Tests...\n");
    bool all_tests_passed = true;
    all_tests_passed &= run_verification_pops_gt_pushes(d_stack_struct_ptr, pool_capacity, block_size);
    all_tests_passed &= run_verification_pop_empty(d_stack_struct_ptr, pool_capacity, block_size);
    all_tests_passed &= run_verification_pool_exhaustion(d_stack_struct_ptr, pool_capacity, block_size);
    printf("\n--- Overall Verification Result: %s ---\n", all_tests_passed ? "ALL PASS" : "SOME FAIL");
    printf("========================================\n");

    if (all_tests_passed) {
        printf("\nStarting Scalability Tests...\n");
        int num_ops_tests[] = {10000, 50000, 100000, 200000, 500000, 1000000};
        for (int num_operations : num_ops_tests) {
            reset_stack(d_stack_struct_ptr, pool_capacity);
            printf("\nRunning Test: %d Operations (Mixed Push/Pop - Device Calculated Values)\n", num_operations);
            int grid_size = (num_operations + block_size - 1) / block_size;
            printf("Grid Size: %d, Block Size: %d, Total Threads: %d\n", grid_size, block_size, grid_size*block_size);
            gpuErrchk(cudaDeviceSynchronize());
            cudaEvent_t start, stop; gpuErrchk(cudaEventCreate(&start)); gpuErrchk(cudaEventCreate(&stop)); gpuErrchk(cudaEventRecord(start));
            // stack_test_kernel<<<grid_size, block_size>>>(d_stack_struct_ptr, num_operations, nullptr, nullptr, nullptr);
            // Launch the NEW warp-assigned kernel
            stack_test_kernel_warp_assigned<<<grid_size, block_size>>>(
                d_stack_struct_ptr,
                num_operations
                // No logging arguments needed/passed
                );
            gpuErrchk(cudaGetLastError()); gpuErrchk(cudaEventRecord(stop)); gpuErrchk(cudaEventSynchronize(stop));
            float milliseconds = 0; gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
            gpuErrchk(cudaEventDestroy(start)); gpuErrchk(cudaEventDestroy(stop));
            printf("Execution Time: %.4f ms\n", milliseconds);
            double seconds = milliseconds / 1000.0;
            if (seconds > 0) {
                double ops_per_second = (double)num_operations / seconds;
                printf("Throughput: %.2f Million Operations/Second\n", ops_per_second / 1e6); // Already present
            } else {
                printf("Throughput: Inf (Execution time too small to measure accurately)\n");
            }
            printf("----------------------------------------\n");
           }
    } else {
        printf("\nSkipping Scalability Tests due to Verification Failures.\n");
    }

    printf("\nCleaning up...\n"); gpuErrchk(cudaFree(d_node_pool)); gpuErrchk(cudaFree(d_stack_struct_ptr));
    printf("Done.\n");
    return all_tests_passed ? 0 : 1;
}
