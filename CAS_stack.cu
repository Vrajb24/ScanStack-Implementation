#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#include <vector>
#include <numeric>
#include <algorithm>
#include <set>
#include <cmath>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

typedef struct {
    int* stack_data;
    int32_t stack_top;
    int stack_capacity;
} Stack;


__device__ bool push_gpu(Stack* stack, int value) {
    int32_t old_top;
    int32_t new_top;

    while (true) {
        old_top = atomicAdd(&stack->stack_top, 0);

        new_top = old_top + 1;

        if (new_top >= stack->stack_capacity) {
            return false;
        }

        int32_t assumed_old_top = atomicCAS((int32_t*)&stack->stack_top, old_top, new_top);

        if (assumed_old_top == old_top) {
            stack->stack_data[new_top] = value;
            return true;
        }
    }
}

__device__ bool pop_gpu(Stack* stack, int* result) {
    int32_t old_top;
    int32_t new_top;

    while (true) {
        old_top = atomicAdd(&stack->stack_top, 0);

        if (old_top < 0) {
            return false;
        }

        new_top = old_top - 1;

        int32_t assumed_old_top = atomicCAS((int32_t*)&stack->stack_top, old_top, new_top);

        if (assumed_old_top == old_top) {
            *result = stack->stack_data[old_top];
            return true;
        }
    }
}

__device__ bool peek_gpu(Stack* stack, int* result) {
    int32_t current_top = atomicAdd(&stack->stack_top, 0);

    if (current_top < 0) {
        return false;
    }

    *result = stack->stack_data[current_top];
    return true;
}


__global__ void stack_test_kernel(Stack* d_stack, int num_ops_total, bool* d_push_results, bool* d_pop_results, int* d_pop_values) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_ops_total) {
        if (tid % 2 == 0) {
            int value_to_push = tid;
            bool success = push_gpu(d_stack, value_to_push);
            if (d_push_results) d_push_results[tid / 2] = success;
        } else {
            int popped_value = -1;
            bool success = pop_gpu(d_stack, &popped_value);
            if (d_pop_results) d_pop_results[tid / 2] = success;
            if (d_pop_values && success) d_pop_values[tid / 2] = popped_value;
        }
    }
}


__global__ void reporting_kernel(Stack* d_stack, int num_ops, int base_value, int op_mode,
                                 int* d_op_tid, int* d_op_type, bool* d_op_success, int* d_op_value)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_ops) {
        d_op_tid[tid] = tid;
        d_op_type[tid] = op_mode;

        if (op_mode == 0) {
            int value_to_push = tid + base_value;
            bool success = push_gpu(d_stack, value_to_push);
            d_op_success[tid] = success;
            d_op_value[tid] = value_to_push;
        } else {
            int popped_value = -1;
            bool success = pop_gpu(d_stack, &popped_value);
            d_op_success[tid] = success;
            d_op_value[tid] = success ? popped_value : -1;
        }
    }
}

__global__ void push_check_success_kernel(Stack* d_stack, int num_pushes, int base_value, bool* d_push_results) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < num_pushes) {
        int value_to_push = tid + base_value;
        bool success = push_gpu(d_stack, value_to_push);
        if(d_push_results) d_push_results[tid] = success;
    }
}



void reset_stack(Stack* d_stack_struct_ptr, int* d_stack_data, int stack_capacity, bool clear_data = true) {
    int initial_top = -1;
    gpuErrchk(cudaMemcpy(&(d_stack_struct_ptr->stack_top), &initial_top, sizeof(int32_t), cudaMemcpyHostToDevice));
    if (clear_data) {
        gpuErrchk(cudaMemset(d_stack_data, 0, stack_capacity * sizeof(int)));
    }
    gpuErrchk(cudaDeviceSynchronize());
}

void print_operation_log(int num_ops, int* h_op_tid, int* h_op_type, bool* h_op_success, int* h_op_value) {
    printf("--- Operation Log ---\n");
    printf("Attempt | TID   | Op   | Value | Success\n");
    printf("--------|-------|------|-------|--------\n");
    for (int i = 0; i < num_ops; ++i) {
        printf("%-7d | %-5d | %-4s | %-5d | %s\n",
               i,
               h_op_tid[i],
               (h_op_type[i] == 0) ? "Push" : "Pop",
               h_op_value[i],
               h_op_success[i] ? "True" : "False");
    }
    printf("---------------------\n");
}



bool run_verification_pops_gt_pushes(Stack* d_stack_struct_ptr, int* d_stack_data, int stack_capacity, int block_size) {
    printf("\n--- Verification [1]: Pops > Pushes ---\n");
    bool test_passed = true;
    const int num_pushes = 10;
    const int num_pops = num_pushes + 5;
    const int base_value = 100;
    const int total_log_size = num_pushes + num_pops;
    printf("Config: Pushes=%d, Pops=%d, Capacity=%d\n", num_pushes, num_pops, stack_capacity);

    if (num_pushes > stack_capacity) {
        printf("Warning: num_pushes exceeds stack capacity for this test.\n");
    }

    int *d_op_tid, *h_op_tid;
    int *d_op_type, *h_op_type;
    bool *d_op_success, *h_op_success;
    int *d_op_value, *h_op_value;

    gpuErrchk(cudaMalloc(&d_op_tid, total_log_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_op_type, total_log_size * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_op_success, total_log_size * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_op_value, total_log_size * sizeof(int)));
    h_op_tid = (int*)malloc(total_log_size * sizeof(int));
    h_op_type = (int*)malloc(total_log_size * sizeof(int));
    h_op_success = (bool*)malloc(total_log_size * sizeof(bool));
    h_op_value = (int*)malloc(total_log_size * sizeof(int));

    reset_stack(d_stack_struct_ptr, d_stack_data, stack_capacity);

    printf("Running Push Phase...\n");
    int grid_size_push = (num_pushes + block_size - 1) / block_size;
    reporting_kernel<<<grid_size_push, block_size>>>(d_stack_struct_ptr, num_pushes, base_value, 0,
                                                    d_op_tid, d_op_type, d_op_success, d_op_value);
    gpuErrchk(cudaDeviceSynchronize());

    printf("Running Pop Phase...\n");
    int grid_size_pop = (num_pops + block_size - 1) / block_size;
    reporting_kernel<<<grid_size_pop, block_size>>>(d_stack_struct_ptr, num_pops, base_value, 1,
                                                   d_op_tid + num_pushes, d_op_type + num_pushes,
                                                   d_op_success + num_pushes, d_op_value + num_pushes);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_op_tid, d_op_tid, total_log_size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_type, d_op_type, total_log_size * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_success, d_op_success, total_log_size * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_value, d_op_value, total_log_size * sizeof(int), cudaMemcpyDeviceToHost));

    print_operation_log(total_log_size, h_op_tid, h_op_type, h_op_success, h_op_value);

    int successful_pushes = 0;
    int successful_pops = 0;
    std::set<int> pushed_values_set;
    std::set<int> popped_values_set;

    for(int i = 0; i < total_log_size; ++i) {
        if (h_op_type[i] == 0) {
            if (h_op_success[i]) {
                successful_pushes++;
                pushed_values_set.insert(h_op_value[i]);
            }
        } else {
            if (h_op_success[i]) {
                successful_pops++;
                popped_values_set.insert(h_op_value[i]);
            }
        }
    }

    printf("Result: Successful Pushes: %d\n", successful_pushes);
    printf("Result: Successful Pops: %d (Expected <= %d)\n", successful_pops, successful_pushes);
    if (successful_pops > successful_pushes) {
        printf("FAIL: More pops succeeded than pushes!\n");
        test_passed = false;
    }
     if (successful_pops != successful_pushes) {
        printf("INFO: Number of successful pops (%d) doesn't exactly match successful pushes (%d). This might be ok if pushes failed or due to races.\n", successful_pops, successful_pushes);
     }


    int32_t final_top;
    gpuErrchk(cudaMemcpy(&final_top, &(d_stack_struct_ptr->stack_top), sizeof(int32_t), cudaMemcpyDeviceToHost));
    printf("Result: Final stack top: %d (Expected: -1)\n", final_top);
     if (final_top != -1) {
        printf("FAIL: Final stack top is not -1.\n");
        test_passed = false;
    }

    if (pushed_values_set != popped_values_set) {
         printf("FAIL: Set of popped values does not match set of successfully pushed values.\n");
         test_passed = false;
    }

    printf("Test Result: %s\n", test_passed ? "PASS" : "FAIL");

    free(h_op_tid); free(h_op_type); free(h_op_success); free(h_op_value);
    gpuErrchk(cudaFree(d_op_tid)); gpuErrchk(cudaFree(d_op_type)); gpuErrchk(cudaFree(d_op_success)); gpuErrchk(cudaFree(d_op_value));
    return test_passed;
}


bool run_verification_pop_empty(Stack* d_stack_struct_ptr, int* d_stack_data, int stack_capacity, int block_size) {
    printf("\n--- Verification [2]: Pop Empty Stack ---\n");
    bool test_passed = true;
    const int num_pops = 10;
    printf("Config: Pops=%d\n", num_pops);

    int *d_op_tid, *h_op_tid;
    int *d_op_type, *h_op_type;
    bool *d_op_success, *h_op_success;
    int *d_op_value, *h_op_value;

    gpuErrchk(cudaMalloc(&d_op_tid, num_pops * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_op_type, num_pops * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_op_success, num_pops * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_op_value, num_pops * sizeof(int)));
    h_op_tid = (int*)malloc(num_pops * sizeof(int));
    h_op_type = (int*)malloc(num_pops * sizeof(int));
    h_op_success = (bool*)malloc(num_pops * sizeof(bool));
    h_op_value = (int*)malloc(num_pops * sizeof(int));

    reset_stack(d_stack_struct_ptr, d_stack_data, stack_capacity);

    printf("Running Pop Phase...\n");
    int grid_size_pop = (num_pops + block_size - 1) / block_size;
    reporting_kernel<<<grid_size_pop, block_size>>>(d_stack_struct_ptr, num_pops, 0, 1,
                                                   d_op_tid, d_op_type, d_op_success, d_op_value);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_op_tid, d_op_tid, num_pops * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_type, d_op_type, num_pops * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_success, d_op_success, num_pops * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_op_value, d_op_value, num_pops * sizeof(int), cudaMemcpyDeviceToHost));

    print_operation_log(num_pops, h_op_tid, h_op_type, h_op_success, h_op_value);

    int successful_pops = 0;
    for(int i = 0; i < num_pops; ++i) {
        if (h_op_success[i]) successful_pops++;
    }
    printf("Result: Successful pops: %d (Expected: %d)\n", successful_pops, 0);
    if (successful_pops != 0) {
        printf("FAIL: Pops succeeded on an empty stack.\n");
        test_passed = false;
    }

    int32_t final_top;
    gpuErrchk(cudaMemcpy(&final_top, &(d_stack_struct_ptr->stack_top), sizeof(int32_t), cudaMemcpyDeviceToHost));
    printf("Result: Final stack top: %d (Expected: %d)\n", final_top, -1);
     if (final_top != -1) {
        printf("FAIL: Final stack top changed.\n");
        test_passed = false;
    }

    printf("Test Result: %s\n", test_passed ? "PASS" : "FAIL");

    free(h_op_tid); free(h_op_type); free(h_op_success); free(h_op_value);
    gpuErrchk(cudaFree(d_op_tid)); gpuErrchk(cudaFree(d_op_type)); gpuErrchk(cudaFree(d_op_success)); gpuErrchk(cudaFree(d_op_value));
    return test_passed;
}

bool run_verification_overflow(Stack* d_stack_struct_ptr, int* d_stack_data, int stack_capacity, int block_size) {
    printf("\n--- Verification [3]: Stack Overflow ---\n");
    bool test_passed = true;
    const int test_capacity = 50;
    const int num_pushes = test_capacity + 10;
    const int base_value = 400;
    printf("Config: Test Capacity (Simulated)=%d, Pushes=%d\n", test_capacity, num_pushes);

    Stack h_stack_orig, h_stack_test;
    gpuErrchk(cudaMemcpy(&h_stack_orig, d_stack_struct_ptr, sizeof(Stack), cudaMemcpyDeviceToHost));
    h_stack_test = h_stack_orig;
    h_stack_test.stack_capacity = test_capacity;
    gpuErrchk(cudaMemcpy(d_stack_struct_ptr, &h_stack_test, sizeof(Stack), cudaMemcpyHostToDevice));

    bool *d_push_results, *h_push_results;
    gpuErrchk(cudaMalloc(&d_push_results, num_pushes * sizeof(bool)));
    h_push_results = (bool*)malloc(num_pushes * sizeof(bool));

    reset_stack(d_stack_struct_ptr, d_stack_data, test_capacity, true);

    printf("Running Push Phase...\n");
    int grid_size_push = (num_pushes + block_size - 1) / block_size;
    push_check_success_kernel<<<grid_size_push, block_size>>>(d_stack_struct_ptr, num_pushes, base_value, d_push_results);
    gpuErrchk(cudaDeviceSynchronize());

    gpuErrchk(cudaMemcpy(h_push_results, d_push_results, num_pushes * sizeof(bool), cudaMemcpyDeviceToHost));

    int successful_pushes = 0;
    for(int i = 0; i < num_pushes; ++i) {
        if (h_push_results[i]) successful_pushes++;
    }
    printf("Result: Successful pushes: %d (Expected: %d)\n", successful_pushes, test_capacity);
    if (successful_pushes != test_capacity) {
        printf("FAIL: Incorrect number of successful pushes during overflow.\n");
        test_passed = false;
    }

    int32_t final_top;
    gpuErrchk(cudaMemcpy(&final_top, &(d_stack_struct_ptr->stack_top), sizeof(int32_t), cudaMemcpyDeviceToHost));
    printf("Result: Final stack top: %d (Expected: %d)\n", final_top, test_capacity - 1);
     if (final_top != test_capacity - 1) {
        printf("FAIL: Final stack top is not capacity-1.\n");
        test_passed = false;
    }

    printf("Test Result: %s\n", test_passed ? "PASS" : "FAIL");

    free(h_push_results);
    gpuErrchk(cudaFree(d_push_results));
    gpuErrchk(cudaMemcpy(d_stack_struct_ptr, &h_stack_orig, sizeof(Stack), cudaMemcpyHostToDevice));
    reset_stack(d_stack_struct_ptr, d_stack_data, stack_capacity, true);

    return test_passed;
}

int main() {
    int stack_capacity = 1024 * 100;
    const int block_size = 256;

    printf("Initializing Parallel Stack...\n");
    printf("Stack Capacity: %d\n", stack_capacity);
    printf("Block Size: %d\n", block_size);
    printf("========================================\n");

    int* d_stack_data;
    Stack* d_stack_struct_ptr;
    Stack h_stack;
    gpuErrchk(cudaMalloc((void**)&d_stack_data, stack_capacity * sizeof(int)));
    gpuErrchk(cudaMalloc((void**)&d_stack_struct_ptr, sizeof(Stack)));
    h_stack.stack_data = d_stack_data;
    h_stack.stack_top = -1;
    h_stack.stack_capacity = stack_capacity;
    gpuErrchk(cudaMemcpy(d_stack_struct_ptr, &h_stack, sizeof(Stack), cudaMemcpyHostToDevice));

    printf("Starting Verification Tests...\n");
    bool all_tests_passed = true;


    all_tests_passed &= run_verification_pops_gt_pushes(d_stack_struct_ptr, d_stack_data, stack_capacity, block_size);
    all_tests_passed &= run_verification_pop_empty(d_stack_struct_ptr, d_stack_data, stack_capacity, block_size);
    all_tests_passed &= run_verification_overflow(d_stack_struct_ptr, d_stack_data, stack_capacity, block_size);


    printf("\n--- Overall Verification Result: %s ---\n", all_tests_passed ? "ALL PASS" : "SOME FAIL");
    printf("========================================\n");


    if (all_tests_passed) {
        printf("\nStarting Scalability Tests...\n");
        int num_ops_tests[] = {1000, 10000, 100000, 500000, 1000000, 2000000};

        for (int i = 0; i < sizeof(num_ops_tests) / sizeof(num_ops_tests[0]); ++i) {
            int num_operations = num_ops_tests[i];
            if (num_operations > stack_capacity * 10) {
                 printf("Skipping scalability test with %d operations: too large relative to capacity.\n", num_operations);
                 continue;
            }

            reset_stack(d_stack_struct_ptr, d_stack_data, stack_capacity);

            printf("\nRunning Test: %d Operations (Mixed Push/Pop)\n", num_operations);
            int grid_size = (num_operations + block_size - 1) / block_size;
            printf("Grid Size: %d\n", grid_size);

            cudaEvent_t start, stop;
            gpuErrchk(cudaEventCreate(&start)); gpuErrchk(cudaEventCreate(&stop));
            gpuErrchk(cudaEventRecord(start));
            stack_test_kernel<<<grid_size, block_size>>>(d_stack_struct_ptr, num_operations, nullptr, nullptr, nullptr);
            gpuErrchk(cudaEventRecord(stop));
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaEventSynchronize(stop));
            float milliseconds = 0;
            gpuErrchk(cudaEventElapsedTime(&milliseconds, start, stop));
            gpuErrchk(cudaEventDestroy(start)); gpuErrchk(cudaEventDestroy(stop));

            printf("Execution Time: %.4f ms\n", milliseconds);
            double seconds = milliseconds / 1000.0;
            if (seconds > 0) {
                double ops_per_second = (double)num_operations / seconds;
                printf("Throughput: %.2f Million Operations/Second\n", ops_per_second / 1e6);
            } else { printf("Throughput: Inf\n"); }
            printf("----------------------------------------");
        }
    } else {
         printf("\nSkipping Scalability Tests due to Verification Failures.\n");
    }

    printf("\nCleaning up...\n");
    gpuErrchk(cudaFree(d_stack_data));
    gpuErrchk(cudaFree(d_stack_struct_ptr));
    printf("Done.\n");
    return all_tests_passed ? 0 : 1;
}