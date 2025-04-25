#ifndef PUSH_POP_SEPERATELY_H
#define PUSH_POP_SEPERATELY_H

#include "common.h"

void push_Pop_Seperately()
{
    // #define STACK_CAPACITY 1024 // Size of stack array
    // #define NUM_OPS 512         // Number of push/pop

    // #define STACK_CAPACITY 262144 // Size of stack array
    // #define NUM_OPS 262144        // Number of push/pop

#define STACK_CAPACITY 10000 // Size of stack array
#define NUM_OPS 10000        // Number of push/pop

    // 1. Allocate device memory for stack array
    int *d_stack_array;
    cudaMalloc(&d_stack_array, STACK_CAPACITY * sizeof(int));

    // Create and initialize ScanStack struct on host
    ScanStack h_stack;
    h_stack.array = d_stack_array;
    h_stack.capacity = STACK_CAPACITY;

    // Allocate device memory for ScanStack struct
    ScanStack *d_stack;
    cudaMalloc(&d_stack, sizeof(ScanStack));
    cudaMemcpy(d_stack, &h_stack, sizeof(ScanStack), cudaMemcpyHostToDevice);

    // Initialize stack (set all cells to EMPTY)
    initScanStack<<<1, 1>>>(d_stack);
    cudaDeviceSynchronize();

    // printFromDevice<<<1, 1>>>(d_stack);

    int NUM_PUSH = 0.6 * NUM_OPS;
    int NUM_POP = NUM_OPS - NUM_PUSH;
    path cwd = std::filesystem::current_path();
    // path path_insert_keys = cwd / "unique_numbers.bin";
    path path_insert_keys = cwd / "unique_numbers_1e8.bin";

    auto *tmp_keys_insert = new uint32_t[NUM_PUSH];
    read_data(path_insert_keys, NUM_PUSH, tmp_keys_insert);

    OpRequest *h_ops = (OpRequest *)malloc(NUM_OPS * sizeof(OpRequest));

    for (int i = 0; i < NUM_PUSH; ++i)
    {
        h_ops[i].type = 1; // push
        // h_ops[i].value = 100 + i;
        h_ops[i].value = tmp_keys_insert[i];
    }

    OpRequest *d_ops_push;
    cudaMalloc(&d_ops_push, NUM_PUSH * sizeof(OpRequest));
    cudaMemcpy(d_ops_push, h_ops, NUM_PUSH * sizeof(OpRequest), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (NUM_PUSH + threadsPerBlock - 1) / threadsPerBlock;

    cout << "No of blocks for push : " << blocks << endl;

    int sharedMemSize = 2 * threadsPerBlock * sizeof(int); // for opType + opValue

    float memsettime;
    cudaEvent_t start, stop;
    // initialize CUDA timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    performOperations<<<blocks, threadsPerBlock, sharedMemSize>>>(d_stack, d_ops_push, NUM_PUSH);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop); // in milliseconds
    printf("Kernel execution time for performing %d PUSH operations: %f ms\n", NUM_PUSH, memsettime);

    for (int i = NUM_PUSH; i < NUM_OPS; ++i)
    {
        h_ops[i].type = -1;
        h_ops[i].value = 0;
    }

    OpRequest *d_ops_pop;
    cudaMalloc(&d_ops_pop, NUM_POP * sizeof(OpRequest));
    cudaMemcpy(d_ops_pop, &h_ops[NUM_PUSH], NUM_POP * sizeof(OpRequest), cudaMemcpyHostToDevice);

    threadsPerBlock = 256;
    blocks = (NUM_POP + threadsPerBlock - 1) / threadsPerBlock;

    cout << "No of blocks for pop : " << blocks << endl;

    sharedMemSize = 2 * threadsPerBlock * sizeof(int); // for opType + opValue

    // initialize CUDA timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    performOperations<<<blocks, threadsPerBlock, sharedMemSize>>>(d_stack, d_ops_pop, NUM_POP);

    cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop); // in milliseconds
    printf("Kernel execution time for performing %d POP operations: %f ms\n", NUM_POP, memsettime);

    // Print the stack after coping to host
    int *hostResults_Array_2 = (int *)malloc(STACK_CAPACITY * sizeof(int)); // Allocate memory for the host array

    ScanStack hostResults_Stack_2;
    cudaMemcpy(&hostResults_Stack_2, d_stack, sizeof(ScanStack), cudaMemcpyDeviceToHost);

    cudaMemcpy(hostResults_Array_2, hostResults_Stack_2.array, STACK_CAPACITY * sizeof(int), cudaMemcpyDeviceToHost);

    printf("Stack after Operations\n");
    for (int i = 0; i < STACK_CAPACITY; i++)
    {
        printf("Index %d: %d\n", i, hostResults_Array_2[i]);
    }
    cudaDeviceSynchronize();

    // Print opertion result after copying to host
    cudaMemcpy(h_ops, d_ops_push, NUM_PUSH * sizeof(OpRequest), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_ops[NUM_PUSH], d_ops_pop, NUM_POP * sizeof(OpRequest), cudaMemcpyDeviceToHost);

    cout << "Results of push and pop operations:\n";
    for (int i = 0; i < NUM_OPS; ++i)
    {
        if (h_ops[i].type == 1)
            cout << "Push(" << h_ops[i].value << ") -> "
                 << (h_ops[i].result == -2 ? "FAILED" : "OK") << "\n";

        else if (h_ops[i].type == -1)
            cout << "Pop() -> ("
                 << (h_ops[i].result == -2 ? "FAILED" : to_string(h_ops[i].result)) << ")\n";
    }

    cudaFree(d_stack_array);
    cudaFree(d_stack);
    cudaFree(d_ops_push);
    cudaFree(d_ops_pop);
    free(h_ops);
}

#endif