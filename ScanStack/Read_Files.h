#ifndef READ_FILES_H
#define READ_FILES_H

#include "common.h"

void Read_Files()
{
#define STACK_CAPACITY 1000 // Size of stack array
#define NUM_OPS 1000        // Number of push/pop

    // Allocate device memory for stack array
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

    path cwd = std::filesystem::current_path();
    path path_insert_keys = cwd / "unique_numbers_1e8.bin";

    auto *tmp_keys_insert = new uint32_t[NUM_PUSH];
    read_data(path_insert_keys, NUM_PUSH, tmp_keys_insert);

    OpRequest *h_ops = (OpRequest *)malloc(NUM_OPS * sizeof(OpRequest));

    for (int i = 0; i < NUM_PUSH; ++i)
    {
        h_ops[i].type = 1;
        h_ops[i].value = tmp_keys_insert[i];
    }
    for (int i = NUM_PUSH; i < NUM_OPS; ++i)
    {
        h_ops[i].type = -1; // pop
        h_ops[i].value = 0; // unused for pop
    }

    // Allocate device memory for operations
    OpRequest *d_ops;
    cudaMalloc(&d_ops, NUM_OPS * sizeof(OpRequest));
    cudaMemcpy(d_ops, h_ops, NUM_OPS * sizeof(OpRequest), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (NUM_OPS + threadsPerBlock - 1) / threadsPerBlock;

    cout << "Blocks : " << blocks << endl;

    int sharedMemSize = 2 * threadsPerBlock * sizeof(int); // for opType + opValue

    float memsettime;
    cudaEvent_t start, stop;
    // initialize CUDA timers
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    auto start_time = chrono::high_resolution_clock::now();

    cudaEventRecord(start, 0);

    performOperations<<<blocks, threadsPerBlock, sharedMemSize>>>(d_stack, d_ops, NUM_OPS);
    cudaDeviceSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&memsettime, start, stop); // in milliseconds
    printf("Kernel execution time for performing %d operations: %f ms\n", NUM_OPS, memsettime);

    auto end_time = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(end_time - start_time).count();

    cout << "Kernel time (host measured): " << duration << " milliseconds" << endl;

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

    // Print opertion result after copying to host
    cudaMemcpy(h_ops, d_ops, NUM_OPS * sizeof(OpRequest), cudaMemcpyDeviceToHost);

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
    cudaFree(d_ops);
}

#endif