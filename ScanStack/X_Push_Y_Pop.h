#ifndef X_PUSH_Y_POP_H
#define X_PUSH_Y_POP_H

#include "common.h"

void X_Push_Y_Pop()
{
#define STACK_CAPACITY 10000 // Size of stack array
#define NUM_OPS 1000         // Number of push/pop

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

    path cwd = std::filesystem::current_path();
    // path path_insert_keys = cwd / "unique_numbers.bin";
    path path_insert_keys = cwd / "unique_numbers_1e8.bin";

    auto *tmp_keys_insert = new uint32_t[NUM_OPS];
    read_data(path_insert_keys, NUM_OPS, tmp_keys_insert);

    OpRequest *h_ops_output = (OpRequest *)malloc(NUM_OPS * sizeof(OpRequest));

    int curr_ops_cnt = 0;
    int total_push = 0;
    while (curr_ops_cnt < NUM_OPS)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<uint32_t> dis_push(0, (NUM_OPS - curr_ops_cnt));

        int NUM_PUSH = dis_push(gen);

        uniform_int_distribution<uint32_t> dis_pop(0, (NUM_OPS - curr_ops_cnt - NUM_PUSH));
        int NUM_POP = dis_pop(gen);

        total_push += NUM_PUSH;
        // cout << NUM_PUSH << " : " << NUM_POP << endl;

        OpRequest *h_ops = (OpRequest *)malloc((NUM_POP + NUM_PUSH) * sizeof(OpRequest));

        for (int i = 0; i < NUM_PUSH; ++i)
        {
            h_ops[i].type = 1; // push
            // h_ops[i].value = 100 + i;
            h_ops[i].value = tmp_keys_insert[i];
        }
        for (int i = NUM_PUSH; i < (NUM_POP + NUM_PUSH); ++i)
        {
            h_ops[i].type = -1; // pop
            h_ops[i].value = 0; // unused for pop
        }

        // Allocate device memory for operations
        OpRequest *d_ops;
        CHECK_CUDA(cudaMalloc(&d_ops, (NUM_POP + NUM_PUSH) * sizeof(OpRequest)));
        CHECK_CUDA(cudaMemcpy(d_ops, h_ops, (NUM_POP + NUM_PUSH) * sizeof(OpRequest), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocks = ((NUM_POP + NUM_PUSH) + threadsPerBlock - 1) / threadsPerBlock;

        int sharedMemSize = 2 * threadsPerBlock * sizeof(int); // for opType + opValue

        float memsettime;
        cudaEvent_t start, stop;
        // initialize CUDA timers
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        performOperations<<<blocks, threadsPerBlock, sharedMemSize>>>(d_stack, d_ops, (NUM_POP + NUM_PUSH));
        cudaDeviceSynchronize();

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&memsettime, start, stop); // in milliseconds
        printf("Kernel execution time for performing %d PUSH and %d POP operations: %f ms\n", NUM_PUSH, NUM_POP, memsettime);

        CHECK_CUDA(cudaMemcpy(&h_ops_output[curr_ops_cnt], d_ops, (NUM_POP + NUM_PUSH) * sizeof(OpRequest), cudaMemcpyDeviceToHost));

        curr_ops_cnt += NUM_PUSH + NUM_POP;
        CHECK_CUDA(cudaFree(d_ops));
    }

    cout << "Total_no_of_PUSH " << total_push << endl;
    cout << "Total_no_of_POP " << NUM_OPS - total_push << endl;

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
    cout << "Results of push and pop operations:\n";
    for (int i = 0; i < NUM_OPS; ++i)
    {
        if (h_ops_output[i].type == 1)
            cout << "Push(" << h_ops_output[i].value << ") -> "
                 << (h_ops_output[i].result == -2 ? "FAILED" : "OK") << "\n";

        else if (h_ops_output[i].type == -1)
            cout << "Pop() -> ("
                 << (h_ops_output[i].result == -2 ? "FAILED" : to_string(h_ops_output[i].result)) << ")\n";
    }

    cudaFree(d_stack_array);
    cudaFree(d_stack);
    free(h_ops_output);
}

#endif