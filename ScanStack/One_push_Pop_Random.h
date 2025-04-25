#ifndef ONE_PUSH_POP_RANDOM_H
#define ONE_PUSH_POP_RANDOM_H

#include "common.h"

void One_push_Pop_Random()
{
#define STACK_CAPACITY 10000 // Size of stack array
#define NUM_OPS 10000        // Number of push/pop

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

    // cout << "Initialised Stack" << endl;
    // printFromDevice<<<1, 1>>>(d_stack);

    path cwd = std::filesystem::current_path();
    // path path_insert_keys = cwd / "unique_numbers.bin";
    path path_insert_keys = cwd / "unique_numbers_1e8.bin";

    auto *tmp_keys_insert = new uint32_t[NUM_OPS];
    read_data(path_insert_keys, NUM_OPS, tmp_keys_insert);

    OpRequest *h_ops = (OpRequest *)malloc(NUM_OPS * sizeof(OpRequest));

    // cudaStream_t streams[NUM_OPS];
    // for (int i = 0; i < NUM_OPS; ++i)
    // {
    //     cudaStreamCreate(&streams[i]);
    // }

    int curr_ops_cnt = 0;
    int Push_cnt = 0;
    while (curr_ops_cnt < NUM_OPS)
    {
        random_device rd;
        mt19937 gen(rd());
        uniform_int_distribution<uint32_t> dis(0, 1);

        int isPush = dis(gen) * 2 - 1;
        // cout << "PUSHUP : " << isPush << endl;
        if (isPush == 1)
        {
            Push_cnt++;
        }

        h_ops[curr_ops_cnt].type = isPush;                                          // push or pop
        h_ops[curr_ops_cnt].value = tmp_keys_insert[curr_ops_cnt] * (isPush != -1); // some values

        OpRequest *d_ops;

        CHECK_CUDA(cudaMalloc(&d_ops, sizeof(OpRequest)));
        CHECK_CUDA(cudaMemcpy(d_ops, &h_ops[curr_ops_cnt], sizeof(OpRequest), cudaMemcpyHostToDevice));

        int threadsPerBlock = 256;
        int blocks = 1;

        int sharedMemSize = 2 * threadsPerBlock * sizeof(int); // for opType + opValue

        float memsettime;
        cudaEvent_t start, stop;
        // initialize CUDA timers
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start, 0);

        performOperations<<<blocks, threadsPerBlock, sharedMemSize>>>(d_stack, d_ops, 1);
        // performOperations<<<blocks, threadsPerBlock, sharedMemSize, streams[curr_ops_cnt]>>>(d_stack, d_ops, 1);
        // cudaStreamSynchronize(streams[curr_ops_cnt]);

        CHECK_CUDA(cudaDeviceSynchronize());

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&memsettime, start, stop); // in milliseconds
        // printf("Kernel execution time for performing %d PUSH/POP operations: %f ms\n", 1, memsettime);

        CHECK_CUDA(cudaMemcpy(&h_ops[curr_ops_cnt], d_ops, sizeof(OpRequest), cudaMemcpyDeviceToHost));

        // OpRequest ht;
        // CHECK_CUDA(cudaMemcpy(&ht, d_ops, sizeof(OpRequest), cudaMemcpyDeviceToHost));
        // cout << ht.value << endl;

        CHECK_CUDA(cudaDeviceSynchronize());
        CHECK_CUDA(cudaFree(d_ops));

        curr_ops_cnt++;
    }

    // for (int i = 0; i < NUM_OPS; ++i)
    // {
    //     cudaStreamDestroy(streams[i]);
    // }

    cout << endl
         << "********************  " << "Push Count : " << Push_cnt << " Pop Count : " << NUM_OPS - Push_cnt << "  ***********************" << endl
         << endl;

    // Print the stack after coping to host
    int *hostResults_Array_2 = (int *)malloc(STACK_CAPACITY * sizeof(int));

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
        if (h_ops[i].type == 1)
            cout << "Push(" << h_ops[i].value << ") -> "
                 << (h_ops[i].result == -2 ? "FAILED" : "OK") << "\n";

        else if (h_ops[i].type == -1)
            cout << "Pop() -> ("
                 << (h_ops[i].result == -2 ? "FAILED" : to_string(h_ops[i].result)) << ")\n";
    }

    cudaFree(d_stack_array);
    cudaFree(d_stack);
    free(h_ops);
}

#endif