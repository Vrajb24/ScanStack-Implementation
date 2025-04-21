#include <cuda_runtime.h>
#include <stdio.h>
#include <bits/stdc++.h>

__device__ const int EMPTY = -1;
__device__ const int INVALID = -2;

#define LA_GRANULARITY 1

struct ScanStack
{
    int *array;
    int capacity;

    __device__ void init()
    {
        for (int i = 0; i < capacity; ++i)
        {
            array[i] = EMPTY;
        }
    }

    __device__ int push(int val)
    {
        int startIndex = 0;
        for (int idx = LA_GRANULARITY; idx < capacity; idx += LA_GRANULARITY)
        {
            if (array[idx] == EMPTY)
            {
                startIndex = idx - LA_GRANULARITY;
                if (startIndex < 0)
                    startIndex = 0;
                break;
            }
        }
        if (startIndex == 0)
        {
            startIndex = capacity - 1;
        }

        int i = startIndex;
        bool retrace = false;
        while (true)
        {
            if (i >= capacity)
            {
                return INVALID;
            }
            int observed = atomicAdd(&array[i], 0);

            if (observed == EMPTY)
            {
                int old = atomicCAS(&array[i], EMPTY, val);
                if (old == EMPTY)
                {
                    return val;
                }
                else
                {
                    int j = i - 1;
                    while (j >= 0 && array[j] < 0)
                    {
                        j--;
                    }
                    if (j < 0)
                    {
                        j = 0;
                    }
                    i = j;
                    retrace = true;
                }
            }
            else if (observed == INVALID)
            {
                int j = i - 1;
                while (j >= 0 && array[j] < 0)
                {
                    j--;
                }
                if (j < 0)
                    j = 0;
                i = j;
                retrace = true;
            }
            i += 1;
        }
    }

    __device__ int pop()
    {
        int startIndex = capacity - 1;
        for (int idx = capacity - 1 - LA_GRANULARITY; idx >= 0; idx -= LA_GRANULARITY)
        {
            if (array[idx] == EMPTY)
            {
                startIndex = idx + LA_GRANULARITY;
                if (startIndex > capacity - 1)
                    startIndex = capacity - 1;
                break;
            }
        }
        if (startIndex < 0)
            startIndex = 0;
        int i = startIndex;
        while (true)
        {
            if (i < 0)
            {
                return INVALID;
            }
            int observed = atomicAdd(&array[i], 0);
            if (observed >= 0)
            {
                int old = atomicCAS(&array[i], observed, INVALID);
                if (old == observed)
                {
                    int poppedValue = observed;
                    return poppedValue;
                }
                else
                {
                    int j = i + 1;
                    while (j < capacity && array[j] != EMPTY)
                    {
                        j++;
                    }
                    if (j >= capacity)
                    {
                        j = capacity - 1;
                    }
                    i = j;
                }
            }
            else if (observed == INVALID)
            {
                if (i < capacity - 1 && array[i + 1] == EMPTY)
                {
                    atomicCAS(&array[i], INVALID, EMPTY);
                }
                int j = i + 1;
                while (j < capacity && array[j] != EMPTY)
                {
                    j++;
                }
                if (j >= capacity)
                {
                    j = capacity - 1;
                }
                i = j;
            }
            i -= 1;
        }
    }
};

struct OpRequest
{
    int type;
    int value;
    int result;
};

__global__ void performOperations(ScanStack *stack, OpRequest *ops, int nOps)
{
    extern __shared__ int shm[];
    int *opType = shm;
    int *opValue = &shm[blockDim.x];
    int tid = threadIdx.x;
    if (tid < nOps)
    {
        opType[tid] = ops[tid].type;
        opValue[tid] = ops[tid].value;
    }
    else
    {
        opType[tid] = 0;
        opValue[tid] = 0;
    }
    ops[tid].result = INVALID;
    __syncthreads();

    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    if (laneId == 0)
    {
        for (int k = warpId * warpSize; k < min((warpId + 1) * warpSize, blockDim.x); ++k)
        {
            if (opType[k] == 1)
            {
                for (int m = warpId * warpSize; m < min((warpId + 1) * warpSize, blockDim.x); ++m)
                {
                    if (opType[m] == -1)
                    {
                        ops[m].result = opValue[k];
                        ops[k].result = opValue[k];
                        opType[k] = 0;
                        opType[m] = 0;
                        break;
                    }
                }
            }
            if (opType[k] == 0)
            {
                continue;
            }
        }
    }
    __syncthreads();

    if (laneId == 0)
    {
        int pushIndex = -1;
        for (int k = warpId * warpSize; k < min((warpId + 1) * warpSize, blockDim.x); ++k)
        {
            if (opType[k] == 1)
            {
                pushIndex = k;
                break;
            }
        }
        if (pushIndex >= 0)
        {
            int pushVal = opValue[pushIndex];
            opType[pushIndex] = 2;
            for (int j = 0; j < blockDim.x; ++j)
            {
                if (opType[j] == -1)
                {
                    if (atomicCAS(&opType[j], -1, 0) == -1)
                    {
                        ops[j].result = pushVal;
                        ops[pushIndex].result = pushVal;
                        opType[pushIndex] = 0;
                        break;
                    }
                }
            }
            if (opType[pushIndex] == 2)
            {
                opType[pushIndex] = 1;
            }
        }
    }
    __syncthreads();

    if (tid < nOps && opType[tid] != 0)
    {
        if (opType[tid] == 1)
        {
            int value = opValue[tid];
            int pushResult = stack->push(value);
            ops[tid].result = pushResult;
        }
        else if (opType[tid] == -1)
        {
            int poppedValue = stack->pop();
            ops[tid].result = poppedValue;
        }
    }
}

#define STACK_CAPACITY 1024
#define NUM_OPS 100

__global__ void initScanStack(ScanStack *stack)
{
    for (int i = 0; i < stack->capacity; ++i)
    {
        stack->array[i] = -1;
    }
}

// void printStackFromHost(int *h_array, int size)
// {
//     for (int i = 0; i < size; i++)
//     {
//         printf("Index %d: %d\n", i, h_array[i]);
//     }
// }

int main()
{
    int *d_stack_array;
    cudaMalloc(&d_stack_array, STACK_CAPACITY * sizeof(int));

    ScanStack h_stack;
    h_stack.array = d_stack_array;
    h_stack.capacity = STACK_CAPACITY;

    ScanStack *d_stack;
    cudaMalloc(&d_stack, sizeof(ScanStack));
    cudaMemcpy(d_stack, &h_stack, sizeof(ScanStack), cudaMemcpyHostToDevice);

    initScanStack<<<1, 1>>>(d_stack);
    cudaDeviceSynchronize();

    OpRequest h_ops[NUM_OPS];
    for (int i = 0; i < NUM_OPS / 2; ++i)
    {
        h_ops[i].type = 1;
        h_ops[i].value = 100 + i;
    }
    for (int i = NUM_OPS / 2; i < NUM_OPS; ++i)
    {
        h_ops[i].type = -1;
        h_ops[i].value = 0;
    }

    OpRequest *d_ops;
    cudaMalloc(&d_ops, NUM_OPS * sizeof(OpRequest));
    cudaMemcpy(d_ops, h_ops, NUM_OPS * sizeof(OpRequest), cudaMemcpyHostToDevice);

    int threadsPerBlock = 32;
    int blocks = 1;
    int sharedMemSize = 2 * threadsPerBlock * sizeof(int);
    performOperations<<<blocks, threadsPerBlock, sharedMemSize>>>(d_stack, d_ops, NUM_OPS);
    cudaDeviceSynchronize();

    cudaMemcpy(h_ops, d_ops, NUM_OPS * sizeof(OpRequest), cudaMemcpyDeviceToHost);

    std::cout << "Results of push and pop operations:\n";
    for (int i = 0; i < NUM_OPS; ++i)
    {
        if (h_ops[i].type == 1)
            std::cout << "Push(" << h_ops[i].value << ") → "
                      << (h_ops[i].result == -2 ? "FAILED" : "OK") << "\n";
        else if (h_ops[i].type == -1)
            std::cout << "Pop() → "
                      << (h_ops[i].result == -2 ? "FAILED" : std::to_string(h_ops[i].result)) << "\n";
    }

    cudaFree(d_stack_array);
    cudaFree(d_stack);
    cudaFree(d_ops);

    return 0;
}