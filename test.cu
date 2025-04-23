#include <cuda_runtime.h>
#include <stdio.h>
#include <bits/stdc++.h>
#include <chrono>
#include <limits.h>

using namespace std::chrono;

using HR = high_resolution_clock;
using HRTimer = HR::time_point;

// --- Elimination-Backoff Definitions ---
#define EMPTY_ELIM      -1
#define INVALID_ELIM    -2
#define ELIM_ARRAY_SIZE 64      // must be >= max threads per block * blocks
#define MAX_BACKOFF     128     // max spin for backoff

enum OpType { OP_NONE=0, OP_PUSH=1, OP_POP=-1 };

struct ThreadInfo {
    int id;      // thread index
    int op;      // OP_PUSH or OP_POP
    int value;   // push value or unused for pop
    unsigned int spin;
};

// Elimination arrays in global memory
__device__ ThreadInfo d_location[ELIM_ARRAY_SIZE];
__device__ int        d_collision[ELIM_ARRAY_SIZE];

// Initialize elimination arrays
__global__ void initElimination() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ELIM_ARRAY_SIZE) {
        d_collision[idx] = EMPTY_ELIM;
        d_location[idx].id = EMPTY_ELIM;
    }
}

// Counters for instrumentation
__device__ int d_central_stack_hits = 0;
__device__ int d_backoff_push_eliminated = 0;
__device__ int d_backoff_pop_eliminated = 0;
__device__ int d_block_eliminated = 0;

__host__ void checkStats(int numOps) {
    int centralHits = 0, backoffPush = 0, backoffPop = 0, blockElim = 0;
    cudaMemcpyFromSymbol(&centralHits, d_central_stack_hits, sizeof(int));
    cudaMemcpyFromSymbol(&backoffPush, d_backoff_push_eliminated, sizeof(int));
    cudaMemcpyFromSymbol(&backoffPop, d_backoff_pop_eliminated, sizeof(int));
    cudaMemcpyFromSymbol(&blockElim, d_block_eliminated, sizeof(int));

    int totalHandled = centralHits + 2 * blockElim + backoffPush + backoffPop;
    printf("\n--- Elimination Stats ---\n");
    printf("Central Stack Ops: %d\n", centralHits);
    printf("Warp/Block Elim Pairs: %d\n", blockElim);
    printf("Backoff Elim Pushes: %d\n", backoffPush);
    printf("Backoff Elim Pops  : %d\n", backoffPop);
    printf("Total Ops Accounted: %d (expected: %d)\n", totalHandled, numOps);

    assert(backoffPush == backoffPop && "Push and pop eliminations in backoff must match");
    assert(totalHandled == numOps && "Sum of all handled ops must equal total ops");
}

// Simple pseudorandom position selector
__device__ int getElimPos(int tid) {
    // use low bits of clock for randomness
    return (tid + (clock() & 0xFFF)) % ELIM_ARRAY_SIZE;
}

// Elimination-backoff wrapper
template<typename Stack>
__device__ int eliminateOrStackOp(Stack *stack, int op, int val, int tid) {
    ThreadInfo ti = { tid, op, val, 1u };
    int res;

    // 1) Try central stack first
    if (ti.op == OP_PUSH) {
        res = stack->push(ti.value);
        if (res != INT_MIN) {
            atomicAdd(&d_central_stack_hits, 1);
            return res;
        } 
    } else {
        res = stack->pop();
        if (res != INT_MIN) {
            atomicAdd(&d_central_stack_hits, 1);
            return res;
        }
    }

    // 2) Elimination-backoff loop
    while (true) {
        // publish our intent
        d_location[tid] = ti;
        // pick a random slot
        int pos = getElimPos(tid);
        // attempt to meet a partner
        int him = atomicExch(&d_collision[pos], tid);
        if (him != EMPTY_ELIM) {
            ThreadInfo other = d_location[him];
            // valid partner of opposite type?
            if (other.id == him && other.op == -ti.op) {
                // try collision
                if (ti.op == OP_PUSH) {
                    // overwrite partner's location slot to indicate done
                    if (atomicCAS(&d_location[him].id, other.id, ti.id) == other.id) {
                        atomicAdd(&d_backoff_push_eliminated, 1);
                        atomicAdd(&d_backoff_pop_eliminated, 1);
                        return ti.value;
                    }
                } else {
                    // pop: claim partner's value
                    if (atomicCAS(&d_location[him].id, other.id, EMPTY_ELIM) == other.id) {
                        atomicAdd(&d_backoff_push_eliminated, 1);
                        atomicAdd(&d_backoff_pop_eliminated, 1);
                        return other.value;
                    }
                }
            }
        }
        // backoff delay
        for (unsigned i = 0; i < ti.spin; ++i) __nanosleep(1);
        ti.spin = min(ti.spin << 1, (unsigned)MAX_BACKOFF);
        // retry central stack
        if (ti.op == OP_PUSH) {
            res = stack->push(ti.value);
            if (res != INT_MIN) {
                atomicAdd(&d_central_stack_hits, 1);
                return res;
            } 
        } else {
            res = stack->pop();
            if (res != INT_MIN) {
                atomicAdd(&d_central_stack_hits, 1);
                return res;
            }
        }
    }
}

// --- Original ScanStack and OpRequest definitions ---
__device__ const int EMPTY = -1;
__device__ const int INVALID = -2;
#define LA_GRANULARITY 1

struct ScanStack {
    int *array;
    int  capacity;

    __device__ void init() {
        for (int i = 0; i < capacity; ++i)
            array[i] = EMPTY;
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
        // while (true)
        int retries =0;
        const int MAX_RETRIES = 10;

        while(retries++ < MAX_RETRIES)
        {
            if(retries == MAX_RETRIES) {
                return INT_MIN;
            }
              

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
        // while (true)
        int retries =0;
        const int MAX_RETRIES = 10;

        while(retries++ < MAX_RETRIES)
        {
            if(retries == MAX_RETRIES) {
                return INT_MIN;
            }
              

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

struct OpRequest { int type; int value; int result; };

// --- Kernel performing operations with elimination-backoff ---
__global__ void performOperations(ScanStack *stack, OpRequest *ops, int nOps) {
    extern __shared__ int shm[];
    int *opType  = shm;
    int *opValue = &shm[blockDim.x];

    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < nOps) {
        opType[threadIdx.x]  = ops[tid].type;
        opValue[threadIdx.x] = ops[tid].value;
    } else {
        opType[threadIdx.x]  = OP_NONE;
        opValue[threadIdx.x] = 0;
    }
    ops[tid].result = INVALID;
    __syncthreads();

    // [warp-level matching code remains unchanged]
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    if (laneId == 0) {
        for (int k = warpId * warpSize; k < min((warpId+1)*warpSize, blockDim.x); ++k) {
            if (opType[k] == OP_PUSH) {
                for (int m = warpId*warpSize; m < min((warpId+1)*warpSize, blockDim.x); ++m) {
                    if (opType[m] == OP_POP) {
                        ops[m].result    = opValue[k];
                        ops[k].result    = opValue[k];
                        atomicAdd(&d_block_eliminated, 1);
                        opType[k] = OP_NONE;
                        opType[m] = OP_NONE;
                        break;
                    }
                }
            }
        }
    }
    __syncthreads();

    if (laneId == 0) {
        int pushIdx = -1;
        for (int k = warpId*warpSize; k < min((warpId+1)*warpSize, blockDim.x); ++k)
            if (opType[k] == OP_PUSH) { pushIdx=k; break; }
        if (pushIdx >= 0) {
            int pv = opValue[pushIdx];
            opType[pushIdx] = 2; // in-elimination
            for (int j=0; j<blockDim.x; ++j) {
                if (opType[j]==OP_POP) {
                    if (atomicCAS(&opType[j], OP_POP, OP_NONE)==OP_POP) {
                        ops[j].result    = pv;
                        ops[pushIdx].result = pv;
                        atomicAdd(&d_block_eliminated, 1);
                        opType[pushIdx] = OP_NONE;
                        break;
                    }
                }
            }
            if (opType[pushIdx]==2) opType[pushIdx] = OP_PUSH;
        }
    }
    __syncthreads();

    // Final: any remaining operations go through elimination/backoff
    if (tid < nOps && opType[threadIdx.x] != OP_NONE) {
        int type = opType[threadIdx.x];
        int value = opValue[threadIdx.x];
        int result = eliminateOrStackOp(stack, type, value, tid);
        ops[tid].result = result;
    }
}

#define STACK_CAPACITY 1024
#define NUM_OPS 100

__global__ void initScanStack(ScanStack *stack) {
    for (int i = 0; i < stack->capacity; ++i)
        stack->array[i] = EMPTY;
}

int main() {
    // Allocate and initialize stack
    int *d_stack_array;
    cudaMalloc(&d_stack_array, STACK_CAPACITY * sizeof(int));
    ScanStack h_stack = { d_stack_array, STACK_CAPACITY };
    ScanStack *d_stack;
    cudaMalloc(&d_stack, sizeof(ScanStack));
    cudaMemcpy(d_stack, &h_stack, sizeof(ScanStack), cudaMemcpyHostToDevice);
    initScanStack<<<1,1>>>(d_stack);
    cudaDeviceSynchronize();

    // Initialize elimination arrays
    initElimination<<<(ELIM_ARRAY_SIZE+31)/32,32>>>();
    cudaDeviceSynchronize();

    // Prepare operations
    OpRequest h_ops[NUM_OPS];
    for (int i = 0; i < NUM_OPS/2; ++i) { h_ops[i].type=OP_PUSH; h_ops[i].value=100+i; }
    for (int i = NUM_OPS/2; i < NUM_OPS; ++i) { h_ops[i].type=OP_POP;  h_ops[i].value=0; }
    OpRequest *d_ops;
    cudaMalloc(&d_ops, NUM_OPS * sizeof(OpRequest));
    cudaMemcpy(d_ops, h_ops, NUM_OPS * sizeof(OpRequest), cudaMemcpyHostToDevice);

    // Launch
    int threadsPerBlock = 32;
    int blocks = 1;
    int sharedMemSize = 2 * threadsPerBlock * sizeof(int);
    
    // HRTimer start = HR::now();
    // start = HR::now();
    performOperations<<<blocks, threadsPerBlock, sharedMemSize>>>(d_stack, d_ops, NUM_OPS);
    cudaDeviceSynchronize();
    // HRTimer end = HR::now();
    // auto duration1 = duration_cast<microseconds>(end - start).count();
    // std::cout << "Time with opt (us): " << duration1 << "\n";

    // Copy back and print
    cudaMemcpy(h_ops, d_ops, NUM_OPS * sizeof(OpRequest), cudaMemcpyDeviceToHost);
    std::cout << "Results of push and pop operations:\n";
    
    // for (int i = 0; i < NUM_OPS; ++i) {
    //     if (h_ops[i].type == OP_PUSH)
    //         std::cout << "Push("<<h_ops[i].value<<") → " << (h_ops[i].result==INVALID? "FAILED" : "OK") << "\n";
    //     else
    //         std::cout << "Pop() → " << (h_ops[i].result==INVALID? "FAILED" : std::to_string(h_ops[i].result)) << "\n";
    // }

    checkStats(blocks*threadsPerBlock);

    cudaFree(d_stack_array);
    cudaFree(d_stack);
    cudaFree(d_ops);
    return 0;
}
