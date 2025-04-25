#ifndef COMMON_H
#define COMMON_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <bits/stdc++.h>

using namespace std;

using std::filesystem::path;

#define CHECK_CUDA(call)                                                                                 \
    do                                                                                                   \
    {                                                                                                    \
        cudaError_t status = call;                                                                       \
        if (status != cudaSuccess)                                                                       \
        {                                                                                                \
            fprintf(stderr, "CUDA Error (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            exit(1);                                                                                     \
        }                                                                                                \
    } while (0)

void read_data(path pth, uint64_t n, uint32_t *data)
{
    FILE *fptr = fopen(pth.string().c_str(), "rb");
    string fname = pth.string();
    if (!fptr)
    {
        string error_msg = "Unable to open file: " + fname;
        perror(error_msg.c_str());
    }
    int freadStatus = fread(data, sizeof(uint32_t), n, fptr);
    if (freadStatus == 0)
    {
        string error_string = "Unable to read the file " + fname;
        perror(error_string.c_str());
    }
    fclose(fptr);
}

__device__ const int EMPTY = -1;   // Empty cell
__device__ const int INVALID = -2; // Invalidated (popped) cell

// #define LA_GRANULARITY 256

struct OpRequest
{
    int type;   // 1 for push, -1 for pop, 0 for none or opeartion already executed
    int value;  // value to push (unused/any_value for pop operations)
    int result; // result of operation (for pop: the popped value or INVALID(-2) if failed(Satck Underflow); for push: pushed value or INVALID(-2) if failed(stack Overflow))
};

struct ScanStack
{
    int *array;
    int capacity;

    // Returns the value pushed on success, or INVALID (-2) if the stack is full (push failure)
    __device__ int push(int val)
    {
        // int LA_GRANULARITY = 256;
        int startIndex = -1;

        int low = 0, high = capacity - 1, mid;

        // Get lower-bound
        while (low <= high)
        {
            mid = low + (high - low) / 2;

            if (array[mid] == EMPTY)
            {
                high = mid - 1;
                startIndex = mid;
            }
            else
            {
                low = mid + 1;
            }
        }

        if (startIndex == -1)
        {
            // return INVALID;
            startIndex = capacity - 1;
        }

        int i = startIndex;
        while (true)
        {
            // Bounds check: if we scanned beyond the end, the stack is full and push fails -> Stack Overflow.
            if (i >= capacity)
            {
                return INVALID;
            }

            // int observed = atomicAdd(&array[i], 0);
            int observed = array[i];

            // Found an empty cell
            if (observed == EMPTY)
            {
                int old = atomicCAS(&array[i], EMPTY, val);
                if (old == EMPTY) // Now array[i] contains the value 'val' and index i is the new top
                {
                    return val; // Return the pushed value
                }
                else // CAS failed-> another thread beat us to this index (contention) - Can be push or pop (pop will leave -2)
                {
                    // Contention handling -> step downward (Assuming pop has happended)
                    int j = i - 1;
                    // Move downwards until we find a non-empty cell i.e having a real valid value -> This will determine the top index of the stack. Or, current size of the stack
                    while (j >= 0 && array[j] < 0) // Keep going downward while stack cell value is empty or invalid
                    {
                        // Skip over any empty (-1) or invalid (-2) cells while moving toward bottom
                        j--;
                    }
                    if (j < 0)
                    {
                        // Reached below index 0: no filled cell found below. This likely means the stack was empty
                        // We reset to start scanning from index 0.
                        j = 0;
                    }
                    // Resume scanning upward from the position of the filled cell we found (j)
                    i = j;
                    // After this, the loop will increment i (see below) to j+1 and continue scanning upward again.
                }
            }
            else if (observed == INVALID)
            {
                // Found an invalidated cell (a cell that was popped but not yet cleared)
                // This indicates the stack has shrunk (popped) below this point after our last observation
                // Handle like contention -> retrace downward below this invalid cell.
                int j = i - 1;
                while (j >= 0 && array[j] < 0)
                {
                    j--;
                }
                if (j < 0)
                    j = 0;
                i = j;
            }
            // Continue scanning upward towards the top of stack.
            i += 1; // Move to the next index upward.
        }
    }

    // Device function for pop operation: pops the top value off the stack.
    // Returns the popped value on success, or INVALID (-2) if the stack is empty (pop failure)
    __device__ int pop()
    {

        // int LA_GRANULARITY = 256;
        int startIndex = -1;
        int low = 0, high = capacity - 1, mid;

        while (low <= high)
        {
            mid = low + (high - low) / 2;

            if (array[mid] > EMPTY)
            {
                low = mid + 1;
                startIndex = mid;
            }
            else
            { // initialize the stack

                high = mid - 1;
            }
        }

        startIndex += 256;

        if (startIndex < 0)
        {
            // if (startIndex >= capacity)
            startIndex = 0; // If none found (stack may be empty), start from bottom (index 0).
        }

        int i = startIndex;
        while (true)
        {
            // If we moved below index 0, stack is empty
            if (i < 0)
            {
                return INVALID; // Stack is empty, pop fails (return -2 indicating falure)
            }

            int observed = array[i];

            if (observed > 0) // If a valid cell value
            {
                // Attempt to pop it and set the cell value to INVALID
                int old = atomicCAS(&array[i], observed, INVALID);
                if (old == observed)
                {
                    // CAS succeeded -> we extracted the value and left an INVALID marker in its place
                    // If this was not actually the top element due to contention, an invalid marker remains
                    // We will rely on subsequent operations to clear invalid markers when safe (see below)
                    return observed; // Return the popped value
                }
                else
                {
                    // CAS failed: another thread modified this cell (contention)
                    // Possibly another pop already took it or a push has happened
                    // Contention handling -> step upward and retrace -> Assuming a push/pop has happened
                    int j = i + 1;
                    // Move upwards (toward index n-1) until we find an empty cell (meaning we've passed the new top)
                    while (j < capacity && array[j] != EMPTY)
                    {
                        j++;
                    }
                    if (j >= capacity)
                    {
                        j = capacity - 1;
                    }
                    // Resume scanning downward (see below)
                    i = j;
                }
            }
            else if (observed == INVALID)
            {
                // Found an invalidated cell (a value was popped by another thread and marked INVALID)
                // Check if the cell above (higher index) is empty, meaning this invalid is may be the current top of stack.
                if (i < capacity - 1 && array[i + 1] == EMPTY)
                {
                    // The invalid cell is directly below an empty cell (top), so we can safely clear it to EMPTY
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
            // observed == EMPTY
            // This cell is empty, meaning we are still above the actual top. Continue scanning downward.
            i -= 1; // Move to the next index downward.
        }
    }
};

__global__ void performOperations(ScanStack *stack, OpRequest *ops, int nOps)
{
    extern __shared__ int shm[];     // Shared memory -> we'll partition it for op types and values
    int *opType = shm;               // [blockDim.x] array for operation type (1=push, -1=pop, 0=none)
    int *opValue = &shm[blockDim.x]; // [blockDim.x] array for operation data (push value or unused for pop)
    // Each thread copies its operation into shared memory

    int local_tid = threadIdx.x;

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < nOps && local_tid < blockDim.x)
    {
        opType[local_tid] = ops[tid].type;
        opValue[local_tid] = ops[tid].value;
    }
    else
        return; // Exit threads beyond nOps

    __syncthreads();

    // Warp-level elimination: each warp's proxy thread scans its warp for push-pop pairs
    int warpId = tid / warpSize;
    int laneId = tid % warpSize;
    // The first thread in each warp (lane 0) act as the proxy for that warp
    if (laneId == 0) //
    {
        // Iterate over all threads in this warp to eliminate pairs within the warp
        for (int k = warpId * warpSize; k < min((warpId + 1) * warpSize, blockDim.x); ++k)
        {
            if (opType[k] == 1)
            {
                // Found a push operation
                // Look for a pop in the same warp to eliminate with
                for (int m = warpId * warpSize; m < min((warpId + 1) * warpSize, blockDim.x); ++m)
                {
                    if (opType[m] == -1)
                    {
                        ops[m].result = opValue[k]; // pop gets the push's value directly
                        ops[k].result = opValue[k]; // gets it's opValue -> Means push successfull
                        opType[k] = 0;              // mark push as eliminated
                        opType[m] = 0;              // Mark pop as eliminated
                        break;
                    }
                }
            }
            if (opType[k] == 0)
            {
                // If this operation got eliminated, skip further checking it.
                continue;
            }
        }
    }
    __syncthreads();

    // Block - level elimination  : use block - wide proxy threads to eliminate remaining ops across warps We will let push operations drive the block - level elimination : each warp's proxy will attempt to match one leftover push with a pop from another warp.
    bool cleared = 0;
    if (laneId == 0)
    {
        while (!cleared)
        {
            // Find one leftover push in this warp (if any)
            int pushIndex = -1;
            for (int k = warpId * warpSize; k < min((warpId + 1) * warpSize, blockDim.x); ++k)
            {
                if (opType[k] == 1)
                {
                    pushIndex = k;
                    break;
                }
                if (k == min((warpId + 1) * warpSize, blockDim.x))
                {
                    cleared = 1;
                }
            }
            if (pushIndex >= 0)
            {
                // Find a pop in any warp to eliminate with this push
                int pushVal = opValue[pushIndex];
                // Mark this push as temporarily invalid (to avoid others taking it) by setting type to 2.
                opType[pushIndex] = INVALID;
                for (int j = 0; j < blockDim.x; ++j)
                {
                    if (opType[j] == -1)
                    {
                        // Found a pop
                        if (atomicCAS(&opType[j], EMPTY, 0) == -1)
                        {
                            // We popped array[j] for elimination.
                            ops[j].result = pushVal;         // pop gets value
                            ops[pushIndex].result = pushVal; // mark push as successful
                            opType[pushIndex] = 0;           // Mark the push as eliminated
                            break;
                        }
                    }
                }
                // If no pop found
                if (opType[pushIndex] == INVALID)
                {
                    opType[pushIndex] = 1; // no pop found, set it to push as it was earlier
                }
            }
        }
    }
    __syncthreads();

    // After elimination, remaining operations (opType still 1 or -1) will access the global stack
    if (tid < nOps && opType[local_tid] != 0 && local_tid < blockDim.x)
    {
        if (opType[local_tid] == 1)
        {
            // Perform actual push on global stack for this thread
            int value = opValue[local_tid];
            int pushResult = stack->push(value);
            // If pushResult is INVALID (-2), it means the stack was full
            ops[tid].result = pushResult;
        }
        else if (opType[local_tid] == -1)
        {
            // Perform actual pop on global stack
            int poppedValue = stack->pop();
            ops[tid].result = poppedValue;
        }
    }
}

__global__ void initScanStack(ScanStack *stack)
{
    for (int i = 0; i < stack->capacity; ++i)
    {
        stack->array[i] = -1;
    }
}

__global__ void printFromDevice(ScanStack *stack)
{
    for (int i = 0; i < stack->capacity; ++i)
    {
        printf("Index %d: %d\n", i, stack->array[i]);
    }
}

void printStackFromHost(int *h_array, int size)
{
    for (int i = 0; i < size; i++)
    {
        printf("Index %d: %d\n", i, h_array[i]);
    }
}
#endif