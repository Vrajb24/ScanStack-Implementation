#include <cuda_runtime.h>
#include <cuda/atomic> // For CUDA C++ atomic operations
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <unistd.h>
#include <limits.h> // For UINT32_MAX

#include <vector>
#include <numeric>
#include <algorithm>
#include <set>
#include <cmath>
#include <cstddef>
#include <cstdlib> // For host malloc/free

// --- Constants ---

// Represents an invalid or null index in the node pool.
#define INVALID_NODE_INDEX UINT32_MAX
// Standard warp size for NVIDIA GPUs (used in one kernel optimization)
#define WARP_SIZE 32

// --- Core Data Structures ---

/**
 * @brief Represents a single node in the linked list used for the stack.
 * Resides in the device's node pool.
 */
typedef struct StackNode {
    int data;               // The actual data stored in this node.
    uint32_t nextNodeIndex; // Index of the next node in the list (main stack or free list).
                           // Uses INVALID_NODE_INDEX if it's the last node.
} StackNode;

/**
 * @brief Combines a node index with a tag. Used for atomic operations on list heads.
 * The tag helps mitigate the ABA problem in lock-free algorithms.
 * When a node is popped and later pushed back (to the free list), its tag is incremented.
 * This prevents a thread from incorrectly succeeding a compare-and-swap if the head node
 * index is the same, but it's actually a *different incarnation* of that node index being used.
 */
typedef struct {
    uint32_t nodeIndex; // Index into the deviceNodePool array.
    uint32_t tag;       // ABA counter/tag. Incremented when a node is reused.
} NodePointerWithTag;

/**
 * @brief Main structure managing the concurrent stack on the GPU device.
 * An instance of this struct exists on the host and is copied to the device.
 * It contains pointers to device memory and atomic variables for managing the stack state.
 */
typedef struct ConcurrentDeviceStack {
    StackNode* deviceNodePool;             // Pointer to the pre-allocated array of StackNodes on the GPU.
    int        nodePoolCapacity;           // Maximum number of nodes the pool can hold.
    cuda::atomic<int, cuda::thread_scope_device> deviceNodeAllocatorIndex; // Atomically incremented index to allocate *new* nodes from the pool
                                                                            // when the free list is empty.
    cuda::atomic<uint64_t, cuda::thread_scope_device> deviceStackTopAtomic; // Atomic variable storing the packed NodePointerWithTag for the
                                                                            // current top of the main stack (LIFO).
    cuda::atomic<uint64_t, cuda::thread_scope_device> deviceFreeListTopAtomic; // Atomic variable storing the packed NodePointerWithTag for the
                                                                                // head of the free list (recycled nodes).
} ConcurrentDeviceStack;


// --- Helper Functions (Device-side) ---

/**
 * @brief Error checking macro for CUDA runtime API calls.
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
}

/**
 * @brief Packs a NodePointerWithTag into a 64-bit unsigned integer for atomic operations.
 * The tag occupies the high 32 bits, and the index occupies the low 32 bits.
 * @param pointerTag The NodePointerWithTag to pack.
 * @return The packed 64-bit representation.
 */
__host__ __device__ inline uint64_t packNodePointerWithTag(NodePointerWithTag pointerTag) {
    // Bitwise OR combines the shifted tag and the index.
    return ((uint64_t)pointerTag.tag << 32) | (uint64_t)pointerTag.nodeIndex;
}

/**
 * @brief Unpacks a 64-bit unsigned integer back into a NodePointerWithTag.
 * Reverses the operation of packNodePointerWithTag.
 * @param packedValue The 64-bit value retrieved from an atomic variable.
 * @return The unpacked NodePointerWithTag.
 */
__host__ __device__ inline NodePointerWithTag unpackNodePointerWithTag(uint64_t packedValue) {
    NodePointerWithTag pointerTag;
    // Mask to get the lower 32 bits (index).
    pointerTag.nodeIndex = (uint32_t)(packedValue & 0xFFFFFFFFULL);
    // Right-shift to get the upper 32 bits (tag).
    pointerTag.tag       = (uint32_t)(packedValue >> 32);
    return pointerTag;
}


/**
 * @brief Converts a node index into a direct pointer to the StackNode within the device pool.
 * Performs bounds checking.
 * @param stack A pointer to the main ConcurrentDeviceStack structure (on the device).
 * @param index The index of the node within the deviceNodePool.
 * @return A device pointer to the StackNode, or nullptr if the index is invalid or out of bounds.
 */
__device__ inline StackNode* getNodePointerFromIndex_device(ConcurrentDeviceStack* stack, uint32_t index) {
    if (index == INVALID_NODE_INDEX) return nullptr;
    // Check if index is within the allocated pool capacity.
    if (index >= (uint32_t)stack->nodePoolCapacity) return nullptr;
    return &stack->deviceNodePool[index];
}


/**
 * @brief Atomically tries to pop a node from the head of a tagged linked list (main stack or free list).
 * Uses atomic compare-and-swap (CAS) in a loop to handle concurrency.
 *
 * @param atomicTaggedListHead Pointer to the atomic 64-bit integer representing the list head (e.g., stack->deviceStackTopAtomic).
 * @param out_poppedNodePointer Pointer to store the NodePointerWithTag of the node that was successfully popped.
 * @param stack Pointer to the main ConcurrentDeviceStack structure (needed for indexToNodePtr).
 * @return true if a node was successfully popped, false if the list was empty or contention prevented the pop temporarily.
 */
__device__ bool atomicPopNodePointerWithTag(
    cuda::atomic<uint64_t, cuda::thread_scope_device>* atomicTaggedListHead,
    NodePointerWithTag* out_poppedNodePointer,
    ConcurrentDeviceStack* stack)
{
    // Load the current head atomically. Acquire semantics ensure memory operations before this load are visible.
    uint64_t currentHeadPacked = atomicTaggedListHead->load(cuda::memory_order_acquire);
    NodePointerWithTag currentHead;
    NodePointerWithTag nextHead;

    // Loop until CAS succeeds or the list is found empty.
    do {
        // Unpack the loaded value.
        currentHead = unpackNodePointerWithTag(currentHeadPacked);

        // If the list is empty (head index is invalid), return false.
        if (currentHead.nodeIndex == INVALID_NODE_INDEX) {
            return false;
        }

        // Get a pointer to the actual node struct for the current head.
        StackNode* currentHeadNodePtr = getNodePointerFromIndex_device(stack, currentHead.nodeIndex);

        // Defensive check: If the pointer is somehow invalid (e.g., index out of bounds, though unlikely if head was valid),
        // reload the head and retry the loop. This might happen under extreme race conditions or if memory gets corrupted.
        if (!currentHeadNodePtr) {
             currentHeadPacked = atomicTaggedListHead->load(cuda::memory_order_acquire); // Re-load and retry
             continue;
        }

        // Prepare the value for the *new* head: the 'next' pointer of the current head.
        nextHead.nodeIndex = currentHeadNodePtr->nextNodeIndex;
        // The tag stays the same for the *list* head pointer itself during a pop. The popped node's tag might change later if reused.
        nextHead.tag = currentHead.tag; // This seems incorrect based on how ABA is typically handled - tag should likely increment ON PUSH. Let's keep original logic for now, but flag it. **Revisiting:** The tag *on the list head* doesn't need to change here. The tag is associated with the *node* being pointed to. When this popped node is pushed to the *free list*, *that* push operation will use an incremented tag. So this is correct.

        // Attempt the atomic Compare-and-Swap.
        // - Compare `atomicTaggedListHead`'s current value with `currentHeadPacked`.
        // - If they match, atomically update `atomicTaggedListHead` to `packNodePointerWithTag(nextHead)`.
        // - `memory_order_release` ensures writes before this CAS are visible to threads that acquire this value later.
        // - `memory_order_acquire` ensures reads after a successful CAS see writes from the thread that set this value.
        // - compare_exchange_weak is used because it can spuriously fail even if values match (on some architectures),
        //   requiring the loop. It can be more performant than compare_exchange_strong in contention.
    } while (!atomicTaggedListHead->compare_exchange_weak(
                 currentHeadPacked,                      // Expected current value
                 packNodePointerWithTag(nextHead),       // New value if exchange succeeds
                 cuda::memory_order_release,             // Memory order for success
                 cuda::memory_order_acquire              // Memory order for failure (updates expected `currentHeadPacked`)
             ));

    // If CAS succeeded, 'currentHeadPacked' holds the value that was popped. Unpack it.
    *out_poppedNodePointer = unpackNodePointerWithTag(currentHeadPacked); // Original value before CAS successful update
    return true;
}

/**
 * @brief Atomically pushes a node onto the head of a tagged linked list (main stack or free list).
 * Uses atomic compare-and-swap (CAS) in a loop to handle concurrency.
 *
 * @param atomicTaggedListHead Pointer to the atomic 64-bit integer representing the list head (e.g., stack->deviceStackTopAtomic).
 * @param nodeIndexToPush The index of the node (from the pool) to be pushed onto the list.
 * @param newTagForNode The tag to associate with this node *when it becomes the head*. This is crucial for the free list to increment tags.
 * @param stack Pointer to the main ConcurrentDeviceStack structure (needed for indexToNodePtr).
 */
__device__ void atomicPushNodePointerWithTag(
    cuda::atomic<uint64_t, cuda::thread_scope_device>* atomicTaggedListHead,
    uint32_t nodeIndexToPush,
    uint32_t newTagForNode, // Tag specifically for the node being pushed
    ConcurrentDeviceStack* stack)
{
    // Load the current head. Relaxed is often okay for the initial read in a CAS loop,
    // as the CAS itself provides synchronization. Acquire might be safer if depending on prior state.
    uint64_t currentHeadPacked = atomicTaggedListHead->load(cuda::memory_order_relaxed);
    NodePointerWithTag currentHead;
    NodePointerWithTag newHead; // The node we are trying to make the new head

    // Prepare the new head information.
    newHead.nodeIndex = nodeIndexToPush;
    newHead.tag = newTagForNode; // Use the provided tag for this node

    // Get a pointer to the node we intend to push.
    StackNode* nodeToPushPtr = getNodePointerFromIndex_device(stack, nodeIndexToPush);

    // Basic validation: If the index doesn't yield a valid pointer, we can't proceed.
    if (!nodeToPushPtr) {
        // Optionally print an error or handle this case.
        // printf("ERROR: atomicPushNodePointerWithTag received invalid node index %u\n", nodeIndexToPush);
        return;
    }

    // Loop until CAS succeeds.
    do {
        // Unpack the value read from the atomic head variable.
        currentHead = unpackNodePointerWithTag(currentHeadPacked);

        // Link the node-to-be-pushed to the *current* head of the list.
        nodeToPushPtr->nextNodeIndex = currentHead.nodeIndex;

        // Attempt the atomic Compare-and-Swap.
        // Tries to replace `currentHeadPacked` with `packNodePointerWithTag(newHead)`.
        // If `atomicTaggedListHead` has changed since we loaded `currentHeadPacked`, the CAS fails,
        // `currentHeadPacked` is updated with the *new* actual head value, and the loop repeats.
    } while (!atomicTaggedListHead->compare_exchange_weak(
                 currentHeadPacked,                 // Expected current value
                 packNodePointerWithTag(newHead),   // New value (our node)
                 cuda::memory_order_release,        // Order on success
                 cuda::memory_order_acquire         // Order on failure (updates expected `currentHeadPacked`)
             ));
     // Note: In the original push, the new_top.tag was set to old_top.tag + 1. This seems wrong.
     // The tag should be associated with the NODE being pushed, especially for the free list.
     // The tag is incremented when moving from stack -> free list, so the tag passed in (`newTagForNode`)
     // should reflect this. Let's stick to the provided parameter `newTagForNode`.
     // --> Updated loop to use `newTagForNode` parameter.
}


// --- Core Stack Operations (Device-side) ---

/**
 * @brief Pushes a data value onto the concurrent device stack.
 * It first tries to reuse a node from the free list. If the free list is empty,
 * it allocates a new node from the node pool using the atomic allocator index.
 *
 * @param stack Pointer to the ConcurrentDeviceStack structure on the device.
 * @param dataValue The integer value to push onto the stack.
 * @return true if the push was successful, false if the node pool was full and no node could be obtained.
 */
__device__ bool pushOntoDeviceStack(ConcurrentDeviceStack* stack, int dataValue) {
    uint32_t nodeIndexToUse = INVALID_NODE_INDEX;
    uint32_t nodeTagToUse = 0; // Initialize tag for the node we'll use
    NodePointerWithTag reusedNodePointer;

    // 1. Try to get a node from the free list (reuse)
    if (atomicPopNodePointerWithTag(&stack->deviceFreeListTopAtomic, &reusedNodePointer, stack)) {
        // Success! We got a node from the free list.
        nodeIndexToUse = reusedNodePointer.nodeIndex;
        // IMPORTANT: The tag associated with this node in the free list is `reusedNodePointer.tag`.
        // We will use *this specific tag* when pushing it onto the main stack.
        nodeTagToUse = reusedNodePointer.tag;
    } else {
        // 2. Free list was empty, try to allocate a new node from the pool.
        // Atomically increment the allocator index to claim a slot. Relaxed order is usually sufficient for counters.
        int allocatedIndex = stack->deviceNodeAllocatorIndex.fetch_add(1, cuda::memory_order_relaxed);

        // Check if the allocated index is within the pool bounds.
        if (allocatedIndex >= stack->nodePoolCapacity) {
            // Pool is full. We failed to get a node. Decrement the counter back (best effort).
             stack->deviceNodeAllocatorIndex.fetch_sub(1, cuda::memory_order_relaxed); // Try to correct counter
            // This push operation fails.
            return false;
        }
        // We successfully allocated a new node index.
        nodeIndexToUse = (uint32_t)allocatedIndex;
        // New nodes start with tag 0.
        nodeTagToUse = 0;
    }

    // 3. We have a node index (either reused or new), get the pointer.
    StackNode* nodeToUsePtr = getNodePointerFromIndex_device(stack, nodeIndexToUse);

    // Safety check: Ensure the node pointer is valid.
    if (!nodeToUsePtr) {
        // This indicates a potential logic error or memory issue if index was thought to be valid.
        // Depending on strategy, could try to push the node back to free list if it came from there.
        // For now, just fail the push.
        return false;
    }

    // 4. Prepare the node and push it onto the main stack.
    nodeToUsePtr->data = dataValue; // Store the user's data.
    // Push onto the main stack using the obtained index and its associated tag.
    atomicPushNodePointerWithTag(&stack->deviceStackTopAtomic, nodeIndexToUse, nodeTagToUse, stack);

    return true; // Push successful
}

/**
 * @brief Pops a data value from the concurrent device stack.
 * The popped node is then pushed onto the free list for potential reuse,
 * with its tag incremented to help prevent the ABA problem.
 *
 * @param stack Pointer to the ConcurrentDeviceStack structure on the device.
 * @param out_dataValue Pointer to an integer where the popped data value will be stored.
 * @return true if a value was successfully popped, false if the stack was empty.
 */
__device__ bool popFromDeviceStack(ConcurrentDeviceStack* stack, int* out_dataValue) {
    NodePointerWithTag poppedNodePointer;

    // 1. Try to pop a node from the main stack.
    if (!atomicPopNodePointerWithTag(&stack->deviceStackTopAtomic, &poppedNodePointer, stack)) {
        // Main stack is empty.
        return false;
    }

    // 2. Get the index and tag of the node that was just popped.
    uint32_t poppedNodeIndex = poppedNodePointer.nodeIndex;
    uint32_t currentTag = poppedNodePointer.tag; // The tag this node had *while on the main stack*.

    // 3. Get a pointer to the popped node structure.
    StackNode* poppedNodePtr = getNodePointerFromIndex_device(stack, poppedNodeIndex);

    // Safety check (should ideally not fail if pop succeeded, but good practice).
    if (!poppedNodePtr) {
        // Error condition: Pop returned a valid index, but we can't get the pointer.
        // The node cannot be pushed to free list, effectively lost. Indicate failure.
        // Maybe log an error if possible in a real scenario.
        return false;
    }

    // 4. Retrieve the data from the node.
    *out_dataValue = poppedNodePtr->data;

    // 5. Prepare the node for the free list: Increment its tag.
    // This signifies a new "version" or "incarnation" of this node index.
    uint32_t nextTag = currentTag + 1;

    // 6. Push the node (now considered 'free') onto the free list using its original index but the *incremented* tag.
    atomicPushNodePointerWithTag(&stack->deviceFreeListTopAtomic, poppedNodeIndex, nextTag, stack);

    return true; // Pop successful
}

/**
 * @brief Peeks at the data value of the top element of the stack without removing it.
 * This is implemented using a read-read-validate approach to handle potential races
 * where the top might change between reading the index/tag and reading the data.
 *
 * @param stack Pointer to the ConcurrentDeviceStack structure on the device.
 * @param out_dataValue Pointer to an integer where the peeked data value will be stored.
 * @return true if the stack was not empty and the peek was successful, false otherwise.
 */
__device__ bool peekDeviceStackTop(ConcurrentDeviceStack* stack, int* out_dataValue) {
    uint64_t currentTopPacked_Phase1; // Value read in first phase
    NodePointerWithTag currentTop;
    int dataRead;

    // Loop to handle potential races.
    while (true) {
        // Phase 1: Read the current top pointer (index + tag). Acquire ensures subsequent reads aren't reordered before this.
        currentTopPacked_Phase1 = stack->deviceStackTopAtomic.load(cuda::memory_order_acquire);
        currentTop = unpackNodePointerWithTag(currentTopPacked_Phase1);

        // If stack is empty, cannot peek.
        if (currentTop.nodeIndex == INVALID_NODE_INDEX) {
            return false;
        }

        // Get pointer to the top node.
        StackNode* currentTopPtr = getNodePointerFromIndex_device(stack, currentTop.nodeIndex);

        // If pointer is invalid (e.g., index out of bounds, unlikely if index != NULL), retry the loop.
        if (!currentTopPtr) {
            continue; // Retry reading the top
        }

        // Read the data from the node.
        dataRead = currentTopPtr->data;

        // Ensure that the read of 'data' is not reordered before the second read of the top pointer.
        // An acquire fence ensures that memory operations after the fence are not moved before it.
        // Crucially, it synchronizes with release operations (like the CAS in push/pop).
        cuda::atomic_thread_fence(cuda::memory_order_acquire); // <<< Crucial Fence

        // Phase 2: Read the top pointer *again*. Relaxed is okay here, we just need to compare bits.
        uint64_t currentTopPacked_Phase2 = stack->deviceStackTopAtomic.load(cuda::memory_order_relaxed);

        // Validation: Check if the top pointer (index and tag) is *exactly* the same as it was in Phase 1.
        if (currentTopPacked_Phase1 == currentTopPacked_Phase2) {
            // Success! The top didn't change between reading the pointer and reading the data.
            // The acquire fence ensures that `dataRead` is consistent with the state observed at `currentTopPacked_Phase1`.
            *out_dataValue = dataRead;
            return true;
        }
        // Else: The top pointer changed. The data we read might be stale or from a different node.
        // Loop again to retry the whole process.
    }
}


// --- Test Kernels (Device-side Functions launched from Host) ---

/**
 * @brief Original test kernel: Each thread decides independently whether to push or pop based on its thread ID (even/odd).
 * Can suffer from intra-warp divergence if threads in a warp take different paths (push vs pop).
 */
__global__ void mixedPushPopTestKernel_device(
    ConcurrentDeviceStack* d_stack,
    int num_ops_total,
    bool* d_push_results, // Optional: Device array to record push success/fail
    bool* d_pop_results,  // Optional: Device array to record pop success/fail
    int* d_pop_values)     // Optional: Device array to record popped values
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Only threads corresponding to an operation index execute.
    if (tid < num_ops_total) {
        if (tid % 2 == 0) {
            // Even threads PUSH
            int value_to_push = tid; // Use tid as a simple unique value
            bool success = pushOntoDeviceStack(d_stack, value_to_push);
            if (d_push_results) d_push_results[tid / 2] = success;
        } else {
            // Odd threads POP
            int popped_value = -1; // Default value if pop fails
            bool success = popFromDeviceStack(d_stack, &popped_value);
            if (d_pop_results) d_pop_results[tid / 2] = success;
            // Only record the value if the pop was successful
            if (d_pop_values && success) d_pop_values[tid / 2] = popped_value;
        }
    }
}

/**
 * @brief Improved test kernel for scalability: Assigns push or pop tasks based on the *warp* ID.
 * All threads within a warp perform the *same* operation (either all push or all pop).
 * This avoids the top-level if/else divergence within a warp, potentially improving performance,
 * although divergence can still occur *inside* the push/pop functions based on stack/freelist state.
 */
__global__ void warpAssignedPushPopTestKernel_device( // Renamed for clarity
    ConcurrentDeviceStack* d_stack,
    int num_ops_total
    // Removed optional logging pointers as they were passed as nullptr in the scalability test anyway
    )
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: Only threads corresponding to operations proceed
    if (tid < num_ops_total) {

        // Determine the warp this thread belongs to. Integer division groups threads.
        int warp_id = tid / WARP_SIZE;

        // Assign task based on WARP ID (even warps push, odd warps pop)
        if (warp_id % 2 == 0) {
            // --- This entire warp performs PUSH operations ---
            // Still use tid to generate unique values for simplicity in this test
            int value_to_push = tid;
            /* bool success = */ pushOntoDeviceStack(d_stack, value_to_push);
            // Note: Can still have divergence *inside* pushOntoDeviceStack (free list vs allocator path).
        } else {
            // --- This entire warp performs POP operations ---
            int popped_value = -1; // Variable needed for popFromDeviceStack signature
            /* bool success = */ popFromDeviceStack(d_stack, &popped_value);
            // Note: Can still have divergence *inside* popFromDeviceStack (stack empty check).
        }
    }
}


/**
 * @brief A kernel specifically designed for detailed reporting during verification tests.
 * It performs either pushes or pops based on `op_mode` and records detailed results.
 * Allows using pre-defined values for pushes.
 */
__global__ void reportingPushPopKernel_device(
    ConcurrentDeviceStack* d_stack,
    int num_ops,                     // Number of operations this kernel launch performs
    int base_value_or_unused,        // Base value if generating push values, unused otherwise
    bool use_predefined_push_values, // Flag: If true, use values from d_predefined_push_values
    const int* d_predefined_push_values, // Device pointer to array of values to push (if flag is true)
    int op_mode,                     // 0 for Push, 1 for Pop
    // Output arrays (device pointers)
    int* d_op_tid,                   // Records the thread ID (tid) for each operation attempt
    int* d_op_type,                  // Records the operation type (0 or 1)
    bool* d_op_success,              // Records success (true) or failure (false)
    int* d_op_value)                 // Records the value pushed OR the value popped (-1 if pop failed)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < num_ops) {
        // Record thread ID and operation type for this attempt
        d_op_tid[tid] = tid;
        d_op_type[tid] = op_mode;

        if (op_mode == 0) { // --- PUSH Operation ---
            int value_to_push = -1; // Default invalid value
            bool can_proceed_with_push = true;

            // Determine the value to push
            if (use_predefined_push_values) {
                if (d_predefined_push_values != nullptr) {
                    // Read the predefined value for this thread
                    value_to_push = d_predefined_push_values[tid];
                } else {
                    // Error: Told to use predefined array, but it's null.
                    printf("Kernel ERROR: Told to use predefined values, but array pointer is NULL! tid=%d\n", tid);
                    can_proceed_with_push = false; // Cannot proceed
                    value_to_push = -999; // Indicate error in log value
                }
            } else {
                // Generate value based on tid and base value
                value_to_push = tid + base_value_or_unused;
            }

            // Perform the push if possible
            bool success = false;
            if (can_proceed_with_push) {
                success = pushOntoDeviceStack(d_stack, value_to_push);
            }

            // Record results
            d_op_success[tid] = success;
            d_op_value[tid] = value_to_push; // Log the value *attempted* to push

        } else { // --- POP Operation ---
            int popped_value = -1; // Default if pop fails
            bool success = popFromDeviceStack(d_stack, &popped_value);

            // Record results
            d_op_success[tid] = success;
            // Log the actual popped value if successful, otherwise log -1
            d_op_value[tid] = success ? popped_value : -1;
        }
    }
}


// --- Host Code (CPU-side Management and Testing) ---

/**
 * @brief Resets the state of the concurrent stack on the device.
 * Sets the main stack top and free list top to NULL/Invalid, and resets the allocator index.
 *
 * @param d_stackManager Pointer to the ConcurrentDeviceStack structure *in device memory*.
 * @param nodePoolCapacity The capacity of the node pool (needed for verification message).
 */
void resetDeviceStack_host(ConcurrentDeviceStack* d_stackManager, int nodePoolCapacity) {
    printf("Resetting device stack (Capacity: %d)...\n", nodePoolCapacity);
    // Prepare a packed representation of {INVALID_NODE_INDEX, 0} on the host
    NodePointerWithTag null_tagged_pointer = {INVALID_NODE_INDEX, 0};
    uint64_t atomic_null_value_packed = packNodePointerWithTag(null_tagged_pointer);

    int zero_allocator_index = 0;

    // Copy the packed null value to the device atomic variables for stack top and free list top
    gpuErrchk(cudaMemcpy(&(d_stackManager->deviceStackTopAtomic), &atomic_null_value_packed, sizeof(uint64_t), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(&(d_stackManager->deviceFreeListTopAtomic), &atomic_null_value_packed, sizeof(uint64_t), cudaMemcpyHostToDevice));

    // Copy zero to reset the node allocator index
    gpuErrchk(cudaMemcpy(&(d_stackManager->deviceNodeAllocatorIndex), &zero_allocator_index, sizeof(int), cudaMemcpyHostToDevice));

    // Ensure all copies are complete before proceeding
    gpuErrchk(cudaDeviceSynchronize());
    printf("Stack reset complete.\n");
}

/**
 * @brief Prints a log of push/pop operations performed during verification tests.
 *
 * @param num_ops Number of operations in the log.
 * @param h_op_tid Host array of thread IDs for each operation.
 * @param h_op_type Host array of operation types (0=Push, 1=Pop).
 * @param h_op_success Host array of success/failure flags.
 * @param h_op_value Host array of values pushed or popped.
 */
void printOperationLog_host(int num_ops, int* h_op_tid, int* h_op_type, bool* h_op_success, int* h_op_value) {
    printf("--- Operation Log ---\n");
    printf("Attempt | TID   | Op   | Value | Success\n");
    printf("--------|-------|------|-------|--------\n");
    for (int i = 0; i < num_ops; ++i) {
        printf("%-7d | %-5d | %-4s | %-5d | %s\n",
               i,
               h_op_tid[i],
               (h_op_type[i] == 0) ? "Push" : "Pop ",
               h_op_value[i], // Shows attempted push value or popped value (-1 if pop fail)
               h_op_success[i] ? "True " : "False");
    }
    printf("---------------------\n");
}


// --- Verification Tests (Host-side Logic) ---

/**
 * @brief Verification Test [1]: Pushes N items, then tries to pop M items (M > N).
 * Verifies that exactly N items are popped successfully, that the values match the
 * ones pushed (using multisets for order-agnostic comparison), and that the stack
 * ends up empty. Uses the reporting kernel and pre-defined push values.
 */
bool runVerificationTest_PopsGreaterThanPushes_host(ConcurrentDeviceStack* d_stackManager, int nodePoolCapacity, int cudaBlockSize) {
    printf("\n--- Verification [1]: Pops > Pushes (Reclamation Enabled, Predefined Values Flag) ---\n");
    bool test_passed = true;
    const int num_pushes = 150;
    const int num_pops = num_pushes + 50; // More pops than pushes
    const int base_value_for_generation = 100; // Used to generate varied push values
    const int total_log_entries = num_pushes + num_pops;

    printf("Config: Initial Pushes=%d, Total Pops=%d, Pool Capacity=%d, Block Size=%d\n",
           num_pushes, num_pops, nodePoolCapacity, cudaBlockSize);
    if (num_pushes > nodePoolCapacity) {
        printf("Warning: num_pushes (%d) exceeds pool capacity (%d). Expecting push failures.\n", num_pushes, nodePoolCapacity);
        // Test should still work correctly if pushes fail due to capacity limit
    }

    // Allocate host and device memory for logging operation results
    int *d_log_tid = nullptr, *h_log_tid = nullptr;
    int *d_log_type = nullptr, *h_log_type = nullptr;
    bool *d_log_success = nullptr, *h_log_success = nullptr;
    int *d_log_value = nullptr, *h_log_value = nullptr;

    gpuErrchk(cudaMalloc(&d_log_tid, total_log_entries * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_log_type, total_log_entries * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_log_success, total_log_entries * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_log_value, total_log_entries * sizeof(int)));

    h_log_tid = (int*)malloc(total_log_entries * sizeof(int));
    h_log_type = (int*)malloc(total_log_entries * sizeof(int));
    h_log_success = (bool*)malloc(total_log_entries * sizeof(bool));
    h_log_value = (int*)malloc(total_log_entries * sizeof(int));

    if (!h_log_tid || !h_log_type || !h_log_success || !h_log_value) {
        fprintf(stderr, "Host memory allocation failed for logging!\n");
        // Need to free any potentially allocated device memory before returning
        cudaFree(d_log_tid); cudaFree(d_log_type); cudaFree(d_log_success); cudaFree(d_log_value);
        return false;
    }

    // Generate distinct values to be pushed on the host
    printf("Generating %d distinct push values on Host...\n", num_pushes);
    int* h_push_values_source = (int*)malloc(num_pushes * sizeof(int));
    if (!h_push_values_source) {
        fprintf(stderr, "Host memory allocation failed for push values!\n");
        // Clean up logs
        free(h_log_tid); free(h_log_type); free(h_log_success); free(h_log_value);
        cudaFree(d_log_tid); cudaFree(d_log_type); cudaFree(d_log_success); cudaFree(d_log_value);
        return false;
    }
    for (int i = 0; i < num_pushes; ++i) {
        // Simple formula to generate somewhat varied values
        h_push_values_source[i] = base_value_for_generation + i * 2 + (i % 3);
    }

    // Allocate device memory for push values and copy them
    int* d_push_values_source = nullptr;
    gpuErrchk(cudaMalloc(&d_push_values_source, num_pushes * sizeof(int)));
    printf("Copying %d push values from Host to Device...\n", num_pushes);
    gpuErrchk(cudaMemcpy(d_push_values_source, h_push_values_source, num_pushes * sizeof(int), cudaMemcpyHostToDevice));

    // Reset the stack on the device before the test
    resetDeviceStack_host(d_stackManager, nodePoolCapacity);

    // --- Phase 1: Push Operations ---
    printf("Launching Push Kernel (%d operations, using predefined values)...\n", num_pushes);
    int grid_size_push = (num_pushes + cudaBlockSize - 1) / cudaBlockSize;
    reportingPushPopKernel_device<<<grid_size_push, cudaBlockSize>>>(
        d_stackManager,
        num_pushes,                      // Number of push ops
        0,                              // Base value unused here
        true,                           // Use predefined values flag
        d_push_values_source,           // Pointer to predefined values on device
        0,                              // Op mode 0 = Push
        d_log_tid, d_log_type, d_log_success, d_log_value // Log output arrays (start)
    );
    gpuErrchk(cudaGetLastError()); // Check for launch errors
    gpuErrchk(cudaDeviceSynchronize()); // Wait for push kernel to finish

    // --- Phase 2: Pop Operations ---
    printf("Launching Pop Kernel (%d operations)...\n", num_pops);
    int grid_size_pop = (num_pops + cudaBlockSize - 1) / cudaBlockSize;
    // Launch kernel for pop operations, writing results to the *offset* portion of the log arrays
    reportingPushPopKernel_device<<<grid_size_pop, cudaBlockSize>>>(
        d_stackManager,
        num_pops,                       // Number of pop ops
        0,                              // Base value unused
        false,                          // Don't use predefined values
        nullptr,                        // Predefined values pointer is null
        1,                              // Op mode 1 = Pop
        d_log_tid + num_pushes,        // Offset log pointers
        d_log_type + num_pushes,
        d_log_success + num_pushes,
        d_log_value + num_pushes
    );
    gpuErrchk(cudaGetLastError()); // Check for launch errors
    gpuErrchk(cudaDeviceSynchronize()); // Wait for pop kernel to finish

    // --- Phase 3: Copy Results Back and Verify ---
    printf("Copying complete log data (%d entries) from Device to Host...\n", total_log_entries);
    gpuErrchk(cudaMemcpy(h_log_tid, d_log_tid, total_log_entries * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_log_type, d_log_type, total_log_entries * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_log_success, d_log_success, total_log_entries * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_log_value, d_log_value, total_log_entries * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the detailed log
    printOperationLog_host(total_log_entries, h_log_tid, h_log_type, h_log_success, h_log_value);

    // Verification logic
    int successful_pushes_counted = 0;
    int successful_pops_counted = 0;
    std::multiset<int> successfully_pushed_values_multiset; // Use multiset to handle potential duplicate values if generation logic allowed it
    std::multiset<int> successfully_popped_values_multiset;

    printf("Verifying results...\n");
    int expected_pushes = 0; // Track how many pushes we *expect* to succeed (limited by capacity)

    for(int i = 0; i < total_log_entries; ++i) {
        if (i < num_pushes) { // --- Analyzing Push Log Entries ---
            if (h_log_type[i] != 0) {
                 printf("WARN: Expected push log entry at index %d, but found type %d\n", i, h_log_type[i]);
            }
            // Find the original value attempted to push (using tid recorded in log)
            int original_push_tid = h_log_tid[i];
            if (original_push_tid >= 0 && original_push_tid < num_pushes) {
                 if (h_log_value[i] != h_push_values_source[original_push_tid]) {
                     printf("WARN: Logged push value %d for tid %d at log index %d does not match original source value %d\n",
                            h_log_value[i], original_push_tid, i, h_push_values_source[original_push_tid]);
                 }
                 // Check if this push *should* have succeeded based on pool capacity
                 // This is approximate due to concurrency, but helps sanity check
                 if (original_push_tid < nodePoolCapacity) {
                     expected_pushes++;
                 }
                 if (h_log_success[i]) {
                     successful_pushes_counted++;
                     successfully_pushed_values_multiset.insert(h_log_value[i]);
                 }
            } else {
                 printf("WARN: Invalid tid %d found in push log entry at index %d\n", original_push_tid, i);
            }
        } else { // --- Analyzing Pop Log Entries ---
            if (h_log_type[i] != 1) {
                printf("WARN: Expected pop log entry at index %d, but found type %d\n", i, h_log_type[i]);
            }
            if (h_log_success[i]) {
                successful_pops_counted++;
                // Check if the popped value is -1, which shouldn't happen if success is true
                if (h_log_value[i] == -1) {
                    printf("WARN: Pop at log index %d reported success but value is -1\n", i);
                }
                successfully_popped_values_multiset.insert(h_log_value[i]);
            } else {
                 // Check if the value is not -1 when pop failed
                 if (h_log_value[i] != -1) {
                     printf("WARN: Pop at log index %d reported failure but value is %d (expected -1)\n", i, h_log_value[i]);
                 }
            }
        }
    }

    // Perform final checks
    printf("Result: Successful Pushes recorded: %d\n", successful_pushes_counted);
    printf("Result: Successful Pops recorded: %d\n", successful_pops_counted);

    // 1. Check: Number of successful pops must equal number of successful pushes
    if (successful_pops_counted != successful_pushes_counted) {
        printf("FAIL: Number of successful pops (%d) does not match number of successful pushes (%d).\n",
               successful_pops_counted, successful_pushes_counted);
        test_passed = false;
    } else {
        printf("Result: Pop count matches push count.\n");
    }

    // 2. Check: The set of successfully popped values must exactly match the set of successfully pushed values
    if (successfully_pushed_values_multiset != successfully_popped_values_multiset) {
        printf("FAIL: The multiset of popped values does not match the multiset of pushed values.\n");
        // Could print contents of both multisets here for debugging if needed
        test_passed = false;
    } else {
        printf("Result: Multisets of pushed and popped values match.\n");
    }

    // 3. Check: Final state of the stack should be empty
    uint64_t final_top_packed;
    gpuErrchk(cudaMemcpy(&final_top_packed, &(d_stackManager->deviceStackTopAtomic), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    NodePointerWithTag final_top_pointer = unpackNodePointerWithTag(final_top_packed);
    printf("Result: Final main stack top index: %u (Expected: %u for empty stack)\n", final_top_pointer.nodeIndex, INVALID_NODE_INDEX);
    if (final_top_pointer.nodeIndex != INVALID_NODE_INDEX) {
       printf("FAIL: Final stack top index is not INVALID_NODE_INDEX, stack is not empty.\n");
       test_passed = false;
    }

    // 4. Check: Final allocator index (informational, less strict check due to concurrency)
    int final_allocator_idx_host = -1;
    gpuErrchk(cudaMemcpy(&final_allocator_idx_host, &(d_stackManager->deviceNodeAllocatorIndex), sizeof(int), cudaMemcpyDeviceToHost));
    // Expected value is roughly the number of nodes allocated, which is limited by capacity
    int expected_alloc_idx_approx = std::min(num_pushes, nodePoolCapacity);
    printf("Result: Final node allocator index: %d (Expected approx %d if pool >= pushes)\n", final_allocator_idx_host, expected_alloc_idx_approx);
    // No strict fail here, just for info.

    printf("Test [1] Result: %s\n", test_passed ? "PASS" : "FAIL");

    // --- Cleanup for this test ---
    free(h_log_tid); free(h_log_type); free(h_log_success); free(h_log_value);
    gpuErrchk(cudaFree(d_log_tid)); gpuErrchk(cudaFree(d_log_type)); gpuErrchk(cudaFree(d_log_success)); gpuErrchk(cudaFree(d_log_value));
    free(h_push_values_source); gpuErrchk(cudaFree(d_push_values_source));
    return test_passed;
}


/**
 * @brief Verification Test [2]: Attempts to pop from a deliberately empty stack.
 * Verifies that all pop operations fail and that the stack remains empty.
 */
bool runVerificationTest_PopEmptyStack_host(ConcurrentDeviceStack* d_stackManager, int nodePoolCapacity, int cudaBlockSize) {
    printf("\n--- Verification [2]: Pop Empty Stack (Reclamation Enabled) ---\n");
    bool test_passed = true;
    const int num_pops_attempted = 50;
    printf("Config: Pops Attempted=%d, Pool Capacity=%d, Block Size=%d\n", num_pops_attempted, nodePoolCapacity, cudaBlockSize);

    // Allocate log arrays
    int *d_log_tid = nullptr, *h_log_tid = nullptr;
    int *d_log_type = nullptr, *h_log_type = nullptr;
    bool *d_log_success = nullptr, *h_log_success = nullptr;
    int *d_log_value = nullptr, *h_log_value = nullptr;

    gpuErrchk(cudaMalloc(&d_log_tid, num_pops_attempted * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_log_type, num_pops_attempted * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_log_success, num_pops_attempted * sizeof(bool)));
    gpuErrchk(cudaMalloc(&d_log_value, num_pops_attempted * sizeof(int)));

    h_log_tid = (int*)malloc(num_pops_attempted * sizeof(int));
    h_log_type = (int*)malloc(num_pops_attempted * sizeof(int));
    h_log_success = (bool*)malloc(num_pops_attempted * sizeof(bool));
    h_log_value = (int*)malloc(num_pops_attempted * sizeof(int));

    if (!h_log_tid || !h_log_type || !h_log_success || !h_log_value) {
        fprintf(stderr, "Host memory allocation failed for logging!\n");
        cudaFree(d_log_tid); cudaFree(d_log_type); cudaFree(d_log_success); cudaFree(d_log_value);
        return false;
    }

    // Reset stack to ensure it's empty
    resetDeviceStack_host(d_stackManager, nodePoolCapacity);

    // --- Launch Pop Kernel ---
    printf("Launching Pop Kernel on empty stack (%d operations)...\n", num_pops_attempted);
    int grid_size_pop = (num_pops_attempted + cudaBlockSize - 1) / cudaBlockSize;
    reportingPushPopKernel_device<<<grid_size_pop, cudaBlockSize>>>(
        d_stackManager, num_pops_attempted, 0, false, nullptr, 1, // OpMode 1 = Pop
        d_log_tid, d_log_type, d_log_success, d_log_value
    );
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Copy Results Back and Verify ---
    printf("Copying log data (%d entries) from Device to Host...\n", num_pops_attempted);
    gpuErrchk(cudaMemcpy(h_log_tid, d_log_tid, num_pops_attempted * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_log_type, d_log_type, num_pops_attempted * sizeof(int), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_log_success, d_log_success, num_pops_attempted * sizeof(bool), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(h_log_value, d_log_value, num_pops_attempted * sizeof(int), cudaMemcpyDeviceToHost));

    printOperationLog_host(num_pops_attempted, h_log_tid, h_log_type, h_log_success, h_log_value);

    // Verification logic
    int successful_pops_counted = 0;
    printf("Verifying results...\n");
    for(int i = 0; i < num_pops_attempted; ++i) {
         if (h_log_type[i] != 1) {
              printf("WARN: Expected pop log entry at index %d, but found type %d\n", i, h_log_type[i]);
         }
         if (h_log_success[i]) {
             successful_pops_counted++;
             printf("FAIL: Pop operation at log index %d (tid %d) succeeded unexpectedly on an empty stack!\n", i, h_log_tid[i]);
             test_passed = false;
         }
         if (h_log_value[i] != -1) {
             printf("WARN: Pop operation at log index %d (tid %d) failed as expected, but value is %d (expected -1)\n", i, h_log_tid[i], h_log_value[i]);
         }
    }

    printf("Result: Total successful pops recorded: %d (Expected: 0)\n", successful_pops_counted);
    if (successful_pops_counted != 0) {
        // Failure already flagged in loop
    }

    // Check final state: Stack and Free List should both be empty
    uint64_t final_top_packed, final_free_packed;
    gpuErrchk(cudaMemcpy(&final_top_packed, &(d_stackManager->deviceStackTopAtomic), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    gpuErrchk(cudaMemcpy(&final_free_packed, &(d_stackManager->deviceFreeListTopAtomic), sizeof(uint64_t), cudaMemcpyDeviceToHost));
    NodePointerWithTag final_top_pointer = unpackNodePointerWithTag(final_top_packed);
    NodePointerWithTag final_free_pointer = unpackNodePointerWithTag(final_free_packed);

    printf("Result: Final main stack top index: %u (Expected: %u)\n", final_top_pointer.nodeIndex, INVALID_NODE_INDEX);
    if (final_top_pointer.nodeIndex != INVALID_NODE_INDEX) {
       printf("FAIL: Final stack top index is not INVALID_NODE_INDEX.\n");
       test_passed = false;
    }
    printf("Result: Final free list top index: %u (Expected: %u)\n", final_free_pointer.nodeIndex, INVALID_NODE_INDEX);
     if (final_free_pointer.nodeIndex != INVALID_NODE_INDEX) {
       printf("FAIL: Final free list top index is not INVALID_NODE_INDEX.\n");
       test_passed = false;
    }

    printf("Test [2] Result: %s\n", test_passed ? "PASS" : "FAIL");

    // --- Cleanup ---
    free(h_log_tid); free(h_log_type); free(h_log_success); free(h_log_value);
    gpuErrchk(cudaFree(d_log_tid)); gpuErrchk(cudaFree(d_log_type)); gpuErrchk(cudaFree(d_log_success)); gpuErrchk(cudaFree(d_log_value));
    return test_passed;
}

/**
 * @brief Verification Test [3]: Attempts to push more items than the node pool capacity.
 * Verifies that pushes succeed up to roughly the capacity limit and then start failing.
 * Checks that the allocator index reflects the exhaustion.
 */
bool runVerificationTest_PoolExhaustion_host(ConcurrentDeviceStack* d_stackManager, int nodePoolCapacity, int cudaBlockSize) {
    printf("\n--- Verification [3]: Node Pool Exhaustion (Reclamation Enabled) ---\n");
    printf("      NOTE: This test uses kernel-calculated push values (tid + base_value).\n");
    bool test_passed = true;
    // Attempt to push slightly more items than the pool capacity
    const int num_pushes_attempted = nodePoolCapacity + 50;
    const int base_value_for_generation = 400;
    printf("Config: Pool Capacity=%d, Pushes Attempted=%d, Block Size=%d\n", nodePoolCapacity, num_pushes_attempted, cudaBlockSize);

    // Allocate minimal log arrays needed (only success status is strictly required for verification)
    int *d_dummy_tid = nullptr, *h_dummy_tid = nullptr; // Using dummy names as not needed for check
    int *d_dummy_type = nullptr, *h_dummy_type = nullptr;
    bool *d_log_success = nullptr, *h_log_success = nullptr;
    int *d_dummy_value = nullptr, *h_dummy_value = nullptr;

    gpuErrchk(cudaMalloc(&d_dummy_tid, num_pushes_attempted * sizeof(int))); // Still need to pass valid pointers to kernel
    gpuErrchk(cudaMalloc(&d_dummy_type, num_pushes_attempted * sizeof(int)));
    gpuErrchk(cudaMalloc(&d_log_success, num_pushes_attempted * sizeof(bool))); // Need success status
    gpuErrchk(cudaMalloc(&d_dummy_value, num_pushes_attempted * sizeof(int)));

    // Host array only for success status
    h_log_success = (bool*)malloc(num_pushes_attempted * sizeof(bool));
    // Allocate others just to free them cleanly later if needed, though not strictly necessary if check fails early
    h_dummy_tid = (int*)malloc(num_pushes_attempted * sizeof(int));
    h_dummy_type = (int*)malloc(num_pushes_attempted * sizeof(int));
    h_dummy_value = (int*)malloc(num_pushes_attempted * sizeof(int));


    if (!h_log_success || !h_dummy_tid || !h_dummy_type || !h_dummy_value ) {
        fprintf(stderr, "Host memory allocation failed for logging!\n");
        // Clean up device memory
        cudaFree(d_dummy_tid); cudaFree(d_dummy_type); cudaFree(d_log_success); cudaFree(d_dummy_value);
        return false;
    }

    // Reset stack (clears free list, resets allocator)
    resetDeviceStack_host(d_stackManager, nodePoolCapacity);

    // --- Launch Push Kernel ---
    printf("Launching Push Kernel (%d operations, expecting pool exhaustion)...\n", num_pushes_attempted);
    int grid_size_push = (num_pushes_attempted + cudaBlockSize - 1) / cudaBlockSize;
    reportingPushPopKernel_device<<<grid_size_push, cudaBlockSize>>>(
        d_stackManager,
        num_pushes_attempted,
        base_value_for_generation,      // Base value for tid+base generation
        false,                          // Don't use predefined values
        nullptr,                        // Predefined values ptr is null
        0,                              // Op Mode 0 = Push
        d_dummy_tid, d_dummy_type, d_log_success, d_dummy_value // Pass log arrays
    );
    gpuErrchk(cudaGetLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // --- Copy Results Back and Verify ---
    printf("Copying success log data (%d entries) from Device to Host...\n", num_pushes_attempted);
    gpuErrchk(cudaMemcpy(h_log_success, d_log_success, num_pushes_attempted * sizeof(bool), cudaMemcpyDeviceToHost));

    // Verification logic
    int successful_pushes_counted = 0;
    int failed_pushes_counted = 0;
    printf("Verifying results...\n");
    for(int i = 0; i < num_pushes_attempted; ++i) {
        if (h_log_success[i]) {
            successful_pushes_counted++;
        } else {
            failed_pushes_counted++;
        }
    }

    printf("Result: Successful pushes recorded: %d\n", successful_pushes_counted);
    printf("Result: Failed pushes recorded: %d\n", failed_pushes_counted);

    // 1. Check: Did pushes fail? (Expected if attempts > capacity)
    if (num_pushes_attempted > nodePoolCapacity) {
        if (failed_pushes_counted == 0) {
            printf("FAIL: Expected push failures as num_pushes_attempted (%d) > capacity (%d), but recorded 0 failures.\n",
                   num_pushes_attempted, nodePoolCapacity);
            test_passed = false;
        } else {
            printf("Result: Observed push failures as expected due to pool capacity limit.\n");
        }
    } else {
        // If attempts <= capacity, we ideally expect no failures, but concurrency might cause some. Issue a warning if failures occur.
        if (failed_pushes_counted > 0) {
            printf("WARN: Pushes failed (%d) even though num_pushes_attempted (%d) <= capacity (%d). This might occur due to high contention for the allocator index but should be investigated if persistent.\n",
                   failed_pushes_counted, num_pushes_attempted, nodePoolCapacity);
        }
    }

    // 2. Check: Number of successful pushes should be around the pool capacity.
    // Allow some slack due to concurrency in accessing the allocator index.
    int lower_bound = nodePoolCapacity - cudaBlockSize * 2; // Heuristic tolerance
    int upper_bound = nodePoolCapacity + cudaBlockSize * 2; // Heuristic tolerance
    lower_bound = std::max(0, lower_bound); // Ensure lower bound isn't negative

    printf("Result: Successful pushes (%d) expected around pool capacity (%d). Checking against approximate bounds [%d, %d].\n",
           successful_pushes_counted, nodePoolCapacity, lower_bound, upper_bound);

    // Check if successful pushes are reasonably close to capacity. This is not a strict failure condition but a sanity check.
    if (successful_pushes_counted < lower_bound || successful_pushes_counted > upper_bound) {
        printf("WARN: Number of successful pushes (%d) is outside the expected approximate range around pool capacity (%d).\n",
               successful_pushes_counted, nodePoolCapacity);
        // Consider making this a failure if the deviation is very large.
    }
    // More strictly, successful pushes should not exceed capacity significantly
    if (successful_pushes_counted > nodePoolCapacity + cudaBlockSize) { // Allow one block worth of over-allocation possibility due to fetch_add races
        printf("FAIL: Number of successful pushes (%d) significantly exceeds pool capacity (%d).\n", successful_pushes_counted, nodePoolCapacity);
        test_passed = false;
    }


    // 3. Check: Final allocator index should indicate exhaustion
    int final_allocator_idx_host = -1;
    gpuErrchk(cudaMemcpy(&final_allocator_idx_host, &(d_stackManager->deviceNodeAllocatorIndex), sizeof(int), cudaMemcpyDeviceToHost));
    printf("Result: Final node allocator index: %d\n", final_allocator_idx_host);

    // If we expected exhaustion (attempts > capacity) and observed failures, the allocator index should be at least capacity.
    if (num_pushes_attempted > nodePoolCapacity && failed_pushes_counted > 0) {
        if (final_allocator_idx_host < nodePoolCapacity) {
            printf("FAIL: Pool exhaustion occurred (pushes failed), but final allocator index %d is less than capacity %d. Allocator may not have advanced correctly.\n",
                   final_allocator_idx_host, nodePoolCapacity);
            test_passed = false;
        } else {
            printf("Result: Final allocator index reached or exceeded capacity, consistent with pool exhaustion.\n");
        }
    } else if (final_allocator_idx_host >= nodePoolCapacity) {
         printf("Result: Final allocator index reached or exceeded capacity (%d).\n", final_allocator_idx_host);
    }


    printf("Test [3] Result: %s\n", test_passed ? "PASS" : "FAIL");

    // --- Cleanup ---
    // Free host memory
    free(h_dummy_tid); free(h_dummy_type); free(h_log_success); free(h_dummy_value);
    // Free device memory
    gpuErrchk(cudaFree(d_dummy_tid)); gpuErrchk(cudaFree(d_dummy_type)); gpuErrchk(cudaFree(d_log_success)); gpuErrchk(cudaFree(d_dummy_value));
    return test_passed;
}


// --- Main Function (Host Entry Point) ---

int main() {
    // Configuration
    int nodePoolCapacity = 1024 * 10;  // ~0.08 MB pool
    // int nodePoolCapacity = 1024 * 1000; // ~8 MB pool
    // int nodePoolCapacity = 1024 * 10000; // ~80MB pool
    const int cudaBlockSize = 256; // Typical block size

    printf("Initializing Lock-Free Linked-List Stack (Node Index + Tag = 64b Atomic)...\n");
    printf("Configuration:\n");
    printf("  Node Pool Capacity: %d nodes\n", nodePoolCapacity);
    printf("  CUDA Block Size:    %d threads\n", cudaBlockSize);
    printf("  StackNode size:     %zu bytes\n", sizeof(StackNode));
    printf("  INVALID_NODE_INDEX: %u\n", INVALID_NODE_INDEX);
    printf("========================================\n");

    // Allocate the node pool on the device
    StackNode* d_nodePoolPtr = nullptr;
    gpuErrchk(cudaMalloc((void**)&d_nodePoolPtr, (size_t)nodePoolCapacity * sizeof(StackNode)));
    printf("Device node pool allocated at %p\n", d_nodePoolPtr);

    // Allocate the main stack management struct on the device
    ConcurrentDeviceStack* d_stackManager = nullptr;
    gpuErrchk(cudaMalloc((void**)&d_stackManager, sizeof(ConcurrentDeviceStack)));
    printf("Device stack manager struct allocated at %p\n", d_stackManager);

    // Create a template structure on the host to copy initial values
    ConcurrentDeviceStack h_stack_template;
    h_stack_template.deviceNodePool = d_nodePoolPtr;        // Set the device pointer
    h_stack_template.nodePoolCapacity = nodePoolCapacity;   // Set the capacity
    // Atomics (stackTop, freeListTop, allocatorIndex) will be initialized by resetDeviceStack_host

    // Copy the template structure (containing the pool pointer and capacity) to the device manager struct
    printf("Copying initial stack manager data (pool ptr, capacity) to device...\n");
    gpuErrchk(cudaMemcpy(d_stackManager, &h_stack_template, sizeof(ConcurrentDeviceStack), cudaMemcpyHostToDevice));

    // Initialize the atomic variables within the device manager struct
    resetDeviceStack_host(d_stackManager, nodePoolCapacity);

    // --- Run Verification Tests ---
    printf("\nStarting Verification Tests...\n");
    bool all_verification_tests_passed = true;
    all_verification_tests_passed &= runVerificationTest_PopsGreaterThanPushes_host(d_stackManager, nodePoolCapacity, cudaBlockSize);
    all_verification_tests_passed &= runVerificationTest_PopEmptyStack_host(d_stackManager, nodePoolCapacity, cudaBlockSize);
    all_verification_tests_passed &= runVerificationTest_PoolExhaustion_host(d_stackManager, nodePoolCapacity, cudaBlockSize);

    printf("\n--- Overall Verification Result: %s ---\n", all_verification_tests_passed ? "ALL PASS" : "SOME FAIL");
    printf("========================================\n");

    // --- Run Scalability/Throughput Tests (only if verification passes) ---
    if (all_verification_tests_passed) {
        printf("\nStarting Scalability/Throughput Tests...\n");
        // Define different numbers of operations to test
        int num_ops_tests[] = {10000, 50000, 100000, 200000, 500000, 1000000, 2000000}; // Added more ops

        for (int num_operations : num_ops_tests) {
             // Ensure stack is reset before each throughput test
            resetDeviceStack_host(d_stackManager, nodePoolCapacity);

            printf("\nRunning Throughput Test: %d Operations\n", num_operations);
            printf("  Using Warp-Assigned Kernel (avoids intra-warp push/pop divergence)\n");

            // Calculate grid size based on total operations and block size
            int grid_size = (num_operations + cudaBlockSize - 1) / cudaBlockSize;
            printf("  Grid Size: %d blocks, Block Size: %d threads, Total Threads Launched: %d\n",
                   grid_size, cudaBlockSize, grid_size * cudaBlockSize);

            // Ensure any previous CUDA activity is finished
            gpuErrchk(cudaDeviceSynchronize());

            // Create CUDA events for timing
            cudaEvent_t start_event, stop_event;
            gpuErrchk(cudaEventCreate(&start_event));
            gpuErrchk(cudaEventCreate(&stop_event));

            // Record start event
            gpuErrchk(cudaEventRecord(start_event));

            // Launch the warp-assigned kernel for performance measurement
            // Passing nullptr for logging arrays as we only measure time here.
            warpAssignedPushPopTestKernel_device<<<grid_size, cudaBlockSize>>>(
                d_stackManager,
                num_operations
                );

            // Record stop event and check for kernel launch errors
            gpuErrchk(cudaGetLastError());
            gpuErrchk(cudaEventRecord(stop_event));

            // Synchronize the host thread with the stop event (waits for kernel completion)
            gpuErrchk(cudaEventSynchronize(stop_event));

            // Calculate elapsed time
            float milliseconds = 0;
            gpuErrchk(cudaEventElapsedTime(&milliseconds, start_event, stop_event));

            // Clean up events
            gpuErrchk(cudaEventDestroy(start_event));
            gpuErrchk(cudaEventDestroy(stop_event));

            // Report results
            printf("  Execution Time: %.4f ms\n", milliseconds);
            double seconds = milliseconds / 1000.0;
            if (seconds > 1e-9) { // Avoid division by zero if timing is too fast
                double ops_per_second = (double)num_operations / seconds;
                printf("  Throughput: %.2f Million Operations/Second (MOps/s)\n", ops_per_second / 1e6);
            } else {
                printf("  Throughput: Inf (Execution time too small to measure accurately)\n");
            }
            printf("----------------------------------------\n");
           }
    } else {
        printf("\nSkipping Scalability/Throughput Tests due to Verification Failures.\n");
        printf("========================================\n");
    }

    // --- Cleanup ---
    printf("\nCleaning up GPU resources...\n");
    // Free the device node pool first
    if (d_nodePoolPtr) {
        gpuErrchk(cudaFree(d_nodePoolPtr));
        printf("Device node pool freed.\n");
    }
    // Free the device stack manager struct
    if (d_stackManager) {
        gpuErrchk(cudaFree(d_stackManager));
        printf("Device stack manager struct freed.\n");
    }

    printf("Done.\n");
    return all_verification_tests_passed ? 0 : 1; // Return 0 on success, 1 on failure
}
