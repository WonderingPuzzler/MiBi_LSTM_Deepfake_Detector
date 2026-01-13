#include <cuda_runtime.h> // for CUDA runtime API
#include <limits> // for std::numeric_limits
#include <thrust/fill.h> // for thrust::fill
#include <thrust/device_ptr.h> // for thrust::device_ptr
#include <thrust/execution_policy.h> // for thrust::device


/* --------------------------------------------------------------------------------------------------------------------------*/


// Define number of banks and log2 of number of banks for shared memory bank conflict avoidance
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4

// Define CONFLICT_FREE_OFFSET macro to avoid shared memory bank conflicts
#ifdef ZERO_BANK_CONFLICTS
// If ZERO_BANK_CONFLICTS is defined, use a more complex offset calculation where each bank has an extra padding of one float
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> (LOG_NUM_BANKS) + (n) >> (2 * LOG_NUM_BANKS))

// Otherwise, use a simpler offset calculation
#else
#define CONFLICT_FREE_OFFSET(n) ((n) >> LOG_NUM_BANKS)
#endif

// Stream pool for efficient reuse (avoids create/destroy overhead)
static const int NUM_STREAMS = 4;
static cudaStream_t stream_pool[NUM_STREAMS];
static bool streams_initialized = false;
static int stream_counter = 0;


/* --------------------------------------------------------------------------------------------------------------------------*/

// Device function for log-space addition
__device__ inline float log_add_exp(float x, float y);

// Device function for computing next power of 2
__host__ __device__ inline int next_power_of_2(int n);

// Device function for padding
__global__ void pad_with_value_kernel(float *data, int start, int end, float value);

// Helper to launch padding kernel
inline void pad_array(float *d_array, int start, int end, float value);

// Single-block Blelloch Scan
__global__ void prescan(float *g_odata, float *g_idata, int n, bool log_space);
__global__ void prescan_blocks(float *g_odata, float *g_idata, float *d_block_sums, int n, bool log_space);
__global__ void add_block_sums(float *g_odata, float *g_block_sums, int n, bool log_space);

// LSTM-style scan with forget gates
__global__ void prescan_lstm(float *g_odata, float *g_b, float *g_a, int n, bool log_space);
__global__ void prescan_blocks_lstm(float *g_odata, float *g_b, float *g_a, float *d_block_sums, float *d_block_a_products, int n, bool log_space);
__global__ void add_block_sums_lstm(float *g_odata, float *g_a, float *g_block_sums, float *g_block_a_products, int n, bool log_space);
void prescan_lstm_2048(float *d_odata, float *d_b, float *d_a, int n, bool log_space, cudaStream_t stream = 0);

// Batched LSTM scan (processes entire batch at once)
__global__ void prescan_lstm_batched(float *g_odata, float *g_b, float *g_a, int batch_size, int seq_length, int hidden_size, bool log_space);
void prescan_lstm_batched_2048(float *d_odata, float *d_b, float *d_a, int batch_size, int seq_length, int hidden_size, bool log_space, cudaStream_t stream = 0);

// Stream functions
void init_streams();
void destroy_streams();
cudaStream_t get_stream();
void sync_all_streams();

// Declare wrapper functions
void prescan_2048_wrapper(float *d_odata, float *d_idata, int n, bool log_space);
void prescan_lstm_2048_wrapper(float* d_odata, float* d_b, float* d_a, int n, bool log_space);
void prescan_lstm_batched_2048_wrapper(float* d_odata, float* d_b, float* d_a, int batch_size, int seq_length, int hidden_size, bool log_space);

void prescan_2048_wrapper_sync(float *d_odata, float *d_idata, int n, bool log_space);
void prescan_lstm_2048_wrapper_sync(float* d_odata, float* d_b, float* d_a, int n, bool log_space);
void prescan_lstm_batched_2048_wrapper_sync(float* d_odata, float* d_b, float* d_a, int batch_size, int seq_length, int hidden_size, bool log_space);





/* --------------------------------------------------------------------------------------------------------------------------*/


// Fast bit manipulation for computing next power of 2
__host__ __device__ inline int next_power_of_2(int n) {

    // If n is already a power of 2, return n
    if (n <= 1) return 1;
    // Otherwise, compute next power of 2 by setting all bits below highest set bit
    n--; // Decrement n to handle exact powers of 2
    n |= n >> 1; // |= is bitwise OR assignment with shifted version
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return n + 1;
}

// Custom kernel for padding
__global__ void pad_with_value_kernel(float *data, int start, int end, float value) {

    // Compute index of current thread
    int idx = blockIdx.x * blockDim.x + threadIdx.x + start;

    // If index is within range, set value
    if (idx < end) {
        // Set data value
        data[idx] = value;
    }
}

// Helper to launch padding kernel
inline void pad_array(float *d_array, int start, int end, float value) {

    // get the padding size necessary
    int padding_size = end - start;

    // if padding is necessary, launch kernel
    if (padding_size > 0) {

        // get the number of threads needed
        int threads = 256;

        // get the number of blocks needed (using ceiling)
        int blocks = (padding_size + threads - 1) / threads;

        // launch kernel
        pad_with_value_kernel<<<blocks, threads>>>(d_array, start, end, value);
    }
}

// Device function to compute log-space addition
// Which computes log(exp(x) + exp(y)) in a numerically stable way
__device__ inline float log_add_exp(float x, float y) {
    // If x is -inf, return y
    if (isinf(x) && x < 0) return y;

    // If y is -inf, return x
    if (isinf(y) && y < 0) return x;

    // Find the maximum of x and y
    float m = fmaxf(x, y);

    // Return m + log(1 + exp(-|x - y|))
    return m + log(1.0f + expf(-fabsf(x - y)));

}

/* --------------------------------------------------------------------------------------------------------------------------*/

// Blelloch Parallel Prefix Sum (Scan) - Educational Implementation
// Computes exclusive prefix sum: out[i] = sum of input[0..i-1]
// Operation: Normal space uses +, Log space uses log_add_exp
// Identity: 0 for normal space, -inf for log space
//
// Algorithm Overview:
// 1. up-sweep (Reduce): Build a balanced binary tree by combining adjacent pairs
//    - Level 0: Combine pairs (0,1), (2,3), (4,5), ...
//    - Level 1: Combine pairs (0-1, 2-3), (4-5, 6-7), ...
//    - Continue until root contains total sum
//
// 2. root = 0: Set root to identity (enables exclusive scan)
//
// 3. down-sweep (Distribute): Propagate partial sums down tree
//    - Parent's value becomes left child's new value
//    - Right child = left child's old value + parent's value
//    - Results in exclusive prefix sum at leaves
//
__global__ void prescan (float *g_odata, float *g_idata, int n, bool log_space) {
    /*
    Parameters:
    g_odata: Device pointer to output array (exclusive prefix sums)
    g_idata: Device pointer to input array
    n: Number of elements (must be power of 2 and <= ELEMENTS_PER_BLOCK)
    log_space: If true, use log_add_exp instead of + (for numerical stability)

    This is Blelloch's classic parallel scan algorithm with bank conflict avoidance.
    */

    // Blelloch scan implementation with shared memory and bank conflict avoidance
    if (!log_space) {
        extern __shared__ float temp[]; //allocate shared memory

        int thid = threadIdx.x; //local thread ID
        int offset = 1; //offset for the up-sweep and down-sweep phases

        int ai = thid; //index for loading data into shared memory
        int bi = thid + (n / 2); //index for loading data into shared memory

        // Compute the bank offsets
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

        // Load input data into shared memory with bank conflict avoidance
        temp[ai + bankOffsetA] = g_idata[ai];
        temp[bi + bankOffsetB] = g_idata[bi];


        // Up-sweep (reduce) phase

        // Build a balanced binary tree bottom-up, computing partial sums
        //
        // Example with 8 elements [a,b,c,d,e,f,g,h]:
        // Start:    a    b    c    d    e    f    g    h
        // Level 0:  a   a+b   c   c+d   e   e+f   g   g+h   (4 threads)
        // Level 1:  a   a+b   c  a+b+c+d  e   e+f   g  e+f+g+h  (2 threads)
        // Level 2:  a   a+b   c  a+b+c+d  e   e+f   g  a+b+c+d+e+f+g+h  (1 thread)
        // Root (index 7) now contains the total sum
        //
        // d = number of active threads at this level (n/2, n/4, n/8, ...)
        // offset = spacing between elements being combined (1, 2, 4, 8, ...)
        // n = total number of threads (power of 2)
        for (int d = n >> 1; d > 0; d >>= 1) {

            // Synchronize to ensure previous level is complete before starting next
            __syncthreads();

            // Only first d threads are active at this level (half of previous level)
            if (thid < d) {

                // Compute indices of the two elements this thread will combine
                //
                // Left child index:  ai = offset * (2*thid + 1) - 1
                // Right child index: bi = offset * (2*thid + 2) - 1
                //
                // Example (offset=1, thread 0): ai=0, bi=1  (combine first two elements)
                // Example (offset=2, thread 0): ai=1, bi=3  (combine results from previous level)
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;

                // Compute the bank offsets
                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                // Add the values
                temp[bi + bankOffsetB] += temp[ai + bankOffsetA];

            }

            offset *= 2; // Double the offset so that it points to the next level


        }

        // Clear the last element (root of tree) to identity value
        // This converts the inclusive scan to exclusive scan
        // After up-sweep, root contains total sum - we don't want that in our output
        // Setting to 0 (identity) means when we propagate down, first element becomes 0
        if (thid == 0) {
            int bankOffset = CONFLICT_FREE_OFFSET(n - 1);
            temp[n - 1 + bankOffset] = 0;  // Identity for addition
        }

        // Down-sweep (distribute) phase

        // Unlike blelloch's, instead of going from lg(tree_level) - 1 to n-1 by 2^d+1.
        // We go from 1 to n/2 by doubling d at each stage
        for (int d = 1; d < n; d *= 2 ) {

            offset >>= 1; // Halve the offset
            __syncthreads();

            if (thid < d) {

                // Once again, similar to blelloch's algorithm
                // For ai, blelloch uses i + 2^d -1
                // However, i = offset * 2 * thid
                // and 2^d = offset
                // Therefore, offset * 2 * thid + offset -1 = offset * (2*thid +1) -1
                int ai = offset * (2 * thid + 1) - 1;

                // For bi, blelloch uses i + 2^(d+1) -1
                // However, i = offset * 2 * thid
                // and 2^(d+1) = offset * 2
                // Therefore, offset * 2 * thid + offset * 2 -1 = offset * (2*thid +2) -1
                int bi = offset * (2 * thid + 2) - 1;

                // Compute bank offsets to avoid shared memory conflicts
                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                // Perform the down-sweep operation:
                // 1. Save left child's old value
                // 2. Left child gets parent's value (from right position)
                // 3. Right child gets old left value + parent value
                //
                // This propagates partial sums: parent becomes left child,
                // and right child accumulates both parent and old left child
                float t = temp[ai + bankOffsetA];  // Save old left child

                // Left child = parent (swap)
                temp[ai + bankOffsetA] = temp[bi + bankOffsetB];

                // Right child = old left child + parent (accumulate)
                temp[bi + bankOffsetB] += t;

            }

        }

        // Final syncthreads to make sure all operations are done before writing back to global memory
        __syncthreads();

        // Write the results to global memory
        g_odata[ai] = temp[ai + bankOffsetA];
        g_odata[bi] = temp[bi + bankOffsetB];

    // Log space scan
    } else {
        extern __shared__ float temp[]; //allocate shared memory

        int thid = threadIdx.x; //local thread ID
        int offset = 1; //offset for the up-sweep and down-sweep phases

        int ai = thid; //index for loading data into shared memory
        int bi = thid + (n / 2); //index for loading data into shared memory

        // Compute the bank offsets
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

        // Load input data into shared memory with bank conflict avoidance
        temp[ai + bankOffsetA] = g_idata[ai];
        temp[bi + bankOffsetB] = g_idata[bi];

        // Up-sweep (reduce) phase

        // Same tree structure as normal space, but use log_add_exp instead of +
        // prevents numerical underflow when working with very small probabilities
        // log(a+b) = log_add_exp(log(a), log(b))
        for (int d = n >> 1; d > 0; d >>= 1) {

            __syncthreads();

            // Only first d threads are active at this level
            if (thid < d) {

                // Compute indices using same formula as normal space
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;

                // Apply bank conflict offset
                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                // Combine using log_add_exp: log(exp(a) + exp(b))
                temp[bi + bankOffsetB] = log_add_exp(temp[bi + bankOffsetB], temp[ai + bankOffsetA]);
            }

            // Double the offset
            offset *= 2;
        }

        // Clear the last element if this is the first thread
        if (thid == 0) {

            int bankOffset = CONFLICT_FREE_OFFSET(n - 1);
            // Clear the last element (Identity for log_add_exp is -inf)
            temp[n - 1 + bankOffset] = -INFINITY;
        }

        // Down-sweep (distribute) phase
        for (int d = 1; d < n; d *= 2 ) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads
            __syncthreads();

            if (thid < d) {

                // Compute indices
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;

                // Compute the bank offsets
                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                // swap the values using log_add_exp
                // t holds the value of temp[ai]
                float t = temp[ai + bankOffsetA];

                // temp[ai] = temp[bi]
                temp[ai + bankOffsetA] = temp[bi + bankOffsetB];

                // temp[bi] += t becomes temp[bi] = log_add_exp(temp[bi], t)
                temp[bi + bankOffsetB] = log_add_exp(temp[bi + bankOffsetB], t);
            }
        }

        // Final syncthreads to make sure all operations are done before writing back to global memory
        __syncthreads();

        // Write the results to global memory
        g_odata[ai] = temp[ai + bankOffsetA];
        g_odata[bi] = temp[bi + bankOffsetB];
    }

}


// Multi-Block Blelloch Scan - Parent Function
//
// For arrays larger than one block can handle, we process in three stages:
// Stage 1: Each block computes its local exclusive prefix sum
// Stage 2:   Scan the array of block totals
// Stage 3: Add scanned block totals to each block's output
//
// Example with 8 elements, 2 blocks of 4:
// Input:  [a, b, c, d | e, f, g, h]
//
// Stage 1:
//   Block 0: [0, a, a+b, a+b+c]     (local scan, store total = a+b+c+d)
//   Block 1: [0, e, e+f, e+f+g]     (local scan, store total = e+f+g+h)
//   Block sums: [a+b+c+d, e+f+g+h]
//
// Stage 2 (recursive on block sums):
//   Scanned block sums: [0, a+b+c+d]
//
// Stage 3 (add_block_sums):
//   Block 0: [0, a, a+b, a+b+c]     (unchanged, first block)
//   Block 1: [a+b+c+d, a+b+c+d+e, a+b+c+d+e+f, a+b+c+d+e+f+g]  (add scanned block sum)
//   Full output: [0, a, a+b, a+b+c, a+b+c+d, a+b+c+d+e, a+b+c+d+e+f, a+b+c+d+e+f+g]
void prescan_2048(float *d_odata, float *d_idata, int n, bool log_space, cudaStream_t stream = 0) {
    /*

    Parameters:
    d_odata: Device pointer to output array
    d_idata: Device pointer to input array

    n: Number of elements in the input array

    log_space: Boolean flag indicating whether to perform calculations in log space
    stream: CUDA stream for async execution (default: 0 for default stream)

    */

    const int THREADS_PER_BLOCK = 1024; // Maximum number of threads per block
    const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2; // Number of elements processed by each block

    // If the input size is less than or equal to ELEMENTS_PER_BLOCK, use single block prescan (base case)
    if (n <= ELEMENTS_PER_BLOCK) {

        // prescan kernel requires power-of-2 size, so round up
        int next_pow2 = next_power_of_2(n);

        // Calculate number of threads needed (must be power of 2)
        int threads = next_pow2 / 2;

        // Calculate shared memory size (+ bank conflict padding)
        int shared_mem_size = next_pow2 * sizeof(float) + (next_pow2 / NUM_BANKS) * sizeof(float);

        // Allocate padded input if needed
        if (n == next_pow2) {
            // Launch prescan kernel directly on the specified stream
            prescan<<<1, threads, shared_mem_size, stream>>>(d_odata, d_idata, next_pow2, log_space);

        } else {
            // Need to pad input to power of 2
            float *d_padded_input, *d_padded_output;
            cudaMalloc(&d_padded_input, next_pow2 * sizeof(float));
            cudaMalloc(&d_padded_output, next_pow2 * sizeof(float));

            // Copy input
            cudaMemcpy(d_padded_input, d_idata, n * sizeof(float), cudaMemcpyDeviceToDevice);

            // Pad with identity element: 0 for normal, -inf for log space
            if (log_space) {
                pad_array(d_padded_input, n, next_pow2, -INFINITY);
            } else {
                pad_array(d_padded_input, n, next_pow2, 0.0f);
            }

            // Run prescan on the specified stream
            prescan<<<1, threads, shared_mem_size, stream>>>(d_padded_output, d_padded_input, next_pow2, log_space);

            // Copy back only the valid results
            cudaMemcpy(d_odata, d_padded_output, n * sizeof(float), cudaMemcpyDeviceToDevice);

            // Free temporary padded memory
            cudaFree(d_padded_input);
            cudaFree(d_padded_output);
        }

        return;

    } else {
        // Recursive case

        // Calculate number of blocks needed
        int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

        // Allocate memory for block sums
        float *d_block_sums;
        float *d_scanned_block_sums;

        // Allocate device memory for block sums and scanned block sums
        cudaMalloc((void**)&d_block_sums, num_blocks * sizeof(float));
        cudaMalloc((void**)&d_scanned_block_sums, num_blocks * sizeof(float));

        // Calculate shared memory size (+ bank conflict padding)
        int shared_mem_size = ELEMENTS_PER_BLOCK * sizeof(float) + (ELEMENTS_PER_BLOCK / NUM_BANKS) * sizeof(float);

        // Launch prescan_blocks kernel to process each block and compute block sums, storing them in d_block_sums
        prescan_blocks<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_odata, d_idata, d_block_sums, n, log_space);



        // Recursively call prescan_2048 on block sums, storing the result in d_scanned_block_sums
        // Eventually, this will reach the base case, and we will have the scanned block sums
        // Note: num_blocks might not be power of 2, but prescan_2048 will handle it recursively
        // Pass stream for async execution
        prescan_2048(d_scanned_block_sums, d_block_sums, num_blocks, log_space, stream);

        // Launch add_block_sums kernel to add scanned block sums to each block's output
        add_block_sums<<<num_blocks, THREADS_PER_BLOCK, 0, stream>>>(d_odata, d_scanned_block_sums, n, log_space);

        // Free allocated device memory
        cudaFree(d_block_sums);
        cudaFree(d_scanned_block_sums);

    }


}


// Multi-block Blelloch Scan - Stage 1: Local Block Scans
// Stage 1:
//   - Block 0: [0, a, a+b, a+b+c]     (local scan, store total = a+b+c+d)
//   - Block 1: [0, e, e+f, e+f+g]     (local scan, store total = e+f+g+h)
//   - Block sums: [a+b+c+d, e+f+g+h]
// Each block computes its local Blelloch scan and stores its total.

__global__ void prescan_blocks(float *g_odata, float *g_idata, float *d_block_sums, int n, bool log_space) {
    /*
    Parameters:
    g_odata: Device pointer to output array (local block results)
    g_idata: Device pointer to input array
    d_block_sums: Device pointer to store each block's total sum
    n: Total number of elements in the input array
    log_space: Boolean flag for log-space computation

    Each block independently computes a local Blelloch scan and stores its total.
    */

    // If not doing calculations in log space
    if (!log_space) {
        extern __shared__ float temp[]; //allocate shared memory

        int thid = threadIdx.x; //local thread ID
        int blockStart = blockIdx.x * blockDim.x * 2; //starting index for this block (each block processes blockDim.x (1024) * 2 elements)

        int ai = thid; //index for loading data into shared memory
        int bi = thid + blockDim.x; //index for loading data into shared memory (second half)

        // Compute the bank offsets
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

        // Load data into shared memory (check bounds by comparing blockStart + ai/bi with n)
        temp[ai + bankOffsetA] = (blockStart + ai < n) ? g_idata[blockStart + ai] : 0;
        temp[bi + bankOffsetB] = (blockStart + bi < n) ? g_idata[blockStart + bi] : 0;

        // Up-sweep (reduce) phase
        int offset = 1;

        // For d, we start from blockDim.x and halve the number of threads at each stage
        // We choose blockDim.x because each block processes 1024 * 2 elements
        for (int d = blockDim.x; d > 0; d >>= 1) {

            __syncthreads(); // Synchronize threads

            // We only need to launch half the number of threads at each stage of the reduction
            if (thid < d) {

                // Compute the indices of the elements to be added
                int ai = (offset * (2 * thid + 1)) - 1;
                int bi = (offset * (2 * thid + 2)) - 1;

                // Compute the bank offsets
                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                // Add the values
                temp[bi + bankOffsetB] += temp[ai + bankOffsetA];

            }

            offset *= 2;

        }

        // Before clearing root, save the total sum for this block
        // This will be used in stage 2 to combine results across blocks
        if (thid == 0) {

            // Get index of last element in this block (root of reduction tree)
            int last = blockDim.x * 2 - 1;

            // Apply bank conflict offset
            int bankOffset = CONFLICT_FREE_OFFSET(last);

            // Store this block's total sum for later use
            // Example: If this block processed [a,b,c,d], root contains a+b+c+d
            d_block_sums[blockIdx.x] = temp[last + bankOffset];

            // Clear the root to identity for down-sweep (makes it exclusive scan)
            temp[last + bankOffset] = 0;
        }

        // Down-sweep (distribute) phase

        // For d, we go from 1 to blockDim.x by doubling d at each stage
        // where blockDim.x is half the number of elements processed by this block
        // blockDim.x * 2 is chosen because each block processes blockDim.x * 2 elements
        for (int d = 1; d < blockDim.x * 2; d *= 2 ) {

            // Halve the offset
            offset >>= 1;


            __syncthreads(); // Synchronize threads

            // We only need to launch half the number of threads at each stage of the reduction
            if (thid < d) {

                // Compute the indices
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;

                // Compute the bank offsets
                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                // swap the values
                // t holds the value of temp[ai]
                float t = temp[ai + bankOffsetA];

                // temp[ai] = temp[bi]
                temp[ai + bankOffsetA] = temp[bi + bankOffsetB];

                // temp[bi] += t
                temp[bi + bankOffsetB] += t;

            }

        }

        // Final syncthreads to make sure all operations are done before writing back to global memory
        __syncthreads();

        // Write the results to global memory if and only if within bounds
        // (blockStart + ai means the global index corresponding to local index ai in this block, which should be less than n, the total number of elements)
        if (blockStart + ai < n){
            g_odata[blockStart + ai] = temp[ai + bankOffsetA];
        }

        // Write the results to global memory if and only if within bounds
        if (blockStart + bi < n){
            g_odata[blockStart + bi] = temp[bi + bankOffsetB];
        }


    // If calculation in log space
    } else {

        extern __shared__ float temp[]; //allocate shared memory

        int thid = threadIdx.x; //local thread ID
        int blockStart = blockIdx.x * blockDim.x * 2; //starting index for this block

        int ai = thid; //index for loading data into shared memory
        int bi = thid + blockDim.x; //index for loading data into shared memory

        // Compute the bank offsets
        int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
        int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

        // Load data into shared memory
        temp[ai + bankOffsetA] = (blockStart + ai < n) ? g_idata[blockStart + ai] : -INFINITY;
        temp[bi + bankOffsetB] = (blockStart + bi < n) ? g_idata[blockStart + bi] : -INFINITY;

        // Up-sweep (reduce) phase
        int offset = 1;

        // For d, we start from blockDim.x and halve the number of threads at each stage
        for (int d = blockDim.x; d > 0; d >>= 1) {

            __syncthreads(); // Synchronize threads

            // We only need to launch half the number of threads at each stage of the reduction
            // since each thread handles two elements
            if (thid < d) {

                int ai = (offset * (2 * thid + 1)) - 1;
                int bi = (offset * (2 * thid + 2)) - 1;

                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                temp[bi + bankOffsetB] = log_add_exp(temp[bi + bankOffsetB], temp[ai + bankOffsetA]);

            }

            offset *= 2; // Double the offset to point to the next level
        }

        // clear the last element if this is the first thread
        if (thid == 0) {
            // Retrieve the last element (total sum for this block)
            int last = blockDim.x * 2 - 1;

            // Compute the bank offset for the last element
            int bankOffset = CONFLICT_FREE_OFFSET(last);

            // Store the total sum in the block sums array
            d_block_sums[blockIdx.x] = temp[last + bankOffset];

            // Clear the last element (Identity for log_add_exp is -inf)
            temp[last + bankOffset] = -INFINITY;
        }

        // Down-sweep (distribute) phase

        // For d, we go from 1 to blockDim.x by doubling d at each stage
        // where blockDim.x is half the number of elements processed by this block
        for (int d = 1; d < blockDim.x * 2; d *= 2 ) {

            offset >>= 1; // Halve the offset

            __syncthreads(); // Synchronize threads

            // We only need to launch half the number of threads at each stage of the reduction
            if (thid < d) {

                // Compute the indices
                int ai = offset * (2 * thid + 1) - 1;
                int bi = offset * (2 * thid + 2) - 1;

                // Compute the bank offsets
                int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
                int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

                // swap the values using log_add_exp
                // t holds the value of temp[ai]
                float t = temp[ai + bankOffsetA];

                // temp[ai] = temp[bi]
                temp[ai + bankOffsetA] = temp[bi + bankOffsetB];

                // temp[bi] += t becomes temp[bi] = log_add_exp(temp[bi], t)
                temp[bi + bankOffsetB] = log_add_exp(temp[bi + bankOffsetB], t);

            }

        }

        // Final syncthreads to make sure all operations are done before writing back to global memory
        __syncthreads();

        // Write the results to global memory if and only if within bounds
        if (blockStart + ai < n){
            g_odata[blockStart + ai] = temp[ai + bankOffsetA];
        }

        // Write the results to global memory if and only if within bounds
        if (blockStart + bi < n){
            g_odata[blockStart + bi] = temp[bi + bankOffsetB];
        }

    }

}

// Multi-Block Scan - Stage 3: Add Block Prefixes
//
// After local scans (Stage 1) and scanning block totals (Stage 2),
// this kernel adds the appropriate prefix to each block's results
//
// Stage 3 (this kernel):
//   Block 0: Add 0 -> [0, a, a+b, a+b+c]  (first block, no change)
//   Block 1: Add a+b+c+d -> [a+b+c+d, a+b+c+d+e, a+b+c+d+e+f, a+b+c+d+e+f+g]
//
// Final result is globally correct exclusive prefix sum!
//
// Without this step: [0, a, a+b, a+b+c, 0, e, e+f, e+f+g] -> exlusive scan means d would never show up, and other elements wouldn't add up correctly!
// With this step:    [0, a, a+b, a+b+c, a+b+c+d, a+b+c+d+e, ...]
//
__global__ void add_block_sums(float *g_odata, float *g_block_sums, int n, bool log_space) {
    /*
    Parameters:
    g_odata: Device pointer to output array (local scan results to be corrected)
    g_block_sums: Device pointer to scanned block sums (prefix to add to each block)
    n: Number of elements in the input array
    log_space: Boolean flag for log-space computation

    */


    // Normal space: Simple addition of block prefix
    if (!log_space){

        // Compute thread ID and this block's starting position in global array
        int thid = threadIdx.x;
        int blockStart = blockIdx.x * blockDim.x * 2;

        // First block (blockIdx.x == 0) needs no correction (its prefix is 0)
        // All other blocks need to add the sum of all previous blocks
        if (blockIdx.x > 0) {

            // Get the scanned sum of all blocks before this one
            // This is the value we need to add to every element in this block
            float blockSum = g_block_sums[blockIdx.x];

            // Compute the indices
            int ai = thid;
            int bi = thid + blockDim.x;

            // Add the block sum to the elements if within bounds
            if (blockStart + ai < n) {
                g_odata[blockStart + ai] += blockSum;
            }

            // Add the block sum to the elements if within bounds
            if (blockStart + bi < n) {
                g_odata[blockStart + bi] += blockSum;
            }

        }

    // If doing calculations in log space
    } else {

        // set thread ID and block start index
        int thid = threadIdx.x;
        int blockStart = blockIdx.x * blockDim.x * 2;

        // If not the first block, add the scanned block sum to each element in this block
        if (blockIdx.x > 0) {

            // Retrieve the block sum for this block
            float blockSum = g_block_sums[blockIdx.x];

            // Compute the indices
            int ai = thid;
            int bi = thid + blockDim.x;

            // Add the block sum to the elements if within bounds
            if (blockStart + ai < n) {
                g_odata[blockStart + ai] = log_add_exp(g_odata[blockStart + ai], blockSum);
            }

            // Add the block sum to the elements if within bounds
            if (blockStart + bi < n) {
                g_odata[blockStart + bi] = log_add_exp(g_odata[blockStart + bi], blockSum);
            }

        }

    }

}

/* -------------------------------------------------------------------------------------------------------------------------*/


// LSTM scan: parallel implementation of h_k = a_k * h_{k-1} + b_k
// Uses Blelloch algorithm with associative operation on (h, a) pairs
//
// LSTM Recurrence: h_k = a_k * h_{k-1} + b_k
//   where: h_k = hidden state at position k
//          a_k = forget gate at position k
//          b_k = input contribution at position k
//          h_{k-1} = hidden state at previous position k-1
//
// Associative Operation to combine adjacent positions (k-1) and k:
//   (h_{k-1}, a_{0...k-1}) ⊕ (b_k, a_k) = (a_k * h_{k-1} + b_k, a_k * a_{0...k-1})
//                                        = (h_k, a_{0...k})
//
// Identity: (0, 1) for normal space because h_k = 1 * 0 + b_k = b_k
//           (-inf, 0) for log space because log(exp(0) * exp(-inf) + exp(b)) = log(exp(b)) = b
//
// If we look at (h_{k-1}, a_{0...k-1}) ⊕ (b_k, a_k) from a matrix perspective:
//   | a_k  b_k |   *   | a_{0...k-1}  h_{k-1}  | = | a_k * a_{0...k-1}    a_k * h_{k-1} + b_k |
//   | 0     1  |   *   | 0            1        | = |      0                     1             |
//
// LSTM Parallel Scan Example (8 elements):
//
// Tree structure (Binary tree built by Blelloch algorithm):
// ========================================================
//
// The algorithm uses a  binary tree where each position stores partial scan results.
// Positions higher in the tree represent larger segments:
//
//                                  [7]
//                                 /   \
//                               /      \
//                            [3]        [7]
//                            /  \       / \
//                          [1]  [3]   [5] [7]
//                          / \  / \   / \  / \
//                         0  1  2  3  4  5 6  7
//
// Position meanings:
//   [7] = mathematical result for full segment [0-7]
//   [3] = result for segment [0-3], [5] = result for segment [4-5]
//   [1] = result for segment [0-1], [3] = result for segment [2-3], etc.
//
// Up-sweep traversal (bottom-up, building tree):
//   Level 0: Process pairs (0,1), (2,3), (4,5), (6,7) -> positions 1,3,5,7 updated
//   Level 1: Process (1,3), (5,7) -> positions 3,7 updated
//   Level 2: Process (3,7) -> position 7 updated (root)
//
// Down-sweep traversal (top-down, distributing values):
//   Level 2: Process pair (3,7) with 7 as parent
//   Level 1: Process pairs (1,3) and (5,7) where 3 and 7 are parents
//   Level 0: Process pairs (0,1), (2,3), (4,5), (6,7)
//
// Parent-child relationships in tree:
//   7 is parent of segment [4-7] (combines with 3 which has [0-3])
//   3 is parent of segment [2-3] (combines with 1 which has [0-1])
//   5 is parent of segment [4-5] (similar pattern)
//
// Given inputs (b values): [2, 1, 3, 1, 2, 1, 1, 2]
// Forget gates (a values): [0.5, 0.8, 0.6, 0.7, 0.9, 0.5, 0.8, 0.6]
//
//
// Sequential computation would be:
//   h[0] = a[0]*0 + b[0] = 0.5*0 + 2 = 2
//   h[1] = a[1]*h[0] + b[1] = 0.8*2 + 1 = 2.6
//   h[2] = a[2]*h[1] + b[2] = 0.6*2.6 + 3 = 4.56
//   h[3] = a[3]*h[2] + b[3] = 0.7*4.56 + 1 = 4.192
//   h[4] = a[4]*h[3] + b[4] = 0.9*4.192 + 2 = 5.773
//   h[5] = a[5]*h[4] + b[5] = 0.5*5.773 + 1 = 3.886
//   h[6] = a[6]*h[5] + b[6] = 0.8*3.886 + 1 = 4.109
//   h[7] = a[7]*h[6] + b[7] = 0.6*4.109 + 2 = 4.465
//   Result: h = [2, 2.6, 4.56, 4.192, 5.773, 3.886, 4.109, 4.465]
//   Result: a = [0.5, 0.4, 0.24, 0.168, 0.151, 0.0756, 0.0605, 0.0362]
//
// Parallel algorithm uses associative operation:
//   Combine segments: (h_left, a_left) ⊕ (b_right, a_right)
//   Result: (h_new, a_new) where:
//     h_new = a_right * h_left + b_right
//     a_new = a_right * a_left
//
//  Up-sweep (builds tree bottom-up):
//
// Level 0 (combine pairs, offset=1):
//   Positions: [0,1], [2,3], [4,5], [6,7]
//
//   Pair [0,1]: (2, 0.5) ⊕ (1, 0.8)
//     h = 0.8*2 + 1 = 2.6
//     a = 0.8*0.5 = 0.4
//   Result at [1]: (2.6, 0.4)
//
//   Pair [2,3]: (3, 0.6) ⊕ (1, 0.7)
//     h = 0.7*3 + 1 = 3.1
//     a = 0.7*0.6 = 0.42
//   Result at [3]: (3.1, 0.42)
//
//   Pair [4,5]: (2, 0.9) ⊕ (1, 0.5)
//     h = 0.5*2 + 1 = 2.0
//     a = 0.5*0.9 = 0.45
//   Result at [5]: (2.0, 0.45)
//
//   Pair [6,7]: (1, 0.8) ⊕ (2, 0.6)
//     h = 0.6*1 + 2 = 2.6
//     a = 0.6*0.8 = 0.48
//   Result at [7]: (2.6, 0.48)
//
// After Level 0: [(2,0.5), (2.6,0.4), (3,0.6), (3.1,0.42), (2,0.9), (2.0,0.45), (1,0.8), (2.6,0.48)]
// Important note in each index: left is h, right is a, but in memory, h and a are stored in separate arrays (s_h and s_a)
//
// Level 1 (offset=2):
//   Positions: [1,3], [5,7]
//
//   Pair [1,3]: (2.6, 0.4) ⊕ (3.1, 0.42)
//     h = 0.42*2.6 + 3.1 = 4.192
//     a = 0.42*0.4 = 0.168
//   Result at [3]: (4.192, 0.168)
//
//   Pair [5,7]: (2.0, 0.45) ⊕ (2.6, 0.48)
//     h = 0.48*2.0 + 2.6 = 3.56
//     a = 0.48*0.45 = 0.216
//   Result at [7]: (3.56, 0.216)
//
// After Level 1: [(2,0.5), (2.6,0.4), (3,0.6), (4.192,0.168), (2,0.9), (2.0,0.45), (1,0.8), (3.56,0.216)]
//
// Level 2 (offset=4):
//   Pair [3,7]: (4.192, 0.168) ⊕ (3.56, 0.216)
//     h = 0.216*4.192 + 3.56 = 4.465
//     a = 0.216*0.168 = 0.0363
//   Result at [7]: (4.465, 0.0363) <- ROOT (cumulative result for all elements)
//
// After Level 2: [(2,0.5), (2.6,0.4), (3,0.6), (4.192,0.168), (2,0.9), (2.0,0.45), (1,0.8), (4.465,0.0363)]
//
// Set root to identity: [7] = (0, 1)
// After root set: [(2,0.5), (2.6,0.4), (3,0.6), (4.192,0.168), (2,0.9), (2.0,0.45), (1,0.8), (0,1)]
//
// Down-sweep (distributes exclusive prefix values top-down):
//
// Level 2 (offset=4, d=1):
//   Process pair [3,7]:
//     temp = s_h[3] = 4.192, temp_a = s_a[3] = 0.168
//     s_h[3] = s_h[7] = 0 (left child gets parent's value)
//     s_a[3] = s_a[7] = 1
//     s_h[7] = 0.168*0 + 4.192 = 4.192 (right child combines)
//     s_a[7] = 0.168*1 = 0.168
//
// After Level 2 down: [(2,0.5), (2.6,0.4), (3,0.6), (0,1), (2,0.9), (2.0,0.45), (1,0.8), (4.192,0.168)]
//
// Level 1 (offset=2, d=2):
//   Process pair [1,3]:
//     s_h[1] = s_h[3] = 0,
//     s_a[1] = s_a[3] = 1
//     s_h[3] = 0.4*0 + 2.6 = 2.6
//     s_a[3] = 0.4*1 = 0.4
//   Process pair [5,7]:
//     s_h[5] = s_h[7] = 4.192
//     s_a[5] = s_a[7] = 0.168
//     s_h[7] = 0.45*4.192 + 2.0 = 3.886
//     s_a[7] = 0.45*0.168 = 0.0756
//
// After Level 1 down: [(2,0.5), (0,1), (3,0.6), (2.6,0.4), (2,0.9), (4.192,0.168), (1,0.8), (3.886,0.0756)]
//
// Level 0 (offset=1, d=4):
//   Process [0,1]:
//     s_h[0]=0
//     s_a[0]=1
//     s_h[1]=0.5*0+2=2
//     s_a[1]=0.5*1=0.5
//   Process [2,3]:
//     s_h[2]=2.6
//     s_a[2]=0.4
//     s_h[3]=0.6*2.6+3=4.56
//     s_a[3]=0.6*0.4=0.24
//   Process [4,5]:
//     s_h[4]=4.192
//     s_a[4]=0.168
//     s_h[5]=0.9*4.192+2=5.773
//     s_a[5]=0.9*0.168=0.151
//   Process [6,7]:
//     s_h[6]=3.886
//     s_a[6]=0.0756
//     s_h[7]=0.8*3.886+1=4.109
//     s_a[7]=0.8*0.0756=0.0605
//
// After down-sweep (exlsuive prefix): [(0,1), (2,0.5), (2.6,0.4), (4.56,0.24), (4.192,0.168), (5.773,0.151), (3.886,0.0756), (4.109,0.0605)]
//   These are h_{k-1} values (prefix up to but NOT including position k)
//
// Final step: Apply h[k] = a[k]*h_{k-1} + b[k] using ORIGINAL (b,a) values:
//   h[0] = 0.5*0 + 2 = 2
//   h[1] = 0.8*2 + 1 = 2.6
//   h[2] = 0.6*2.6 + 3 = 4.56
//   h[3] = 0.7*4.56 + 1 = 4.192
//   h[4] = 0.9*4.192 + 2 = 5.773
//   h[5] = 0.5*5.773 + 1 = 3.886
//   h[6] = 0.8*3.886 + 1 = 4.109
//   h[7] = 0.6*4.109 + 2 = 4.465
//
// Final output: [2, 2.6, 4.56, 4.192, 5.773, 3.886, 4.109, 4.465]
// for s_a: [1, 0.5, 0.4, 0.24, 0.168, 0.151, 0.0756, 0.0605]
//
__global__ void prescan_lstm(float *g_odata, float *g_b, float *g_a, int n, bool log_space) {
    /*
    Parameters:
    g_odata: Device pointer to output array (h_k values - hidden states)
    g_b: Device pointer to input array (b_k values - inputs to add)
    g_a: Device pointer to forget gate array (a_k values - multiplicative gates)
    n: Number of elements in the input array (must be power of 2 and <= ELEMENTS_PER_BLOCK)
    log_space: Boolean flag indicating whether to perform calculations in log space

    */

    // Shared memory allocation: [b values (inputs)][a values (forget gates)]
    // During scan, b values evolve into h values (hidden states) via h_k = a_k * h_{k-1} + b_k
    extern __shared__ float temp[];

    // find id of thread
    int thid = threadIdx.x;

    // Compute the indices for loading data into shared memory
    int ai = thid;
    int bi = thid + (n / 2);

    // Compute the bank offsets
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Shared memory layout:
    // s_h[] starts as b_k values (input contributions), becomes h_k (hidden states) during scan
    // s_a[] stores a_k values (forget gates) and their cumulative products
    float *s_h = temp;  // Will hold b->h transformation
    int a_base = n + (n / NUM_BANKS);
    float *s_a = &temp[a_base];


    // Load b_k values (input contributions) into shared memory
    s_h[ai + bankOffsetA] = g_b[ai];
    s_h[bi + bankOffsetB] = g_b[bi];


    // Load a_k values (forget gates) into shared memory
    s_a[ai + bankOffsetA] = g_a[ai];
    s_a[bi + bankOffsetB] = g_a[bi];

    // Initialize offset
    // Choose 1 as initial offset since we are dealing with pairs
    int offset = 1;

    // Up-sweep (reduce) phase

    // Combine adjacent positions using LSTM equation: h_k = a_k * h_{k-1} + b_k
    // Combines left segment [0...j] with right segment [j+1...k]
    // to produce (h_k, a_{0...k}) for the merged segment [0...k]
    if (!log_space) {

        // For each level of the tree
        for (int d = n >> 1; d > 0; d >>= 1) {

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = (offset * (2 * thid + 1)) - 1;
                int bi_idx = (offset * (2 * thid + 2)) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Get values at positions k-1 and k
                float h_k_1 = s_h[ai_idx + bankOffsetAi];     // h_{k-1}: hidden state up to k-1
                float a_0_k_1 = s_a[ai_idx + bankOffsetAi];   // a_{0...k-1}: forget product up to k-1
                float b_k = s_h[bi_idx + bankOffsetBi];       // b_k: input value at position k
                float a_k = s_a[bi_idx + bankOffsetBi];       // a_k: forget gate at position k

                // Apply LSTM equation: h_k = a_k * h_{k-1} + b_k
                float h_k = a_k * h_k_1 + b_k;
                float a_0_k = a_k * a_0_k_1;  // a_{0...k}: cumulative product through k

                // Store results back in shared memory on the right position
                s_h[bi_idx + bankOffsetBi] = h_k;
                s_a[bi_idx + bankOffsetBi] = a_0_k;
            }

            // Double the offset
            offset *= 2;
        }

    // If in log space
    } else {

        // Log space: multiplication becomes addition, addition becomes log_add_exp
        // For each level of the tree
        for (int d = n >> 1; d > 0; d >>= 1) {

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = (offset * (2 * thid + 1)) - 1;
                int bi_idx = (offset * (2 * thid + 2)) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);


                // Get values at positions k-1 and k (log space)
                float h_k_1 = s_h[ai_idx + bankOffsetAi];     // log(h_{k-1})
                float a_0_k_1 = s_a[ai_idx + bankOffsetAi];   // log(a_{0...k-1})


                // Load position k values
                float b_k = s_h[bi_idx + bankOffsetBi];       // log(b_k)
                float a_k = s_a[bi_idx + bankOffsetBi];       // log(a_k)

                // Log space LSTM: log(h_k) = log(a_k * h_{k-1} + b_k)
                //                           = log(a_k + h_{k-1}) + log(b_k)
                float h_k = log_add_exp(a_k + h_k_1, b_k);
                float a_0_k = a_k + a_0_k_1;  // log(a_k * a_{0...k-1}) = log(a_k) + log(a_{0...k-1})

                // Store results back in shared memory on the right position
                s_h[bi_idx + bankOffsetBi] = h_k;
                s_a[bi_idx + bankOffsetBi] = a_0_k;
            }

            // Double the offset
            offset *= 2;
        }
    }

    // Identity for h_k = a_k * h_{k-1} + b_k is (b=0, a=1)
    // This ensures: h_k = 1 * 0 + b_k = b_k (first element gets its b_k value)
    if (thid == 0) {

        // Compute the bank offset for the last element
        int bankOffset = CONFLICT_FREE_OFFSET(n - 1);

        // Set root to identity
        if (!log_space) {

            s_h[n - 1 + bankOffset] = 0.0f;  // Identity b: 0 (so h_k = a*0 + b = b)
            s_a[n - 1 + bankOffset] = 1.0f;  // Identity a: 1 (so h_k = 1*h + b = h + b)

        } else {

            s_h[n - 1 + bankOffset] = -INFINITY;  // Identity b: -inf (log of 0)
            s_a[n - 1 + bankOffset] = 0.0f;       // Identity a: 0 (log of 1)

        }
    }

    // Down-sweep (distribute) phase

    // Propagate partial results down the tree
    // basically calculates an exclusive scan using the associative operation defined above
    if (!log_space) {

        // For each level of the tree
        for (int d = 1; d < n; d *= 2) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Save old left child (this segment's local contribution)
                float b_local = s_h[ai_idx + bankOffsetAi];   // Local b from this segment
                float a_local = s_a[ai_idx + bankOffsetAi];   // Local forget product

                // Left child gets parent's value (prefix from all earlier segments)
                float h_k_1 = s_h[bi_idx + bankOffsetBi];     // h_{k-1} from parent
                float a_0_k_1 = s_a[bi_idx + bankOffsetBi];   // a_{0...k-1} from parent

                // Assign parent's value to left child within the segment
                s_h[ai_idx + bankOffsetAi] = h_k_1;
                s_a[ai_idx + bankOffsetAi] = a_0_k_1;

                // Right: Apply h_k = a_k * h_{k-1} + b_k combine parents prefix with local contribution
                // where h_{k-1} = h_k_1 (from right), a_k = a_local, b_k = b_local
                float h_k = a_local * h_k_1 + b_local;
                float a_0_k = a_local * a_0_k_1;

                // Store results back in right child position
                s_h[bi_idx + bankOffsetBi] = h_k;
                s_a[bi_idx + bankOffsetBi] = a_0_k;
            }
        }

    // If in log space
    } else {

        // For each level of the tree
        for (int d = 1; d < n; d *= 2) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Save old left child values (local contribution)
                float b_local = s_h[ai_idx + bankOffsetAi];   // log(b_local)
                float a_local = s_a[ai_idx + bankOffsetAi];   // log(a_local)

                // Get parent's value (prefix from earlier)
                float h_k_1 = s_h[bi_idx + bankOffsetBi];     // log(h_{k-1})
                float a_0_k_1 = s_a[bi_idx + bankOffsetBi];   // log(a_{0...k-1})

                // Left child gets parent's value
                s_h[ai_idx + bankOffsetAi] = h_k_1;
                s_a[ai_idx + bankOffsetAi] = a_0_k_1;

                // Right child: log(h_k) = log(a_local * h_{k-1} + b_local)
                //                       = log_add_exp(log(a_local) + log(h_{k-1}), log(b_local))
                float h_k = log_add_exp(a_local + h_k_1, b_local);
                float a_0_k = a_local + a_0_k_1;  // log space multiplication

                // Store results back in right child position
                s_h[bi_idx + bankOffsetBi] = h_k;
                s_a[bi_idx + bankOffsetBi] = a_0_k;
            }
        }
    }

    // Final syncthreads to make sure all operations are done before writing back to global memory
    __syncthreads();


    // Blelloch gives exclusive prefix (result up to but NOT including current position)
    // For LSTM we need inclusive (result including current position)
    // Apply equation one more time: h_k = a_k * h_{k-1} + b_k
    //   where h_{k-1} is the exclusive prefix result
    //   and (a_k, b_k) are the original values at position k

    // Reload original (b_k, a_k) values from equation h_k = a_k * h_{k-1} + b_k
    // Global memory doesn't use shared memory bank offsets
    float orig_b_left = g_b[ai];    // b_k (input value) at left position
    float orig_a_left = g_a[ai];   // a_k (forget gate) at left position
    float orig_b_right = g_b[bi];    // b_k (input value) at right position
    float orig_a_right = g_a[bi];   // a_k (forget gate) at right position

    // Get exclusive prefix values (these are h_{k-1} in the equation)
    float excl_h_left = s_h[ai + bankOffsetA];  // h_{k-1} at left position
    float excl_h_right = s_h[bi + bankOffsetB];  // h_{k-1} at right position

    // Compute inclusive by applying equation: h_k = a_k * h_{k-1} + b_k
    if (!log_space) {

        // Direct application: h_k = orig_a * excl_h + orig_b
        //                          = a_k   * h_{k-1} + b_k
        g_odata[ai] = orig_a_left * excl_h_left + orig_b_left;
        g_odata[bi] = orig_a_right * excl_h_right + orig_b_right;

    } else {

        // Log space: h_k = log(exp(a_k + h_{k-1}) + exp(b_k))
        g_odata[ai] = log_add_exp(orig_a_left + excl_h_left, orig_b_left);
        g_odata[bi] = log_add_exp(orig_a_right + excl_h_right, orig_b_right);

    }

}

// Multi-Block LSTM Scan Example (3 blocks of 4 elements each = 12 total):
//
// Block 0: b=[2,1,3,1], a=[0.5,0.8,0.6,0.7]
// Block 1: b=[2,1,1,2], a=[0.9,0.5,0.8,0.6]
// Block 2: b=[1,3,2,1], a=[0.7,0.4,0.9,0.5]
//
// Stage 1: Each block computes local scan and stores totals
// =========================================================
//
// Block 0 local scan (same process as single-block example above):
//   h[0]=2, h[1]=2.6, h[2]=4.56, h[3]=4.192
//   Final: h_sum[0] = 4.192
//   Total forget product: a_prod[0] = 0.5*0.8*0.6*0.7 = 0.168
//   Stored: d_block_sums[0] = 4.192, d_block_a_products[0] = 0.168
//
// Block 1 local scan (starts fresh with h[-1]=0):
//   h[0]=2, h[1]=2.0, h[2]=2.6, h[3]=3.56
//   Final: h_sum[1] = 3.56
//   Total forget product: a_prod[1] = 0.9*0.5*0.8*0.6 = 0.216
//   Stored: d_block_sums[1] = 3.56, d_block_a_products[1] = 0.216
//
// Block 2 local scan:
//   h[0]=1, h[1]=3.4, h[2]=5.06, h[3]=3.53
//   Final: h_sum[2] = 3.53
//   Total forget product: a_prod[2] = 0.7*0.4*0.9*0.5 = 0.126
//   Stored: d_block_sums[2] = 3.53, d_block_a_products[2] = 0.126
//
// Arrays after Stage 1:
//   d_block_sums = [4.192, 3.56, 3.53]
//   d_block_a_products = [0.168, 0.216, 0.126]
//
// Stage 2: Recursive LSTM scan on block totals
// ==============================================
//
// This treats each block as *one* timestep in a meta-LSTM:
//   prescan_lstm_2048(d_scanned_block_sums, d_block_sums, d_block_a_products, 3, log_space)
//
// Input: b_meta=[4.192, 3.56, 3.53], a_meta=[0.168, 0.216, 0.126]
//
// Meta-scan computes (inclusive):
//   H[0] = 4.192 (first block's total)
//   H[1] = 0.216 * 4.192 + 3.56 = 4.465
//     (block 1 sees block 0's output, scaled by block 1's total forget product)
//   H[2] = 0.126 * 4.465 + 3.53 = 4.093
//     (block 2 sees combined output of blocks 0+1, scaled by block 2's forget product)
//
// Result: d_scanned_block_sums = [4.192, 4.465, 4.093]
//
// Usage in add_block_sums_lstm (via blockIdx_offset - 1 indexing):
//   Block 0: skipped (no previous blocks)
//   Block 1: reads g_block_sums[0] = 4.192 (cumulative from block 0)
//   Block 2: reads g_block_sums[1] = 4.465 (cumulative from blocks 0+1)
//
// Stage 3: Propagate cross-block contributions
// ============================================
//
// Block 0: Skip (already correct, adds 0)
//
// Block 1: Must add h_prev=4.192 from all previous blocks
//   But each position in block 1 needs DIFFERENT contribution:
//
//   Position 0: h[4] = 0.9 * 4.192 + 2 = 5.773
//   Position 1: h[5] = 0.5 * 5.773 + 1 = 3.886
//     (note: h_prev scaled by prefix product a[4]*a[5] = 0.9*0.5 = 0.45)
//   Position 2: h[6] = 0.8 * 3.886 + 1 = 4.109
//   Position 3: h[7] = 0.6 * 4.109 + 2 = 4.465
//
//   To compute this efficiently, add_block_sums_lstm:
//     1. Computes prefix products: [0.9, 0.45, 0.36, 0.216]
//     2. Scales h_prev by each: [3.773, 1.886, 1.509, 0.906]
//     3. Adds to local results: [5.773, 3.886, 4.109, 4.465]
//
// Block 2: Similar process with h_prev=4.465
//   Prefix products: [0.7, 0.28, 0.252, 0.126]
//   Scaled h_prev: [3.126, 1.250, 1.125, 0.563]
//   Final results: [4.125, 4.65, 6.185, 4.093]
//
// Final: [2, 2.6, 4.56, 4.192, 5.773, 3.886, 4.109, 4.465, 4.125, 4.65, 6.185, 4.093]
//
// LSTM scan for arbitrary sizes (hierarchical decomposition)
// Only works for 1D arrays
void prescan_lstm_2048(float *d_odata, float *d_b, float *d_a, int n, bool log_space, cudaStream_t stream) {
    /*

    Parameters:
    d_odata: Device pointer to output array
    d_b: Device pointer to input array (b values)

    d_a: Device pointer to forget gate array (a values)

    n: Number of elements in the input array

    log_space: Boolean flag indicating whether to perform calculations in log space
    stream: CUDA stream for async execution

    */

    const int THREADS_PER_BLOCK = 1024; // Maximum number of threads per block
    const int ELEMENTS_PER_BLOCK = THREADS_PER_BLOCK * 2; // Number of elements processed by each block

    // If the input size is less than or equal to ELEMENTS_PER_BLOCK, use single block prescan (base case)
    if (n <= ELEMENTS_PER_BLOCK) {

        // prescan_lstm kernel requires power-of-2 size, so round up
        int next_pow2 = next_power_of_2(n);

        // Calculate number of threads needed (must be power of 2)
        int threads = next_pow2 / 2;

        // Calculate shared memory size (data + forget gates + bank conflict padding)
        int shared_mem_size = (next_pow2 * 2 + next_pow2 * 2 / NUM_BANKS) * sizeof(float);

        // Allocate padded input if needed
        if (n == next_pow2) {
            // Launch prescan_lstm kernel directly on the specified stream
            prescan_lstm<<<1, threads, shared_mem_size, stream>>>(d_odata, d_b, d_a, next_pow2, log_space);

        } else {

            // Need to pad input to power of 2
            float *d_padded_b, *d_padded_a, *d_padded_output;
            cudaMalloc(&d_padded_b, next_pow2 * sizeof(float));
            cudaMalloc(&d_padded_a, next_pow2 * sizeof(float));
            cudaMalloc(&d_padded_output, next_pow2 * sizeof(float));

            // Copy input
            cudaMemcpy(d_padded_b, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_padded_a, d_a, n * sizeof(float), cudaMemcpyDeviceToDevice);

            // Pad with identity elements: 0 for normal input, 1.0 for normal forget; -inf for log input, 0.0 for log forget
            if (!log_space) {
                // Pad input with 0.0 for normal space
                pad_array(d_padded_b, n, next_pow2, 0.0f);

                // Forget gate identity is 1.0 for normal space because it multiplies the previous hidden state in a way that does not change it
                pad_array(d_padded_a, n, next_pow2, 1.0f);

            } else {
                // Pad input with -INFINITY for log space as identity for log_add_exp is -inf
                pad_array(d_padded_b, n, next_pow2, -INFINITY);

                // Forget gate identity in log space is 0.0 because log_space turns multiplication into addition!
                pad_array(d_padded_a, n, next_pow2, 0.0f);
            }

            // Run prescan on the specified stream
            prescan_lstm<<<1, threads, shared_mem_size, stream>>>(d_padded_output, d_padded_b, d_padded_a, next_pow2, log_space);

            // Copy back only the valid results
            cudaMemcpy(d_odata, d_padded_output, n * sizeof(float), cudaMemcpyDeviceToDevice);

            // Free temporary padded memory
            cudaFree(d_padded_b);
            cudaFree(d_padded_a);
            cudaFree(d_padded_output);
        }

        return;

    // Else, recursive case
    } else {

        // Calculate number of blocks needed
        int num_blocks = (n + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;

        // Pad input to multiple of ELEMENTS_PER_BLOCK to avoid partial blocks
        int padded_n = num_blocks * ELEMENTS_PER_BLOCK;

        // Allocate padded arrays if needed
        float *d_padded_b, *d_padded_a, *d_padded_output;

        // If padding is needed because n is not a multiple of ELEMENTS_PER_BLOCK
        if (padded_n > n) {

            // Allocate padded memory
            cudaMalloc(&d_padded_b, padded_n * sizeof(float));
            cudaMalloc(&d_padded_a, padded_n * sizeof(float));
            cudaMalloc(&d_padded_output, padded_n * sizeof(float));

            // Copy original data
            cudaMemcpy(d_padded_b, d_b, n * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(d_padded_a, d_a, n * sizeof(float), cudaMemcpyDeviceToDevice);

            // Pad with identity elements
            // If not in log space
            if (!log_space) {
                // Pad input with 0.0 for normal space
                pad_array(d_padded_b, n, padded_n, 0.0f);

                // Forget gate identity is 1.0 for normal space
                pad_array(d_padded_a, n, padded_n, 1.0f);

            // If in log space
            } else {
                // Pad input with -INFINITY for log space
                pad_array(d_padded_b, n, padded_n, -INFINITY);

                // Forget gate identity in log space is 0.0
                pad_array(d_padded_a, n, padded_n, 0.0f);
            }

        // If no padding is needed
        } else {

            // No padding needed
            d_padded_b = d_b;
            d_padded_a = d_a;
            d_padded_output = d_odata;

        }

        // Calculate shared memory size (data + forget gates + bank conflict padding)
        int shared_mem_size = (ELEMENTS_PER_BLOCK * 2 + ELEMENTS_PER_BLOCK * 2 / NUM_BANKS) * sizeof(float);

        // Allocate memory for block sums and forget gate products
        float *d_block_sums, *d_block_a_products;
        cudaMalloc(&d_block_sums, num_blocks * sizeof(float));
        cudaMalloc(&d_block_a_products, num_blocks * sizeof(float));

        // Launch prescan_blocks_lstm kernel to process each block
        prescan_blocks_lstm<<<num_blocks, THREADS_PER_BLOCK, shared_mem_size>>>(d_padded_output, d_padded_b, d_padded_a, d_block_sums, d_block_a_products, padded_n, log_space);



        // Allocate memory for scanned block sums
        float *d_scanned_block_sums;
        cudaMalloc(&d_scanned_block_sums, num_blocks * sizeof(float));

        // Recursively scan the block sums (treating forget products as the new forget gates)
        // Pass stream for async execution
        prescan_lstm_2048(d_scanned_block_sums, d_block_sums, d_block_a_products, num_blocks, log_space, stream);

        // Copy scanned block sums back
        cudaMemcpy(d_block_sums, d_scanned_block_sums, num_blocks * sizeof(float), cudaMemcpyDeviceToDevice);

        // Release scanned block sums memory
        cudaFree(d_scanned_block_sums);

        // Launch add_block_sums_lstm kernel to propagate hidden states across blocks
        // The first block does not need any addition, so we launch num_blocks - 1
        // Shared memory needed for parallel prefix product of forget gates
        int add_shared_mem_size = ELEMENTS_PER_BLOCK * sizeof(float);
        add_block_sums_lstm<<<num_blocks - 1, THREADS_PER_BLOCK, add_shared_mem_size, stream>>>(d_padded_output, d_padded_a, d_block_sums, d_block_a_products, padded_n, log_space);




        // Copy back only valid results if we padded
        // Otherwise we never allocated extra memory
        if (padded_n > n) {

            // Copy back only the valid results
            // If we didn't do this, the output would contain extra padded elements
            cudaMemcpy(d_odata, d_padded_output, n * sizeof(float), cudaMemcpyDeviceToDevice);

            // Free temporary padded memory
            cudaFree(d_padded_b);
            cudaFree(d_padded_a);
            cudaFree(d_padded_output);

        }

        // Free allocated device memory
        cudaFree(d_block_sums);
        cudaFree(d_block_a_products);
    }
}


// Multi-block LSTM scan: compute local scans and store block summaries
// PARALLEL VERSION using Blelloch algorithm
// Implements: h_k = a_k * h_{k-1} + b_k in parallel across blocks
__global__ void prescan_blocks_lstm(float *g_odata, float *g_b, float *g_a,
                                     float *d_block_sums, float *d_block_a_products, int n, bool log_space) {
    /*
    Parameters:
    g_odata: Device pointer to output array (h_k values)
    g_b: Device pointer to input array (b_k values)
    g_a: Device pointer to forget gate array (a_k values)
    d_block_sums: Device pointer to array to store block sums (last h value per block)
    d_block_a_products: Device pointer to array to store cumulative a per block
    n: Number of elements in the input array
    log_space: Boolean flag indicating whether to perform calculations in log space

    Each block computes h_k = a_k * h_{k-1} + b_k locally using parallel Blelloch scan.
    Stores block summary for later hierarchical combination across blocks.
    */

    extern __shared__ float temp[];

    // Calculate block size
    // (times 2 because each thread processes 2 elements)
    int blockSize = blockDim.x * 2;

    // Shared memory layout: s_h for h values, s_a for cumulative forget products
    float *s_h = temp;

    // second half of shared memory for a values ( padding for bank conflicts )
    float *s_a = &temp[blockSize + blockSize / NUM_BANKS];

    // Compute this block's starting index in global arrays
    int blockStart = blockIdx.x * blockSize;

    // find thread ID
    int thid = threadIdx.x;

    // Compute indices for loading data
    int ai = thid;
    int bi = thid + blockDim.x;

    // Compute bank offsets
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Load data into shared memory
    s_h[ai + bankOffsetA] = g_b[blockStart + ai];
    s_a[ai + bankOffsetA] = g_a[blockStart + ai];

    // second element
    s_h[bi + bankOffsetB] = g_b[blockStart + bi];
    s_a[bi + bankOffsetB] = g_a[blockStart + bi];

    // Declare offset
    int offset = 1;

    // Up-sweep (reduce) phase
    if (!log_space) {

        // For each level of the tree
        for (int d = blockDim.x; d > 0; d >>= 1) {

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = (offset * (2 * thid + 1)) - 1;
                int bi_idx = (offset * (2 * thid + 2)) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Get values at positions k-1 and k
                float h_k_1 = s_h[ai_idx + bankOffsetAi];
                float a_0_k_1 = s_a[ai_idx + bankOffsetAi];
                float b_k = s_h[bi_idx + bankOffsetBi];
                float a_k = s_a[bi_idx + bankOffsetBi];

                // h_k = a_k * h_{k-1} + b_k
                s_h[bi_idx + bankOffsetBi] = a_k * h_k_1 + b_k;
                s_a[bi_idx + bankOffsetBi] = a_k * a_0_k_1;
            }

            // Double the offset
            offset *= 2;
        }

    // If in log space
    } else {

        // For each level of the tree
        for (int d = blockDim.x; d > 0; d >>= 1) {

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = (offset * (2 * thid + 1)) - 1;
                int bi_idx = (offset * (2 * thid + 2)) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Get values at positions k-1 and k
                float h_k_1 = s_h[ai_idx + bankOffsetAi];
                float a_0_k_1 = s_a[ai_idx + bankOffsetAi];
                float b_k = s_h[bi_idx + bankOffsetBi];
                float a_k = s_a[bi_idx + bankOffsetBi];

                // Log space: log(h_k) = log_add_exp(log(a_k) + log(h_{k-1}), log(b_k))
                s_h[bi_idx + bankOffsetBi] = log_add_exp(a_k + h_k_1, b_k);
                s_a[bi_idx + bankOffsetBi] = a_k + a_0_k_1;
            }

            // Double the offset
            offset *= 2;
        }
    }

    // Synchronize threads before storing block sums
    __syncthreads();

    // Store the total (h, a) for this block BEFORE setting root to identity
    // This block is important for summing across blocks later
    if (thid == 0) {

        // Store the total (h, a) for this block
        int last_idx = (blockSize - 1) + CONFLICT_FREE_OFFSET(blockSize - 1);

        // Store the block's final h value and cumulative a value in global memory
        d_block_sums[blockIdx.x] = s_h[last_idx];
        d_block_a_products[blockIdx.x] = s_a[last_idx];

    }

    // Set root to identity
    if (thid == 0) {

        // Compute bank offset for the last element
        int bankOffset = CONFLICT_FREE_OFFSET(blockSize - 1);

        // If not in log space
        if (!log_space) {

            // Set root to identity (b=0, a=1)
            s_h[blockSize - 1 + bankOffset] = 0.0f;
            s_a[blockSize - 1 + bankOffset] = 1.0f;

        // Else in log space
        } else {

            // Set root to identity (b=-inf, a=0)
            s_h[blockSize - 1 + bankOffset] = -INFINITY;
            s_a[blockSize - 1 + bankOffset] = 0.0f;

        }
    }

    // Down-sweep (distribute) phase
    if (!log_space) {

        // For each level of the tree
        for (int d = 1; d < blockSize; d *= 2) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads
            __syncthreads();


            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);


                // Save old left child (local contribution)
                float b_local = s_h[ai_idx + bankOffsetAi];
                float a_local = s_a[ai_idx + bankOffsetAi];


                // Left child gets parent's value (exclusive prefix)
                s_h[ai_idx + bankOffsetAi] = s_h[bi_idx + bankOffsetBi];
                s_a[ai_idx + bankOffsetAi] = s_a[bi_idx + bankOffsetBi];

                // Right child: apply h_k = a_local * h_{k-1} + b_local
                float h_k_1 = s_h[bi_idx + bankOffsetBi];
                float a_0_k_1 = s_a[bi_idx + bankOffsetBi];

                // h_k = a_local * h_{k-1} + b_local
                s_h[bi_idx + bankOffsetBi] = a_local * h_k_1 + b_local;
                s_a[bi_idx + bankOffsetBi] = a_local * a_0_k_1;
            }
        }

    // If in log space
    } else {

        // For each level of the tree
        for (int d = 1; d < blockSize; d *= 2) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Save old left child (local contribution)
                float b_local = s_h[ai_idx + bankOffsetAi];
                float a_local = s_a[ai_idx + bankOffsetAi];

                // Left child gets parent's value (exclusive prefix)
                s_h[ai_idx + bankOffsetAi] = s_h[bi_idx + bankOffsetBi];
                s_a[ai_idx + bankOffsetAi] = s_a[bi_idx + bankOffsetBi];

                // Right child: apply h_k = a_local * h_{k-1} + b_local
                float h_k_1 = s_h[bi_idx + bankOffsetBi];
                float a_0_k_1 = s_a[bi_idx + bankOffsetBi];

                // Log space: log(h_k) = log_add_exp(log(a_local) + log(h_{k-1}), log(b_local))
                s_h[bi_idx + bankOffsetBi] = log_add_exp(a_local + h_k_1, b_local);
                s_a[bi_idx + bankOffsetBi] = a_local + a_0_k_1;
            }
        }
    }

    // Final syncthreads to ensure all operations are done before writing back to global memory
    __syncthreads();

    // Apply h_k = a_k * h_{k-1} + b_k one final time
    // to convert from exclusive to inclusive scan
    // where h_{k-1} is exclusive prefix, and (a_k, b_k) are originals

    // Reload original (b_k, a_k) values from global memory
    // Global memory is immutable, so this always gives us the original values
    float orig_b_left = g_b[blockStart + ai];    // b_k (input value) at left position
    float orig_a_left = g_a[blockStart + ai];   // a_k (forget gate) at left position
    float orig_b_right = g_b[blockStart + bi];    // b_k (input value) at right position
    float orig_a_right = g_a[blockStart + bi];   // a_k (forget gate) at right position

    // Get exclusive prefix values (these are h_{k-1} in the equation)
    float h_k_1_left = s_h[ai + bankOffsetA];  // h_{k-1} at left position
    float h_k_1_right = s_h[bi + bankOffsetB];  // h_{k-1} at right position

    // If not in log space
    if (!log_space) {

        // h_k = a_k * h_{k-1} + b_k
        g_odata[blockStart + ai] = orig_a_left * h_k_1_left + orig_b_left;
        g_odata[blockStart + bi] = orig_a_right * h_k_1_right + orig_b_right;

    } else {

        // Log space: h_k = log_add_exp(log(a_k) + log(h_{k-1}), log(b_k))
        g_odata[blockStart + ai] = log_add_exp(orig_a_left + h_k_1_left, orig_b_left);
        g_odata[blockStart + bi] = log_add_exp(orig_a_right + h_k_1_right, orig_b_right);

    }

}


// Kernel to propagate cumulative hidden states across blocks
// Parallel version: Uses parallel prefix product for forget gates within each block
// Applies equation h_k = a_k * h_{k-1} + b_k across block boundaries
// If we didn't do this, each block would be independent and not account for previous blocks
// Basically, very similar to Blelloch's scan but now we have to apply the LSTM equation
__global__ void add_block_sums_lstm(float *g_odata, float *g_a,
                                     float *g_block_sums, float *g_block_a_products, int n, bool log_space) {
    /*
    Parameters:

    g_odata: Device pointer to output array (local h_k to be updated with previous blocks)
    g_a: Device pointer to forget gate array (a_k values)
    g_block_sums: Device pointer to scanned block sums (cumulative h from previous blocks)
    g_block_a_products: Device pointer to block forget gate products
    n: Number of elements in the input array
    log_space: Boolean flag indicating whether to perform calculations in log space

    */

    // Shared memory for parallel prefix product of forget gates
    extern __shared__ float s_forget_prefix[];

    int blockIdx_offset = blockIdx.x + 1;  // Skip first block

    int blockStart = blockIdx_offset * blockDim.x * 2; // Start index for this block

    int blockSize = blockDim.x * 2; // Number of elements per block

    // Declare thread ID
    int thid = threadIdx.x;

    // Compute indices for loading data
    int ai = thid;
    int bi = thid + blockDim.x;

    // Compute global indices
    int global_ai = blockStart + ai;
    int global_bi = blockStart + bi;

    // Get cumulative hidden state from previous blocks
    float cum_h = g_block_sums[blockIdx_offset - 1];

    // Load forget gates into shared memory for parallel prefix product
    float f_ai = (global_ai < n) ? g_a[global_ai] : (log_space ? 0.0f : 1.0f);
    float f_bi = (global_bi < n) ? g_a[global_bi] : (log_space ? 0.0f : 1.0f);

    // Store in shared memory
    s_forget_prefix[ai] = f_ai;
    s_forget_prefix[bi] = f_bi;

    // Synchronize to ensure all forget gates are loaded into shared memory
    __syncthreads();

    if (!log_space) {

        // Up-sweep (reduce) phase for multiplication

        // Initialize offset
        int offset = 1;

        // For each level of the tree
        // After we do this, s_forget_prefix[i] = f_0 * f_1 * ... * f_i
        for (int d = blockSize >> 1; d > 0; d >>= 1) {

            // Synchronize threads to ensure previous level is done
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Multiply values
                s_forget_prefix[bi_idx] *= s_forget_prefix[ai_idx];

            }

            // Double the offset
            offset *= 2;
        }

        // Set last element to identity (1.0) and save total
        __syncthreads();

        // Down-sweep (distribute) phase for exclusive prefix product
        // When this is done, s_forget_prefix[i] = f_0 * f_1 * ... * f_{i-1}
        for (int d = 1; d < blockSize; d *= 2) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads to ensure previous level is done
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Create temp variable to hold left child's value
                float t = s_forget_prefix[ai_idx];

                // Swap the left child's value with the right child's value
                s_forget_prefix[ai_idx] = s_forget_prefix[bi_idx];

                // Update right child's value
                s_forget_prefix[bi_idx] *= t;

            }
        }

        // Final synchronize to ensure down-sweep is done
        __syncthreads();

        // Convert exclusive to inclusive: multiply by original forget gate
        float incl_ai = s_forget_prefix[ai] * f_ai;
        float incl_bi = s_forget_prefix[bi] * f_bi;

        // Apply LSTM formula: h_i = f_prefix_i * cum_h + local_h_i
        // where cum_h is cumulative hidden from previous blocks
        // and local_h_i is the current g_odata value
        // Apply LSTM formula for ai
        if (global_ai < n) {

            // g_odata[global_ai] = incl_ai * cum_h + g_odata[global_ai]
            g_odata[global_ai] = incl_ai * cum_h + g_odata[global_ai];
        }

        // Apply LSTM formula for bi
        if (global_bi < n) {

            // g_odata[global_bi] = incl_bi * cum_h + g_odata[global_bi]
            g_odata[global_bi] = incl_bi * cum_h + g_odata[global_bi];
        }

    // If in log space
    } else {

        // Log space: multiplication becomes addition
        int offset = 1;


        // Up-sweep (reduce) phase for addition
        for (int d = blockSize >> 1; d > 0; d >>= 1) {

            // Synchronize threads to ensure previous level is done
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Add values for log space
                s_forget_prefix[bi_idx] += s_forget_prefix[ai_idx];

            }

            // Double the offset
            offset *= 2;
        }

        // Synchronize before down-sweep to ensure up-sweep is done
        __syncthreads();

        // Down-sweep (distribute) phase for exclusive prefix sum
        for (int d = 1; d < blockSize; d *= 2) {
            offset >>= 1;

            // Synchronize threads to ensure previous level is done
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Swap and add for log space
                float t = s_forget_prefix[ai_idx];
                s_forget_prefix[ai_idx] = s_forget_prefix[bi_idx];
                s_forget_prefix[bi_idx] += t;

            }
        }

        // Final synchronize to ensure down-sweep is done
        __syncthreads();

        // Convert exclusive to inclusive: add original log forget gate
        float incl_ai = s_forget_prefix[ai] + f_ai;
        float incl_bi = s_forget_prefix[bi] + f_bi;

        // Apply LSTM formula in log space
        if (global_ai < n) {

            // g_odata[global_ai] = log_add_exp(incl_ai + cum_h, g_odata[global_ai])
            g_odata[global_ai] = log_add_exp(incl_ai + cum_h, g_odata[global_ai]);

        }
        if (global_bi < n) {

            // g_odata[global_bi] = log_add_exp(incl_bi + cum_h, g_odata[global_bi])
            g_odata[global_bi] = log_add_exp(incl_bi + cum_h, g_odata[global_bi]);

        }
    }
}



/*-------------------------------------------------------------------------------------------------------------------------*/


// Batched LSTM Parallel Scan - Process Multiple Sequences Simultaneously
////
// Memory Layout: [batch][seq_length][hidden_size] flattened
// Example: batch=2, seq=4, hidden=3
//   Indices: [b0s0h0, b0s0h1, b0s0h2, b0s1h0, b0s1h1, b0s1h2, ..., b1s3h2]
//
//   Grid dimension: (batch_size, hidden_size)
//   Each CUDA block processes one (batch_idx, hidden_idx) sequence independently
//   Within each block, threads cooperate to scan the sequence in parallel
//
// batch=2, seq=4, hidden=2
// ============================================
//
// Input tensor shape: [2, 4, 2]
//   Sample 0, Hidden 0: b=[1,2,1,1], a=[0.5,0.8,0.6,0.9]
//   Sample 0, Hidden 1: b=[2,1,3,1], a=[0.7,0.5,0.8,0.4]
//   Sample 1, Hidden 0: b=[1,1,2,1], a=[0.9,0.6,0.7,0.5]
//   Sample 1, Hidden 1: b=[3,1,1,2], a=[0.4,0.8,0.6,0.9]
//
// Flattened memory (stride=hidden=2):
//   Position: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
//   b_values: [1, 2, 2, 1, 1, 3, 1, 1, 1, 3, 1, 1, 2, 1, 1, 2]
//   a_values: [.5,.7,.8,.5,.6,.8,.9,.4,.9,.4,.6,.8,.7,.6,.5,.9]
//
// CUDA Grid: 2x2 = 4 blocks
//   Block (0,0): Sample 0, Hidden 0 → processes positions [0, 2, 4, 6]
//   Block (0,1): Sample 0, Hidden 1 → processes positions [1, 3, 5, 7]
//   Block (1,0): Sample 1, Hidden 0 → processes positions [8, 10, 12, 14]
//   Block (1,1): Sample 1, Hidden 1 → processes positions [9, 11, 13, 15]
//
// Block (0,0) computation (Sample 0, Hidden 0):
//   base_offset = 0*4*2 + 0 = 0
//   Loads from positions: 0, 2, 4, 6 (stride=2)
//   Local arrays: b=[1,2,1,1], a=[0.5,0.8,0.6,0.9]
//   Parallel scan (same as single-block LSTM):
//     h[0] = 0.5*0 + 1 = 1
//     h[1] = 0.8*1 + 2 = 2.8
//     h[2] = 0.6*2.8 + 1 = 2.68
//     h[3] = 0.9*2.68 + 1 = 3.412
//   Writes to positions: 0, 2, 4, 6
//
// Block (0,1) computation (Sample 0, Hidden 1):
//   base_offset = 0*4*2 + 1 = 1
//   Loads from positions: 1, 3, 5, 7 (stride=2)
//   Local arrays: b=[2,1,3,1], a=[0.7,0.5,0.8,0.4]
//   Scan result:
//     h[0] = 0.7*0 + 2 = 2
//     h[1] = 0.5*2 + 1 = 2.0
//     h[2] = 0.8*2.0 + 3 = 4.6
//     h[3] = 0.4*4.6 + 1 = 2.84
//   Writes to positions: 1, 3, 5, 7
//
// Block (1,0) computation (Sample 1, Hidden 0):
//   base_offset = 1*4*2 + 0 = 8
//   Loads from positions: 8, 10, 12, 14 (stride=2)
//   Local arrays: b=[1,1,2,1], a=[0.9,0.6,0.7,0.5]
//   Scan result:
//     h[0] = 0.9*0 + 1 = 1
//     h[1] = 0.6*1 + 1 = 1.6
//     h[2] = 0.7*1.6 + 2 = 3.12
//     h[3] = 0.5*3.12 + 1 = 2.56
//   Writes to positions: 8, 10, 12, 14
//
// Block (1,1) computation (Sample 1, Hidden 1):
//   base_offset = 1*4*2 + 1 = 9
//   Loads from positions: 9, 11, 13, 15 (stride=2)
//   Local arrays: b=[3,1,1,2], a=[0.4,0.8,0.6,0.9]
//   Scan result:
//     h[0] = 0.4*0 + 3 = 3
//     h[1] = 0.8*3 + 1 = 3.4
//     h[2] = 0.6*3.4 + 1 = 3.04
//     h[3] = 0.9*3.04 + 2 = 4.736
//   Writes to positions: 9, 11, 13, 15
//
// Final output tensor [2, 4, 2] = [batch, seq, hidden]:
//   Sample 0, timestep 0: [1.0, 2.0]    (hidden dims 0 and 1)
//   Sample 0, timestep 1: [2.8, 2.0]
//   Sample 0, timestep 2: [2.68, 4.6]
//   Sample 0, timestep 3: [3.412, 2.84]
//   Sample 1, timestep 0: [1.0, 3.0]
//   Sample 1, timestep 1: [1.6, 3.4]
//   Sample 1, timestep 2: [3.12, 3.04]
//   Sample 1, timestep 3: [2.56, 4.736]
//
// Each block operates on a strided 1D sequence, all in parallel.
//
__global__ void prescan_lstm_batched(
    float *g_odata,      // Output: [batch * seq_length * hidden] flattened
    float *g_b,      // Input:  [batch * seq_length * hidden] flattened (b_k values)
    float *g_a,     // Forget: [batch * seq_length * hidden] flattened (a_k values)
    int batch_size,
    int seq_length,
    int hidden_size,
    bool log_space
) {
    /*

    Parameters:
    g_odata: Device pointer to output array (h_k values)
    g_b: Device pointer to input array (b_k values)
    g_a: Device pointer to forget gate array (a_k values)
    batch_size: Number of sequences in the batch
    seq_length: Length of sequence (MUST be power of 2 - padding done by host)
    hidden_size: Size of the hidden dimension
    log_space: Boolean flag indicating whether to perform calculations in log space

    */

    // Each CUDA block handles one (batch_idx, hidden_idx) pair
    int batch_idx = blockIdx.x;
    int hidden_idx = blockIdx.y;

    // Bounds check
    if (batch_idx >= batch_size || hidden_idx >= hidden_size) return;

    // Memory layout: [batch][seq][hidden] flattened
    // Each (batch_idx, hidden_idx) pair has its own sequence of length seq_length
    int base_offset = batch_idx * (seq_length * hidden_size) + hidden_idx;
    int stride = hidden_size;

    // Shared memory - seq_length is guaranteed to be power of 2
    extern __shared__ float shared_mem[];

    // Shared memory layout: s_h for h values, s_a for cumulative forget products
    float *s_h = shared_mem;
    int a_base = seq_length + (seq_length / NUM_BANKS);
    float *s_a = &shared_mem[a_base];

    // Thread ID
    int thid = threadIdx.x;

    // Compute indices for first element
    int ai = thid;

    // Compute index for second element
    int bi = thid + (seq_length / 2);

    // Compute bank offsets
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    // Load b_k values (input contributions) into shared memory
    s_h[ai + bankOffsetA] = g_b[base_offset + ai * stride];
    s_h[bi + bankOffsetB] = g_b[base_offset + bi * stride];

    // Load a_k values (forget gates) into shared memory
    s_a[ai + bankOffsetA] = g_a[base_offset + ai * stride];
    s_a[bi + bankOffsetB] = g_a[base_offset + bi * stride];

    // Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    // Declare offset
    int offset = 1;

    // Up-sweep (reduce) phase

    // If not in log space
    if (!log_space) {

        // For each level of the tree
        for (int d = seq_length >> 1; d > 0; d >>= 1) {

            // Synchronize threads to ensure previous level is done
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = (offset * (2 * thid + 1)) - 1;
                int bi_idx = (offset * (2 * thid + 2)) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Get values at positions k-1 and k
                float h_k_1 = s_h[ai_idx + bankOffsetAi];
                float a_0_k_1 = s_a[ai_idx + bankOffsetAi];
                float b_k = s_h[bi_idx + bankOffsetBi];
                float a_k = s_a[bi_idx + bankOffsetBi];

                // h_k = a_k * h_{k-1} + b_k
                s_h[bi_idx + bankOffsetBi] = a_k * h_k_1 + b_k;
                s_a[bi_idx + bankOffsetBi] = a_k * a_0_k_1;
            }

            // Double the offset
            offset *= 2;
        }

    // Else if in log space
    } else {

        // For each level of the tree
        for (int d = seq_length >> 1; d > 0; d >>= 1) {

            // Synchronize threads to ensure previous level is done
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = (offset * (2 * thid + 1)) - 1;
                int bi_idx = (offset * (2 * thid + 2)) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Get values at positions k-1 and k
                float h_k_1 = s_h[ai_idx + bankOffsetAi];
                float a_0_k_1 = s_a[ai_idx + bankOffsetAi];
                float b_k = s_h[bi_idx + bankOffsetBi];
                float a_k = s_a[bi_idx + bankOffsetBi];

                // Log space: log(h_k) = log_add_exp(log(a_k) + log(h_{k-1}), log(b_k))
                // clamp to prevent accumulation overflow (values grow by ~log(seq_len) over sequence)
                float h_k = log_add_exp(a_k + h_k_1, b_k);
                s_h[bi_idx + bankOffsetBi] = fminf(fmaxf(h_k, -15.0f), 15.0f);
                s_a[bi_idx + bankOffsetBi] = fminf(fmaxf(a_k + a_0_k_1, -15.0f), 15.0f);
            }

            // Double the offset
            offset *= 2;
        }
    }

    // Synchronize to ensure up-sweep is done
    __syncthreads();

    // Clear the last element (root of tree) to identity for down-sweep
    // This converts from reduction to exclusive scan
    if (thid == 0) {

        // Calculate the last index in the sequence
        int last_idx = seq_length - 1;
        int bankOffset = CONFLICT_FREE_OFFSET(last_idx);

        // If not in log space
        if (!log_space) {

            // Set root to identity (b=0, a=1)
            s_h[last_idx + bankOffset] = 0.0f;
            s_a[last_idx + bankOffset] = 1.0f;

        // Else in log space
        } else {

            // Set root to identity (b=-inf, a=0)
            s_h[last_idx + bankOffset] = -INFINITY;
            s_a[last_idx + bankOffset] = 0.0f;
        }
    }

    // Down-sweep (distribute) phase
    // Traverse back down tree to compute exclusive prefix for each position


    // If not in log space
    if (!log_space) {

        // For each level of the tree
        for (int d = 1; d < seq_length; d *= 2) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Save old left child (local contribution)
                float b_local = s_h[ai_idx + bankOffsetAi];
                float a_local = s_a[ai_idx + bankOffsetAi];

                // Left child gets parent's value (exclusive prefix)
                s_h[ai_idx + bankOffsetAi] = s_h[bi_idx + bankOffsetBi];
                s_a[ai_idx + bankOffsetAi] = s_a[bi_idx + bankOffsetBi];

                // Right child: apply h_k = a_local * h_{k-1} + b_local
                float h_k_1 = s_h[bi_idx + bankOffsetBi];
                float a_0_k_1 = s_a[bi_idx + bankOffsetBi];

                // h_k = a_local * h_{k-1} + b_local
                s_h[bi_idx + bankOffsetBi] = a_local * h_k_1 + b_local;
                s_a[bi_idx + bankOffsetBi] = a_local * a_0_k_1;
            }
        }

    // If in log space
    } else {

        // For each level of the tree
        for (int d = 1; d < seq_length; d *= 2) {

            // Halve the offset
            offset >>= 1;

            // Synchronize threads
            __syncthreads();

            // Only threads less than d participate
            if (thid < d) {

                // Compute indices
                int ai_idx = offset * (2 * thid + 1) - 1;
                int bi_idx = offset * (2 * thid + 2) - 1;

                // Compute bank offsets
                int bankOffsetAi = CONFLICT_FREE_OFFSET(ai_idx);
                int bankOffsetBi = CONFLICT_FREE_OFFSET(bi_idx);

                // Save old left child (local contribution)
                float b_local = s_h[ai_idx + bankOffsetAi];
                float a_local = s_a[ai_idx + bankOffsetAi];

                // Left child gets parent's value (exclusive prefix)
                s_h[ai_idx + bankOffsetAi] = s_h[bi_idx + bankOffsetBi];
                s_a[ai_idx + bankOffsetAi] = s_a[bi_idx + bankOffsetBi];

                // Right child: apply h_k = a_local * h_{k-1} + b_local
                float h_k_1 = s_h[bi_idx + bankOffsetBi];
                float a_0_k_1 = s_a[bi_idx + bankOffsetBi];

                // Log space: log(h_k) = log_add_exp(log(a_local) + log(h_{k-1}), log(b_local))
                // clamp to prevent accumulation overflow
                float h_k = log_add_exp(a_local + h_k_1, b_local);
                s_h[bi_idx + bankOffsetBi] = fminf(fmaxf(h_k, -15.0f), 15.0f);
                s_a[bi_idx + bankOffsetBi] = fminf(fmaxf(a_local + a_0_k_1, -15.0f), 15.0f);
            }
        }
    }

    // Final syncthreads to ensure all operations are done before writing back to global memory
    __syncthreads();

    // Apply h_k = a_k * h_{k-1} + b_k one final time
    // to convert from exclusive to inclusive scan
    // where h_{k-1} is exclusive prefix, and (a_k, b_k) are originals

    // Reload original (b_k, a_k) values from global memory
    // Global memory is immutable, so this always gives us the original values
    float orig_b_left = g_b[base_offset + ai * stride];    // b_k (input value) at left position
    float orig_a_left = g_a[base_offset + ai * stride];   // a_k (forget gate) at left position
    float orig_b_right = g_b[base_offset + bi * stride];    // b_k (input value) at right position
    float orig_a_right = g_a[base_offset + bi * stride];   // a_k (forget gate) at right position

    // Get exclusive prefix values (these are h_{k-1} in the equation)
    float excl_h_left = s_h[ai + bankOffsetA];  // h_{k-1} at left position
    float excl_h_right = s_h[bi + bankOffsetB];  // h_{k-1} at right position

    // If not in log space
    if (!log_space) {

        // h_k = a_k * h_{k-1} + b_k
        g_odata[base_offset + ai * stride] = orig_a_left * excl_h_left + orig_b_left;
        g_odata[base_offset + bi * stride] = orig_a_right * excl_h_right + orig_b_right;

    } else {

        // Log space: h_k = log_add_exp(log(a_k) + log(h_{k-1}), log(b_k))
        // clamp final output to prevent exp() explosion later (exp(15) ≈ 3.3M, exp(10) ≈ 22K)
        float h_left = log_add_exp(orig_a_left + excl_h_left, orig_b_left);
        float h_right = log_add_exp(orig_a_right + excl_h_right, orig_b_right);
        g_odata[base_offset + ai * stride] = fminf(fmaxf(h_left, -15.0f), 15.0f);
        g_odata[base_offset + bi * stride] = fminf(fmaxf(h_right, -15.0f), 15.0f);

    }
}

// Host function for batched LSTM scan (single-block only for now)
void prescan_lstm_batched_2048(
    float *d_odata,
    float *d_b,
    float *d_a,
    int batch_size,
    int seq_length,
    int hidden_size,
    bool log_space,
    cudaStream_t stream
) {
    const int MAX_THREADS_PER_BLOCK = 1024;
    const int MAX_ELEMENTS_PER_BLOCK = MAX_THREADS_PER_BLOCK * 2;

    // note: seq_length must be power of 2 (enforced by C++ wrapper padding)
    // Equation for mfcc sequence length: seq_len = ((frequency - seconds) / hop_length) + 1
    if (seq_length <= MAX_ELEMENTS_PER_BLOCK) {
        // Single-block case: each block processes one (sample, hidden) pair

        // Grid: one block per (sample, hidden) pair
        dim3 grid(batch_size, hidden_size);

        // Number of threads = seq_length / 2 (each thread processes 2 elements)
        int threads_per_block = seq_length / 2;

        // Shared memory: data + forget gates + bank conflict padding for both
        int shared_mem_size = (seq_length * 2 + seq_length * 2 / NUM_BANKS) * sizeof(float);

        // Launch batched kernel on the specified stream
        prescan_lstm_batched<<<grid, threads_per_block, shared_mem_size, stream>>>(
            d_odata, d_b, d_a,
            batch_size, seq_length, hidden_size,
            log_space
        );

    // Else if multi-block case
    } else {

        // Multi-block case not implemented in this example
        // Would require additional logic to handle block sums across multiple blocks
        throw std::runtime_error("Multi-block batched LSTM scan not implemented.");

    }
}


/*-------------------------------------------------------------------------------------------------------------------------*/

// Initialize stream pool once
void init_streams() {

    // If not initialized yet
    if (!streams_initialized) {

        // For each stream
        for (int i = 0; i < NUM_STREAMS; i++) {
            // Create CUDA stream
            cudaStreamCreate(&stream_pool[i]);
        }

        // Mark as initialized
        streams_initialized = true;
    }
}

// Cleanup stream pool (call at program exit)
void destroy_streams() {

    // If initialized
    if (streams_initialized) {

        // For each stream
        for (int i = 0; i < NUM_STREAMS; i++) {

            // Destroy CUDA stream
            cudaStreamDestroy(stream_pool[i]);

        }

        // Mark as uninitialized
        streams_initialized = false;
    }
}

// Get stream from pool using round-robin
cudaStream_t get_stream() {

    // Ensure streams are initialized
    init_streams();

    // Return next stream in round-robin fashion
    return stream_pool[(stream_counter++) % NUM_STREAMS];

}

/*-------------------------------------------------------------------------------------------------------------------------*/

// Wrapper function for prescan_2048 to be called from Python
void prescan_2048_wrapper(float* d_odata, float* d_idata, int n, bool log_space)
{
    // Get stream from pool (efficient reuse)
    cudaStream_t stream = get_stream();

    // Launch kernel asynchronously on the stream
    prescan_2048(d_odata, d_idata, n, log_space, stream);

}

// Synchronous version (for backward compatibility, older version)
void prescan_2048_wrapper_sync(float* d_odata, float* d_idata, int n, bool log_space)
{
    cudaStream_t stream = get_stream();
    prescan_2048(d_odata, d_idata, n, log_space, stream);
    cudaStreamSynchronize(stream);
}


// Wrapper function for prescan_lstm_2048 to be called from Python
void prescan_lstm_2048_wrapper(float* d_odata, float* d_b, float* d_a, int n, bool log_space)
{
    // Get stream from pool (efficient reuse)
    cudaStream_t stream = get_stream();

    // Launch kernel asynchronously on the stream
    prescan_lstm_2048(d_odata, d_b, d_a, n, log_space, stream);

}

// Synchronous version (for backward compatibility, older version)
void prescan_lstm_2048_wrapper_sync(float* d_odata, float* d_b, float* d_a, int n, bool log_space)
{
    cudaStream_t stream = get_stream();
    prescan_lstm_2048(d_odata, d_b, d_a, n, log_space, stream);
    cudaStreamSynchronize(stream);
}

// Wrapper function for batched LSTM scan to be called from Python
void prescan_lstm_batched_2048_wrapper(float* d_odata, float* d_b, float* d_a, int batch_size, int seq_length, int hidden_size, bool log_space)
{
    // Get stream from pool (efficient reuse)
    cudaStream_t stream = get_stream();

    // Launch kernel asynchronously on the stream
    prescan_lstm_batched_2048(d_odata, d_b, d_a,
                                  batch_size, seq_length, hidden_size, log_space, stream);

}

// Synchronous version (for backward compatibility, older version)
void prescan_lstm_batched_2048_wrapper_sync(float* d_odata, float* d_b, float* d_a, int batch_size, int seq_length, int hidden_size, bool log_space)
{
    cudaStream_t stream = get_stream();
    prescan_lstm_batched_2048(d_odata, d_b, d_a,
                                  batch_size, seq_length, hidden_size, log_space, stream);
    cudaStreamSynchronize(stream);
}
