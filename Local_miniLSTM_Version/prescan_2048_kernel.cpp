#include <torch/extension.h>

/* ------------------------------------------------------------------------------------------------------------------- */

// Declarations of the CUDA wrapper functions (async versions)
void prescan_2048_wrapper(float*, float*, int, bool);
void prescan_lstm_2048_wrapper(float*, float*, float*, int, bool);
void prescan_lstm_batched_2048_wrapper(float*, float*, float*, int, int, int, bool);

// Synchronous versions (for backward compatibility)
void prescan_2048_wrapper_sync(float*, float*, int, bool);
void prescan_lstm_2048_wrapper_sync(float*, float*, float*, int, bool);
void prescan_lstm_batched_2048_wrapper_sync(float*, float*, float*, int, int, int, bool);

// Stream management functions
void init_streams();
void destroy_streams();

// Declarations of the main functions
torch::Tensor prescan_2048_cuda(torch::Tensor input, bool log_space);
torch::Tensor prescan_lstm_2048_cuda(torch::Tensor b_values, torch::Tensor a_values, bool log_space);
torch::Tensor prescan_lstm_batched_2048_cuda(torch::Tensor b_values, torch::Tensor a_values, bool log_space);

torch::Tensor prescan_lstm_backward_batched_2048_cuda(torch::Tensor grad_values, torch::Tensor weight_values);
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prescan_lstm_full_backward_cuda(
    torch::Tensor grad_h,
    torch::Tensor a_fwd,
    torch::Tensor h_prev
);

/* ------------------------------------------------------------------------------------------------------------------- */

// Fast bit manipulation for computing next power of 2
inline int next_power_of_2(int n) {

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

/* ------------------------------------------------------------------------------------------------------------------- */

// Parallel prefix sum (scan) for 1D tensor up to 2048 elements
torch::Tensor prescan_2048_cuda(torch::Tensor input, bool log_space) {

    // Input shape: (size)
    int size = input.size(0);
    const int ELEMENTS_PER_BLOCK = 2048;

    // For sizes <= 2048, handle padding to power of 2
    if (size <= ELEMENTS_PER_BLOCK) {

        // Find next power of 2
        int padded_size = next_power_of_2(size);

        // If already power of 2, no padding needed
        if (padded_size == size) {

            // Create an output tensor of the same size as input
            torch::Tensor output = torch::zeros_like(input);
            prescan_2048_wrapper(output.data_ptr<float>(), input.data_ptr<float>(), size, log_space);


            return output;

        }

        // Need padding
        // Options is used to create tensors with same dtype/device as input
        torch::TensorOptions options = torch::TensorOptions().dtype(input.dtype()).device(input.device());
        torch::Tensor padded_input;

        // Initialize with identity values
        if (!log_space) {

            padded_input = torch::zeros({padded_size}, options);

        } else {

            padded_input = torch::full({padded_size}, -std::numeric_limits<float>::infinity(), options);

        }

        // Copy original data
        torch::narrow(padded_input, 0, 0, size).copy_(input);

        // Create padded output
        torch::Tensor padded_output = torch::zeros({padded_size}, options);

        // Call kernel with padded data
        prescan_2048_wrapper(padded_output.data_ptr<float>(), padded_input.data_ptr<float>(), padded_size, log_space);


        // Return only valid portion
        return torch::narrow(padded_output, 0, 0, size).contiguous();

    } else {
        throw std::runtime_error("Regular scan for sizes > 2048 not implemented.");
    }
}

/* ------------------------------------------------------------------------------------------------------------------- */

// LSTM prescan for 1D sequences
// Input shape: (seq_length,) - one feature across up to 2048 timesteps
// To process multiple features or batches, requires external loop (use batched version instead)
// Uses LSTM recurrence: h[k] = a[k] * h[k-1] + b[k]
//   b_values = input contributions (what to add)
//   a_values = forget gates (how much to remember)
torch::Tensor prescan_lstm_2048_cuda(torch::Tensor b_values, torch::Tensor a_values, bool log_space) {

    int size = b_values.size(0);
    const int ELEMENTS_PER_BLOCK = 2048;

    // For sizes <= 2048, handle padding to power of 2
    if (size <= ELEMENTS_PER_BLOCK) {

        // Find next power of 2
        int padded_size = next_power_of_2(size);

        // If already power of 2, no padding needed
        if (padded_size == size) {

            // Create an output tensor of the same size as b_values
            torch::Tensor output = torch::zeros_like(b_values);
            prescan_lstm_2048_wrapper(output.data_ptr<float>(), b_values.data_ptr<float>(),
                                         a_values.data_ptr<float>(), size, log_space);


            return output;
        }

        // Need padding
        torch::TensorOptions options = torch::TensorOptions().dtype(b_values.dtype()).device(b_values.device());
        torch::Tensor padded_b;
        torch::Tensor padded_a;

        // Initialize with identity values for LSTM: (b=0, a=1) or (b=-inf, a=0)
        if (!log_space) {

            padded_b = torch::zeros({padded_size}, options);
            padded_a = torch::ones({padded_size}, options);

        } else {

            padded_b = torch::full({padded_size}, -std::numeric_limits<float>::infinity(), options);
            padded_a = torch::zeros({padded_size}, options);

        }

        // Copy original data
        torch::narrow(padded_b, 0, 0, size).copy_(b_values);
        torch::narrow(padded_a, 0, 0, size).copy_(a_values);

        // Create padded output
        torch::Tensor padded_output = torch::zeros({padded_size}, options);

        // Call kernel with padded data
        prescan_lstm_2048_wrapper(padded_output.data_ptr<float>(), padded_b.data_ptr<float>(),
                                     padded_a.data_ptr<float>(), padded_size, log_space);

        // Return only valid portion
        return torch::narrow(padded_output, 0, 0, size).contiguous();

    } else {
        throw std::runtime_error("LSTM scan for sizes > 2048 not implemented.");
    }
}

/* ------------------------------------------------------------------------------------------------------------------- */

// Batched LSTM prescan for 3D tensors: [batch, seq_length, hidden_size]
// Uses LSTM recurrence: h[k] = a[k] * h[k-1] + b[k]
//   b_values = input contributions (what to add)
//   a_values = forget gates (how much to remember)
torch::Tensor prescan_lstm_batched_2048_cuda(torch::Tensor b_values, torch::Tensor a_values, bool log_space) {

    // Get tensor dimensions for batching and padding
    int batch_size = b_values.size(0);
    int seq_length = b_values.size(1);
    int hidden_size = b_values.size(2);

    const int ELEMENTS_PER_BLOCK = 2048;  // Max elements per block for single-block kernel

    // For sequences <= 2048, use padding to power of 2
    if (seq_length <= ELEMENTS_PER_BLOCK) {

        // Find next power of 2 for padding
        int padded_seq_length = next_power_of_2(seq_length);

        // If already power of 2, no padding needed
        if (padded_seq_length == seq_length) {

            // Create an output tensor of the same size as b_values
            torch::Tensor output = torch::zeros_like(b_values);

            // Call the batched LSTM prescan wrapper function
            prescan_lstm_batched_2048_wrapper(
                output.data_ptr<float>(),
                b_values.data_ptr<float>(),
                a_values.data_ptr<float>(),
                batch_size, seq_length, hidden_size,
                log_space
            );


            return output;
        }

        // Need to pad to power of 2
        // Create padded tensors [batch, padded_seq_length, hidden]
        // Options is used to create tensors with same dtype/device as b_values
        torch::TensorOptions options = torch::TensorOptions().dtype(b_values.dtype()).device(b_values.device());

        // Initialize with identity values
        // For LSTM: identity is (b=0, a=1) in normal space, (b=-inf, a=0) in log space
        torch::Tensor padded_b;
        torch::Tensor padded_a;

        // Normal space padding
        if (!log_space) {

            padded_b = torch::zeros({batch_size, padded_seq_length, hidden_size}, options);
            padded_a = torch::ones({batch_size, padded_seq_length, hidden_size}, options);

        // Log space padding
        } else {

            padded_b = torch::full({batch_size, padded_seq_length, hidden_size}, -std::numeric_limits<float>::infinity(), options);
            padded_a = torch::zeros({batch_size, padded_seq_length, hidden_size}, options);

        }

        // Copy original data into padded tensors
        torch::narrow(padded_b, 1, 0, seq_length).copy_(b_values);
        torch::narrow(padded_a, 1, 0, seq_length).copy_(a_values);

        // Create padded output tensor
        torch::Tensor padded_output = torch::zeros({batch_size, padded_seq_length, hidden_size}, options);

        // Call the batched LSTM prescan wrapper function with padded data
        prescan_lstm_batched_2048_wrapper(
            padded_output.data_ptr<float>(),
            padded_b.data_ptr<float>(),
            padded_a.data_ptr<float>(),
            batch_size, padded_seq_length, hidden_size,
            log_space
        );


        // Extract and return only the valid (non-padded) portion
        return torch::narrow(padded_output, 1, 0, seq_length).contiguous();

    } else {

    // For sequences > 2048, throw error (not implemented)
        throw std::runtime_error("Batched LSTM scan for sequences > 2048 not implemented.");
    }
}

/* ------------------------------------------------------------------------------------------------------------------- */

// Complete backward pass for parallel scan LSTM
// For normal space: h[k] = a[k] * h[k-1] + b[k]
//   ∂h/∂a = h[k-1], ∂h/∂b = 1, ∂h/∂h[k-1] = a[k]
//
// grad_h: upstream gradient, shape [batch, seq, hidden]
// a_fwd: a values from forward pass, shape [batch, seq, hidden]
// h_prev: previous hidden states (constructed in Python), shape [batch, seq, hidden]
//
// Returns: tuple of (grad_b, grad_a, grad_h0)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> prescan_lstm_full_backward_cuda(
    torch::Tensor grad_h,
    torch::Tensor a_fwd,
    torch::Tensor h_prev
) {
    // Step 1: Run backward scan with a_fwd as weights
    // This computes total_grad_h where total[t] = grad[t] + a[t+1] * total[t+1]
    torch::Tensor total_grad_h = prescan_lstm_backward_batched_2048_cuda(grad_h, a_fwd);

    // Step 2: Compute final gradients
    // grad_a = total_grad_h * h_prev  (∂L/∂a = total * h[k-1])
    // grad_b = total_grad_h           (∂L/∂b = total * 1)
    torch::Tensor grad_a = total_grad_h * h_prev;
    torch::Tensor grad_b = total_grad_h;  // Alias total_grad_h since we modify it in-place last

    // Step 3: Compute gradient for h0
    // grad_h0 = total_grad_h[:, 0:1, :] * a_fwd[:, 0:1, :]
    torch::Tensor grad_h0 = (torch::narrow(total_grad_h, 1, 0, 1) * torch::narrow(a_fwd, 1, 0, 1)).contiguous();

    // Step 4: Clean up any NaN/Inf (replace with zeros) using in-place nan_to_num_
    grad_a.nan_to_num_(0.0, 0.0, 0.0);
    grad_b.nan_to_num_(0.0, 0.0, 0.0);
    grad_h0.nan_to_num_(0.0, 0.0, 0.0);

    return std::make_tuple(grad_b, grad_a, grad_h0);
}

/* ------------------------------------------------------------------------------------------------------------------- */

// Backward scan for gradient computation: total[t] = grad[t] + weight[t+1] * total[t+1]
// Computed by:
// 1. Flip grad and weight along sequence dimension
// 2. Shift weight (prepend identity, remove last)
// 3. Run forward scan
// 4. Flip result back
//
// Input shapes: [batch, seq_length, hidden_size]
// grad_values = upstream gradients
// weight_values = weights for accumulation (always in normal space)
torch::Tensor prescan_lstm_backward_batched_2048_cuda(torch::Tensor grad_values, torch::Tensor weight_values) {

    int batch_size = grad_values.size(0);
    int seq_length = grad_values.size(1);
    int hidden_size = grad_values.size(2);

    torch::TensorOptions options = torch::TensorOptions().dtype(grad_values.dtype()).device(grad_values.device());

    // Step 1: Flip both tensors along sequence dimension (dim=1)
    torch::Tensor grad_flipped = torch::flip(grad_values, {1}).contiguous();
    torch::Tensor weight_flipped = torch::flip(weight_values, {1}).contiguous();

    // Step 2: Shift weight - for backward scan total[t] = grad[t] + weight[t+1] * total[t+1]
    // After flip, we need: b_back[i] = grad_flipped[i], a_back[i] = weight_flipped[i-1]
    // So prepend identity (0 for multiplication) and remove last element
    // Normal space: identity for multiplication is 0 (since we want a*h + b with a=0 initially)
    torch::Tensor zero_slice = torch::zeros({batch_size, 1, hidden_size}, options);
    torch::Tensor weight_shifted = torch::cat({zero_slice, torch::narrow(weight_flipped, 1, 0, seq_length - 1)}, 1);

    // Step 3: Run forward scan with flipped/shifted inputs (always in normal space)
    // This computes the backward recurrence in the flipped domain
    torch::Tensor result_flipped = prescan_lstm_batched_2048_cuda(grad_flipped, weight_shifted, false);

    // Step 4: Flip result back to original order
    result_flipped = torch::flip(result_flipped, {1}).contiguous();

    return result_flipped;
}


/* ------------------------------------------------------------------------------------------------------------------- */


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("prescan_2048", &prescan_2048_cuda, "Parallel prefix sum on arrays up to 2048 elements");
    m.def("prescan_lstm_2048", &prescan_lstm_2048_cuda, "Parallel LSTM scan on arrays up to 2048 elements");
    m.def("prescan_lstm_batched_2048", &prescan_lstm_batched_2048_cuda, "Batched parallel LSTM scan on 3D tensors (seq_len <= 2048)");
    m.def("prescan_lstm_backward_batched_2048", &prescan_lstm_backward_batched_2048_cuda, "Backward scan for gradient accumulation - normal space only (seq_len <= 2048)");
    m.def("prescan_lstm_full_backward", &prescan_lstm_full_backward_cuda, "Complete backward pass for parallel scan LSTM (normal space only)");

    // Stream management functions
    m.def("init_streams", &init_streams, "Initialize CUDA stream pool for efficient reuse");
    m.def("destroy_streams", &destroy_streams, "Destroy CUDA stream pool (call at program exit)");
}