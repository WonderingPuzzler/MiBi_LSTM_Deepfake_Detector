from DeepFake_Detector_Imports import *

class ParallelScanLSTM(torch.autograd.Function):
    """
    Class implementing parallel scan LSTM using a custom autograd function.
    By creating an autograd Function, we can define both forward and backward passes
    for gradient computation through the parallel scan.

    """

    @staticmethod
    def forward(ctx, b_values: torch.Tensor, a_values: torch.Tensor, h0: torch.Tensor, log_space: bool) -> torch.Tensor:

        """
        Forward pass of parallel scan LSTM.

        Parameters:
            ctx: Autograd context for saving tensors
            b_values: Input tensor (b values), shape (batch, seq, hidden)
            a_values: Forget gates (a values), shape (batch, seq, hidden)
            h0: Initial hidden state, shape (batch, 1, hidden)
            log_space: Whether to use log-space computation for forward pass

        Returns:
            Output tensor with computed hidden states, shape (batch, seq, hidden)
        """

        # CUDA kernel requires tensors on GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        b_values = b_values.to(device)
        a_values = a_values.to(device)
        h0 = h0.to(device)

        batch_size, seq_length, hidden_size = b_values.size()

        # Allocate buffers for values
        b_buffer = torch.zeros(batch_size, seq_length + 1, hidden_size, device=device)
        a_buffer = torch.zeros(batch_size, seq_length + 1, hidden_size, device=device)

        # The first sequence corresponds to initial hidden state h0
        b_buffer[:, 0:1, :] = h0

        # The other sequences correspond to b_values
        b_buffer[:, 1:, :] = b_values

        # The first sequence corresponds to initial hidden state h0 (just use identity for a to keep dimensions correct)
        a_buffer[:, 0:1, :] = 1.0 if not log_space else 0.0

        # The other sequences correspond to a_values
        a_buffer[:, 1:, :] = a_values

        # Set the a and b values up for the CUDA kerne;
        b_values = b_buffer
        a_values = a_buffer

        # Call the CUDA kernel for parallel scan
        output = prescan_2048_cuda.prescan_lstm_batched_2048(b_values, a_values, log_space)


        # take output excluding the initial h0
        # The output consists of the hidden states for all time steps including the initial state
        output = output[:, 1:, :]

        # Save for backward pass (include h0 for correct gradient computation)
        ctx.save_for_backward(b_values[:, 1:, :], a_values[:, 1:, :], output, h0)

        # Save non-tensor context
        ctx.log_space = log_space

        return output

    @staticmethod
    def backward(ctx, *grad_outputs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        """
        Backward pass for parallel scan LSTM.

        For normal-space: h[k] = a[k] * h[k-1] + b[k]
            ∂h[k]/∂a[k] = h[k-1]
            ∂h[k]/∂b[k] = 1
            ∂h[k]/∂h[k-1] = a[k]


        Implemented in CUDA/C++ for maximum speed.
        Always computed in normal space for numerical stability.
        """

        # Retrieve saved tensors
        b_fwd, a_fwd, h_fwd, h0 = ctx.saved_tensors
        log_space = ctx.log_space

        # Gradient from next layer
        grad_h = grad_outputs[0].contiguous()

        # Move tensors to correct device
        device = grad_h.device

        b_fwd: torch.Tensor = b_fwd.to(device)
        a_fwd: torch.Tensor = a_fwd.to(device)
        h_fwd: torch.Tensor = h_fwd.to(device)
        h0: torch.Tensor = h0.to(device)

        if log_space:

            # Convert from log space to normal space using exp()
            # b_fwd is not needed for gradients in normal space (grad_b = total_grad)
            a_fwd: torch.Tensor = torch.exp(a_fwd)
            h_fwd: torch.Tensor = torch.exp(h_fwd)
            h0: torch.Tensor = torch.exp(h0)

            grad_b: torch.Tensor
            grad_a: torch.Tensor
            grad_h0: torch.Tensor

            # Construct h_prev in Python: [h0, h_1, ..., h_{T-1}]
            h_prev = torch.cat([h0, h_fwd[:, :-1, :]], dim=1)

            # Does in C++: backward scan, gradient computation
            grad_b, grad_a, grad_h0 = prescan_2048_cuda.prescan_lstm_full_backward(
                grad_h, a_fwd, h_prev
            )

        # Normal space backward
        else:

            grad_b: torch.Tensor
            grad_a: torch.Tensor
            grad_h0: torch.Tensor

            # Construct h_prev in Python: [h0, h_1, ..., h_{T-1}]
            h_prev = torch.cat([h0, h_fwd[:, :-1, :]], dim=1)

            # Does in C++: backward scan, gradient computation
            grad_b, grad_a, grad_h0 = prescan_2048_cuda.prescan_lstm_full_backward(
                grad_h, a_fwd, h_prev
            )


        return grad_b, grad_a, grad_h0, None