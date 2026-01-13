from DeepFake_Detector_Imports import *
from torch import nn
from parallel_scan_LSTM import ParallelScanLSTM

# Try to enable torch.compile if Triton is available
COMPILE_ENABLED = False
try:
    # Test with actual operation that would use Triton
    @torch.compile

    def _test_fn(x, y):
        """
        Simple test function to check torch.compile with Triton.
        Does similar math as gate normalization.
        """
        return torch.sigmoid(x) / (torch.sigmoid(x) + torch.sigmoid(y) + 1e-6)
    
    if torch.cuda.is_available():

        # Pick two random tensors
        x = torch.randn(2, 3, device='cuda')
        y = torch.randn(2, 3, device='cuda')

        # Run the compiled function
        result = _test_fn(x, y)

        result.sum().backward()  # Test backward pass too

        torch.cuda.synchronize()

        # If things worked correctly compile with torch.compile
        COMPILE_ENABLED = True
        print("torch.compile enabled - Triton available")

except Exception as e:
    COMPILE_ENABLED = False
    print(f"torch.compile disabled - Triton not available ({type(e).__name__})")

def safe_compile(function):
    """
    Decorator that applies torch.compile only if available.
    
    Parameters:
        function: The function to potentially compile.
        
    Returns:
        The compiled function if COMPILE_ENABLED, else the original function.
    
    """

    if COMPILE_ENABLED:
        return torch.compile(function)
    else:
        return function

@safe_compile
def g(x: torch.Tensor) -> torch.Tensor:
    """
    Continuous activation function ensuring h_tilde > 0 for log-space.
    From paper Appendix B.3, Listing 6.

    g(x) = x + 0.5 if x >= 0
            sigmoid(x) if x < 0

    Parameters:
        x: Input tensor

    Returns:
        Continuous activation function
    """

    # Where x >= 0, use x + 0.5, else use sigmoid(x)
    return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))

@safe_compile
def log_g(x: torch.Tensor) -> torch.Tensor:
    """
    Log of g function for numerical stability.
    From paper Appendix B.3, Listing 6.

    log_g(x) = log(x + 0.5) if x >= 0
                -softplus(-x) if x < 0

    Parameters:
        x: Input tensor

    Returns:
        Log of g function
    """

    # Where x >= 0, use log(x + 0.5), else use -softplus(-x)
    return torch.where(x >= 0, torch.log(F.relu(x) + 0.5 + 1e-6), -F.softplus(-x))


def compute_normalized_gates(f: torch.Tensor, i: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute normalized forget and input gates with kernel fusion.
    f_prime = f / (f + i), i_prime = i / (f + i)
    """

    # Fused computation
    denom = f + i + 1e-6
    return f / denom, i / denom


@safe_compile
def compute_gates_and_normalize(f_raw: torch.Tensor, i_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply sigmoid and normalize gates in a single fused operation.

    Parameters:
        f_raw: Raw forget gate values, shape (batch, seq, hidden)
        i_raw: Raw input gate values, shape (batch, seq, hidden)

    Returns:
        Tuple of normalized forget and input gates
    """

    f_sig = torch.sigmoid(f_raw)
    i_sig = torch.sigmoid(i_raw)

    denom = f_sig + i_sig + 1e-6
    return f_sig / denom, i_sig / denom


@safe_compile
def compute_log_gates(f_raw: torch.Tensor, i_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute log-space gates with kernel fusion.

    Parameters:
        f_raw: Raw forget gate values, shape (batch, seq, hidden)
        i_raw: Raw input gate values, shape (batch, seq, hidden)

    Returns:
        Tuple of log-space forget and input gates
    """

    # Calculate the diff needed for
    diff = F.softplus(-f_raw) - F.softplus(-i_raw)
    return -F.softplus(diff), -F.softplus(-diff)


class MiBiLSTM(nn.Module):
    """
    Class to create from-scratch MiBiLSTM model for DeepFake detection.
    Based off of the architecture proposed in:
    Leo Feng, Fredrik Tung, Mohamed Osama Ahmed, Yoshua Bengio, and Hossein Hajimirsadeghi. 2024.
    "Were RNNs All We Needed?" https://arxiv.org/pdf/2410.01201
    """

    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, mode: str, log_space: bool) -> None:
        """
        Initializes the MiBiLSTM model.

        Has three modes:
        - "Sequential": for sequential data input.
        - "Parallel": for parallel data input.
        - "Bi-Directional": for bi-directional parallel data input.

        Parameters:
            input_size (int): Size of the input features.
            hidden_size (int): Size of the hidden state.
            num_layers (int): Number of MiBiLSTM layers.
            dropout (float): Dropout rate between layers.
            mode (str): Mode of operation ("Sequential", "Parallel", "Bi-Directional").
            log_space (bool): Whether to use log-space computations in parallel scan.

        Returns:
            None
        """

        # Initialize the parent class
        super(MiBiLSTM, self).__init__()
        self.input_size: int = 0
        self.hidden_size: int = 0
        self.num_layers: int = 0
        self.dropout: float = 0.0

        # Setup mode and log-space flags
        self.mode: str = mode
        self.log_space: bool = log_space


        # Ensure proper input to the class
        self.set_input_size(input_size)
        self.set_hidden_size(hidden_size)
        self.set_num_layers(num_layers)
        self.set_dropout(dropout)
        self.set_mode(mode)

        # Create layers
        self.layers: nn.ModuleList = nn.ModuleList()

        # For each layer, create the necessary linear and dropout layers
        for layer in range(self.num_layers):

            # If this is the first layer, input size is input_size
            if layer == 0:
                layer_input_size: int = self.input_size

            # For subsequent layers, input size depends on mode
            else:

                # If Bi-Directional, input size doubles every layer
                if self.mode.lower().strip() == "bi-directional":
                    layer_input_size = self.hidden_size * 2

                # Otherwise, input size is hidden_size
                else:
                    layer_input_size = self.hidden_size

            # Create a layer dictionary to hold the components of this layer
            layer_dict: dict[str, nn.Module] = {
                'linear_fih': nn.Linear(layer_input_size, self.hidden_size * 3),
                'dropout': nn.Dropout(p=self.dropout)
           }

            # For bi-directional mode, add SEPARATE backward linear layers
            # This allows forward and backward passes to learn different representations
            if self.mode.lower().strip() == "bi-directional":
                layer_dict['linear_fih_backward'] = nn.Linear(layer_input_size, self.hidden_size * 3)

            # Append the layer dictionary as a ModuleDict to the layers list
            self.layers.append(nn.ModuleDict(layer_dict))

    def get_input_size(self) -> int:
        """
        Gets the input size.

        Parameters:
            None

        Returns:
            int: Input size.
        """

        # Validate input size
        if not isinstance(self.input_size, int) or self.input_size <= 0:
            raise TypeError("Input size must be a positive integer")
        else:
            return self.input_size

    def set_input_size(self, input_size: int) -> None:
        """
        Sets the input size.

        Parameters:
            input_size (int): Input size to set.

        Returns:
            None
        """

        # Validate input size
        if not isinstance(input_size, int) or input_size <= 0:
            raise TypeError("Input size must be a positive integer")
        else:
            self.input_size = input_size

    def get_hidden_size(self) -> int:
        """
        Gets the hidden size.

        Parameters:
            None

        Returns:
            int: Hidden size.
        """

        # Validate hidden size
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise TypeError("Hidden size must be a positive integer")
        else:
            return self.hidden_size

    def set_hidden_size(self, hidden_size: int) -> None:
        """
        Sets the hidden size.

        Parameters:
            hidden_size (int): Hidden size to set.

        Returns:
            None
        """

        # Validate hidden size
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise TypeError("Hidden size must be a positive integer")
        else:
            self.hidden_size = hidden_size

    def get_num_layers(self) -> int:
        """
        Gets the number of layers.

        Parameters:
            None

        Returns:
            int: Number of layers.
        """

        # Validate number of layers
        if not isinstance(self.num_layers, int) or self.num_layers <= 0:
            raise TypeError("Number of layers must be a positive integer")
        else:
            return self.num_layers

    def set_num_layers(self, num_layers: int) -> None:
        """
        Sets the number of layers.

        Parameters:
            num_layers (int): Number of layers to set.

        Returns:
            None
        """

        # Validate number of layers
        if not isinstance(num_layers, int) or num_layers <= 0:
            raise TypeError("Number of layers must be a positive integer")
        else:
            self.num_layers = num_layers

    def get_dropout(self) -> float:
        """
        Gets the dropout rate.
        Dropout decides how many neurons to drop between layers.

        Parameters:
            None

        Returns:
            float: Dropout rate.
        """

        # Validate dropout
        if not isinstance(self.dropout, float) or not (0.0 <= self.dropout <= 1.0):
            raise TypeError("Dropout must be a float greater than or equal to 0.0 and less than or equal to 1.0")
        else:
            return self.dropout


    def set_dropout(self, dropout: float) -> None:
        """
        Sets the dropout rate.
        Dropout decides how many neurons to drop between layers.

        Parameters:
            dropout (float): Dropout rate to set.

        Returns:
            None
        """

        # Check if dropout is float and in valid range
        if not isinstance(dropout, float) or not (0.0 <= dropout <= 1.0):
            raise TypeError("Dropout must be a float greater than or equal to 0.0 and less than or equal to 1.0")
        else:
            self.dropout = dropout


    def get_mode(self) -> str:
        """
        Gets the mode of operation.

        Parameters:
            None

        Returns:
            str: Mode of operation.
        """


        # Validate mode
        if not isinstance(self.mode, str) or self.mode.lower().strip() not in ["sequential", "parallel", "bi-directional"]:
            raise TypeError("Mode must be a string and one of 'Sequential', 'Parallel', or 'Bi-Directional'")
        else:
            return self.mode

    def set_mode(self, mode: str) -> None:
        """
        Sets the mode of operation.

        Parameters:
            mode (str): Mode of operation to set.

        Returns:
            None
        """

        # Strip whitespace and lower
        mode = mode.lower().strip()

        # Validate mode
        if not isinstance(mode, str) or mode.lower().strip() not in ["sequential", "parallel", "bi-directional"]:
            raise TypeError("Mode must be a string and one of 'Sequential', 'Parallel', or 'Bi-Directional'")
        else:
            self.mode = mode

    def get_log_space(self) -> bool:
        """
        Gets the log-space computation flag.

        Parameters:
            None

        Returns:
            bool: Log-space computation flag.
        """
        if not isinstance(self.log_space, bool):
            raise TypeError("log_space must be a boolean value")
        else:
            return self.log_space

    def set_log_space(self, log_space: bool) -> None:
        """
        Sets the log-space computation flag.

        Parameters:
            log_space (bool): Log-space computation flag to set.

        Returns:
            None
        """

        # Validate log_space
        if not isinstance(log_space, bool):
            raise TypeError("log_space must be a boolean value")
        else:
            self.log_space = log_space


    def parallel_scan_lstm(self, b_values: torch.Tensor, a_values: torch.Tensor, h0: torch.Tensor) -> torch.Tensor:
        """
        Performs parallel LSTM scan using CUDA kernel (O(log n) parallel algorithm).

        Computes h[k] = a[k] * h[k-1] + b[k] for the full sequence.

        - Forward: CUDA parallel scan kernel
        - Backward: CUDA backward parallel scan kernel

        Parameters:
            b_values (torch.Tensor): Input values (i' * h_tilde), shape (batch, seq, hidden)
            a_values (torch.Tensor): Forget gates (f'), shape (batch, seq, hidden)
            h0 (torch.Tensor): Initial hidden state, shape (batch, 1, hidden)

        Returns:
            torch.Tensor: Hidden states, shape (batch, seq, hidden)
        """

        # Use CUDA kernel for both forward and backward
        result = ParallelScanLSTM.apply(b_values, a_values, h0, self.log_space)

        # Check result type to ensure it's a tensor
        if not isinstance(result, torch.Tensor):
            raise TypeError("Result from parallel scan must be a torch.Tensor")

        return result



    def forward(self, batch_data: torch.Tensor, log_space: bool) -> torch.Tensor:
        """
        Forward pass of the MiBiLSTM model.

        Processes input through num_layers of MiBiLSTM layers with dropout between layers.
        Each layer applies forget gate (f), input gate (i), and candidate state (h_tilde),
        then computes: h[t] = f'[t] * h[t-1] + i'[t] * h_tilde[t]
        where f' and i' are length-independent normalized gates.

        Mode Behavior:
            - "Sequential": Processes input sequentially (one timestep at a time).
                           Uses recurrent computation: h[t] depends on h[t-1].
                           Typically used for seq_length=1 or when parallelism not needed.

            - "Parallel": Processes entire sequence using CUDA parallel scan (O(log n)).
                         Computes all timesteps simultaneously via parallel prefix scan.
                         Much faster than sequential for long sequences (seq_length > 16).

            - "Bi-Directional": Runs parallel scan in both forward and backward directions,
                               then concatenates results along hidden dimension.
                               Uses separate linear layers for backward pass.
                               Output size doubles: (batch, seq, 2*hidden_size).

        Input/Output Shapes:
            batch_data: (batch_size, seq_length, input_size)
                       - For first layer: input_size from __init__
                       - For layer > 0: input_size = hidden_size (or 2*hidden_size if bi-directional)

            h_0: (batch_size, 1, hidden_size)
                - Initial hidden state for first timestep
                - Must be positive values if log_space=True (will be converted via log_g)
                - Typically initialized to zeros: torch.zeros(batch, 1, hidden_size)

            output: (batch_size, seq_length, output_size)
                   - output_size = hidden_size for Sequential/Parallel modes
                   - output_size = 2*hidden_size for Bi-Directional mode
                   - Contains processed hidden states for all timesteps

        Log-Space Computation:
            When log_space=True:
                - Gates computed as: log_f' = -softplus(diff), log_i' = -softplus(-diff)
                - Candidate state: log_g(linear_h(x)) ensures positivity
                - CUDA kernel operates in log-space for numerical stability
                - Output automatically converted back via exp() before returning

            When log_space=False:
                - Standard sigmoid gates with normalization: f' = f/(f+i), i' = i/(f+i)
                - Direct computation of h[t] = f'*h[t-1] + i'*h_tilde

        Parameters:
            batch_data (torch.Tensor): Input sequences, shape (batch_size, seq_length, input_size)
            log_space (bool): If True, use log-space computation for numerical stability.

        Returns:
            torch.Tensor: Processed hidden states, shape (batch_size, seq_length, output_size)
                         where output_size = hidden_size (or 2*hidden_size if bi-directional)
        """

        # Keep internal flag consistent with the argument for this pass
        self.set_log_space(log_space)

        # The batch data will serve as our input to the model
        mibi_rnn_input: torch.Tensor = batch_data


        # For each layer
        for layer in range(self.num_layers):


            # Get batch size and hidden size
            batch_size = mibi_rnn_input.size(0)

            # Start by setting up h_0
            # For log_space mode, h0 must be positive before log() is applied in the forward pass
            if self.log_space == True and self.mode.lower().strip() in ['sequential', 'parallel']:
                h_0: torch.Tensor = torch.full((batch_size, 1, self.hidden_size), 0.01, device=mibi_rnn_input.device, dtype=mibi_rnn_input.dtype)
            elif self.log_space != True and self.mode.lower().strip() in ['sequential', 'parallel']:
                h_0: torch.Tensor = torch.zeros(batch_size, 1, self.hidden_size, device=mibi_rnn_input.device, dtype=mibi_rnn_input.dtype)
            elif self.log_space == True and self.mode.lower().strip() == "bi-directional" :
                h_0: torch.Tensor = torch.full((batch_size, 1, self.hidden_size), 0.01, device=mibi_rnn_input.device, dtype=mibi_rnn_input.dtype)
            else:
                h_0: torch.Tensor = torch.zeros(batch_size, 1, self.hidden_size, device=mibi_rnn_input.device, dtype=mibi_rnn_input.dtype)





            # Get the layer dictionary for the layer we're currently at
            layer_dict: nn.Module = self.layers[layer]

            # Extract layer components
            linear_fih: nn.Module = cast(nn.Module, layer_dict._modules['linear_fih'])

            # Set up linear layer
            fih = linear_fih(mibi_rnn_input)

            # Set up the three gates by using torch.chunk alonge the sequence dimension
            f, i, h = torch.chunk(fih, 3, dim=2)

            # Set up the dropout layer
            dropout_layer: nn.Module = cast(nn.Module, layer_dict._modules['dropout'])


            # If not in log space
            if log_space is not True:

                if self.mode.lower().strip() == "sequential":


                    # Compute and normalize gates using fused operation
                    f_prime_t, i_prime_t = compute_gates_and_normalize(f, i)

                    # Compute candidate hidden state
                    tilde_hidden_next: torch.Tensor = h

                    # Compute next hidden state
                    # Output shape: [batch, seq, hidden]
                    mibi_hidden_state_next: torch.Tensor = f_prime_t * h_0 + i_prime_t * tilde_hidden_next

                elif self.mode.lower().strip() == "parallel":


                    # Compute and normalize gates using fused operation
                    f_prime, i_prime = compute_gates_and_normalize(f, i)

                    # Compute candidate hidden state
                    tilde_hidden_next: torch.Tensor = h

                    # Compute i' * h_tilde (element-wise)
                    b_values: torch.Tensor = i_prime * tilde_hidden_next  # (batch, seq, hidden)
                    a_values: torch.Tensor = f_prime  # (batch, seq, hidden)

                    # Run parallel scan
                    mibi_hidden_state_next: torch.Tensor = self.parallel_scan_lstm(b_values, a_values, h_0)

                elif self.mode.lower().strip() == "bi-directional":

                    # Get backward linear layers (separate weights for backward direction)
                    linear_fih_backward: nn.Module = cast(nn.Module, layer_dict._modules['linear_fih_backward'])

                    # Process forward and backward together as doubled batch
                    batch_both = torch.cat([mibi_rnn_input, torch.flip(mibi_rnn_input, dims=[1])], dim=0)

                    # Apply forward linear layer to first half, backward linear layer to second half
                    batch_size_orig = mibi_rnn_input.size(0)

                    # Forward direction
                    fih_fwd = linear_fih(batch_both[:batch_size_orig])
                    f_fwd, i_fwd, h_fwd = torch.chunk(fih_fwd, 3, dim=2)

                    # Backward direction (on flipped input)
                    fih_bwd = linear_fih_backward(batch_both[batch_size_orig:])
                    f_bwd, i_bwd, h_bwd = torch.chunk(fih_bwd, 3, dim=2)

                    # Stack gates for batched processing
                    f_both = torch.cat([f_fwd, f_bwd], dim=0)
                    i_both = torch.cat([i_fwd, i_bwd], dim=0)
                    h_both = torch.cat([h_fwd, h_bwd], dim=0)

                    # Apply activations and normalize for both directions simultaneously
                    f_prime_both, i_prime_both = compute_gates_and_normalize(f_both, i_both)

                    # Compute b and a values for both directions
                    b_both = i_prime_both * h_both
                    a_both = f_prime_both

                    # Create doubled h0: [h0_forward, h0_backward]
                    h0_forward = h_0
                    h0_backward = torch.zeros_like(h_0)
                    h0_both = torch.cat([h0_forward, h0_backward], dim=0)

                    # Single parallel scan call for both directions!
                    h_both_out = self.parallel_scan_lstm(b_both, a_both, h0_both)

                    # Split results
                    h_forward = h_both_out[:batch_size_orig]
                    h_backward = h_both_out[batch_size_orig:]

                    # Flip backward result back to original order
                    h_backward = torch.flip(h_backward, dims=[1])

                    # Concatenate forward and backward hidden states along hidden dimension
                    # Output shape: [batch, seq, 2*hidden]
                    mibi_hidden_state_next: torch.Tensor = torch.cat([h_forward, h_backward], dim=2)

                else:
                    raise ValueError("Invalid mode. Choose from 'Sequential', 'Parallel', or 'Bi-Directional'.")


                if layer != self.num_layers - 1:
                    # Apply dropout except for the last layer
                    mibi_hidden_state_next: torch.Tensor = dropout_layer(mibi_hidden_state_next)
                else:
                    mibi_hidden_state_next: torch.Tensor = mibi_hidden_state_next

                mibi_rnn_input: torch.Tensor = mibi_hidden_state_next

            # If in log space
            else:

                if self.mode.lower().strip() == "sequential":

                    # Compute the forget and input gates
                    f_t: torch.Tensor = torch.sigmoid(f)
                    i_t: torch.Tensor = torch.sigmoid(i)

                    # Normalize gates
                    f_prime_t: torch.Tensor = f_t / (f_t + i_t)
                    i_prime_t: torch.Tensor = i_t / (f_t + i_t)


                    # Compute candidate hidden state in log space
                    tilde_hidden_next: torch.Tensor = g(h)


                    # Compute next hidden state in log space
                    mibi_hidden_state_next: torch.Tensor = f_prime_t * h_0 + i_prime_t * tilde_hidden_next


                elif self.mode.lower().strip() == "parallel":

                    # Compute log-space gates using fused operation
                    log_f_prime, log_i_prime = compute_log_gates(f, i)

                    # Setup initial hidden state in log space using log_g
                    # log_h0 = log_g(h0) as per paper
                    log_h_0: torch.Tensor = log_g(h_0 + 1e-6)

                    # Compute candidate hidden state in log space using log_g
                    # log_h_tilde = log_g(linear_h(x)) as per paper
                    linear_h_output: torch.Tensor = h

                    log_tilde_hidden_next: torch.Tensor = log_g(linear_h_output)

                    # Compute b = log(i' * h_tilde) = log_i' + log_h_tilde, a = log_f'
                    b_values: torch.Tensor = log_i_prime + log_tilde_hidden_next  # (batch, seq, hidden)
                    a_values: torch.Tensor = log_f_prime  # (batch, seq, hidden)

                    # Forward scan
                    mibi_hidden_state_next: torch.Tensor = self.parallel_scan_lstm(b_values, a_values, log_h_0)


                elif self.mode.lower().strip() == "bi-directional":

                    # Get backward linear layers (separate weights for backward direction)
                    linear_fih_backward: nn.Module = cast(nn.Module, layer_dict._modules['linear_fih_backward'])

                    # Concatenate both forward and reverse input - shape: (2*batch, seq, features)
                    batch_both = torch.cat([mibi_rnn_input, torch.flip(mibi_rnn_input, dims=[1])], dim=0)

                    # Apply forward linear layer to first half, backward linear layer to second half
                    batch_size_orig = mibi_rnn_input.size(0)

                    # Forward direction
                    fih_fwd = linear_fih(batch_both[:batch_size_orig])
                    f_fwd, i_fwd, h_fwd = torch.chunk(fih_fwd, 3, dim=2)

                    # Backward direction (on flipped input)
                    fih_bwd = linear_fih_backward(batch_both[batch_size_orig:])
                    f_bwd, i_bwd, h_bwd = torch.chunk(fih_bwd, 3, dim=2)

                    # Stack outputs for batched processing
                    f_both = torch.cat([f_fwd, f_bwd], dim=0)
                    i_both = torch.cat([i_fwd, i_bwd], dim=0)
                    h_both = torch.cat([h_fwd, h_bwd], dim=0)

                    # Compute log-space gates for both directions simultaneously
                    log_f_prime_both, log_i_prime_both = compute_log_gates(f_both, i_both)

                    # Compute candidate hidden state in log space for both
                    log_tilde_hidden_both = log_g(h_both)

                    # Compute b and a for both directions
                    b_both = log_i_prime_both + log_tilde_hidden_both
                    a_both = log_f_prime_both

                    # Create doubled h0: [log_h0_forward, log_h0_backward]
                    log_h_0_fwd = log_g(h_0 + 1e-6)
                    h0_backward_raw = torch.full_like(h_0, 0.01)
                    log_h_0_bwd = log_g(h0_backward_raw + 1e-6)
                    log_h0_both = torch.cat([log_h_0_fwd, log_h_0_bwd], dim=0)

                    # Single parallel scan call for both directions!
                    h_both_out = self.parallel_scan_lstm(b_both, a_both, log_h0_both)

                    # Split results
                    h_forward = h_both_out[:batch_size_orig]
                    h_backward = h_both_out[batch_size_orig:]

                    # Flip backward result back to original order
                    h_backward = torch.flip(h_backward, dims=[1])

                    # Concatenate forward and backward hidden states along hidden dimension
                    # Output shape: [batch, seq, 2*hidden]

                    mibi_hidden_state_next: torch.Tensor = torch.cat([h_forward, h_backward], dim=2)


                else:
                    raise ValueError("Invalid mode. Choose from 'Sequential', 'Parallel', or 'Bi-Directional'.")

                if layer != self.num_layers - 1:
                    # Apply dropout except for the last layer
                    # For log-space, we need to exp() to get real values for next layer
                    mibi_hidden_state_next = torch.exp(torch.clamp(mibi_hidden_state_next, min=-10.0, max=10.0))
                    mibi_hidden_state_next: torch.Tensor = dropout_layer(mibi_hidden_state_next)
                else:
                    # Final layer: exp() to convert from log-space to real values
                    mibi_hidden_state_next: torch.Tensor = torch.exp(torch.clamp(mibi_hidden_state_next, min=-10.0, max=10.0))

            # Make sure next layer receives the previous layers output as input
            mibi_rnn_input: torch.Tensor = mibi_hidden_state_next

        # Get final output
        output: torch.Tensor = mibi_rnn_input

        # return result
        return output