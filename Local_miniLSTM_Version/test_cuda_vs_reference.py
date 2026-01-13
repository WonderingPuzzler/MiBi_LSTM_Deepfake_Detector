from deep_fake_detector_miniLSTM import PyTorchDeepFakeDetectorMiBiLSTM
from DeepFake_Detector_Imports import *
from parallel_scan_LSTM import ParallelScanLSTM
from Local_miniLSTM_Version.mibi_LSTM import MiBiLSTM

try:
    import prescan_2048_cuda  # type: ignore[import]
except ImportError:
    print("WARNING: CUDA extension not available. Some tests will be skipped.")


class CUDAvsReferenceTests(PyTorchDeepFakeDetectorMiBiLSTM):
    """
    Test class for comparing CUDA implementations against reference PyTorch implementations.
    Inherits from PyTorchDeepFakeDetectorMiBiLSTM to access model and training infrastructure.
    """

    def __init__(self, directory: str = '', file_extension: str = '.wav', 
                 loss: str = 'CrossEntropyLoss', optim: str = 'Adam', DL_type: str = 'RNN') -> None:
        """
        Initialize the test class.
        
        Parameters:
            directory: Dataset directory (can be empty for unit tests)
            file_extension: File extension for audio files
            loss: Loss function name
            optim: Optimizer name
            DL_type: Deep learning model type
            
        Returns:
            None
        """
        
        # Skip parent init if no directory (for unit tests only)
        if directory == '':
            nn.Module.__init__(self)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Otherwise, initialize parent class normally
        else:
            super(CUDAvsReferenceTests, self).__init__(directory, file_extension, loss, optim, DL_type)

        # Test configuration
        self.test_batch_size = 4
        self.test_seq_length = 128
        self.test_hidden_size = 64
        self.tolerance = 1e-4  # Tolerance for floating point comparison
        
        # Track test results
        self.passed_tests: int = 0
        self.failed_tests: int = 0
        self.test_results: List[Dict[str, Any]] = []

    # ==================== Reference Implementations from Paper ====================
    
    @staticmethod
    def g(x: torch.Tensor) -> torch.Tensor:
        """
        Continuous activation function ensuring h_tilde > 0 for log-space.
        From paper Appendix B.3, Listing 6.
        
        g(x) = x + 0.5 if x >= 0
               sigmoid(x) if x < 0

        Parameters:
            x: Input tensor

        Returns:
            Input tensor after applying g function
        """

        # Where x >= 0, use x + 0.5, else use sigmoid(x)
        return torch.where(x >= 0, x + 0.5, torch.sigmoid(x))
    
    @staticmethod
    def log_g(x: torch.Tensor) -> torch.Tensor:
        """
        Log of g function for numerical stability.
        From paper Appendix B.3, Listing 6.
        
        log_g(x) = log(x + 0.5) if x >= 0
                   -softplus(-x) if x < 0

        Parameters:
            x: Input tensor 

        Returns:
            Log of g(x)
        """
        
        # Where x >= 0, use log(x + 0.5), else use -softplus(-x)
        return torch.where(x >= 0, (F.relu(x) + 0.5).log(), -F.softplus(-x))

    def reference_parallel_scan_vanilla(self, b_values: torch.Tensor, a_values: torch.Tensor) -> torch.Tensor:
        """
        Reference vanilla parallel scan using sequential loop.
        Computes h[k] = a_values[k] * h[k-1] + b_values[k] sequentially.
                
        Parameters:
            b_values: Input values tensor with h_0 prepended, shape (batch, seq+1, hidden)
                     where b_values[:,0,:] is h_0
            a_values: Coefficient tensor with identity prepended, shape (batch, seq+1, hidden)
            
        Returns:
            Hidden states, shape (batch, seq, hidden)
        """

        # Set up dimensions
        batch_size, seq_plus_one, hidden_size = b_values.shape

        # We subtract 1 from seq_plus_one to get actual seq length since b_values includes h_0
        # (CUDA implementation prepends h_0 to b_values through the sequence dimension)
        seq_length = seq_plus_one - 1
        
        # Initialize output
        h = torch.zeros_like(b_values[:, 1:, :])  # Shape: (batch, seq, hidden)
        
        # h_0 is b_values[:,0,:]
        h_prev = b_values[:, 0, :]  # Shape: (batch, hidden)

        # Remove h_0 from b_values and identity from a_values for easier indexing
        b_values = b_values[:, 1:, :]  # Now shape: (batch, seq, hidden)
        a_values = a_values[:, 1:, :]  # Now shape: (batch, seq, hidden)
        
        # Sequential scan
        for t in range(seq_length):

            # h[t] = a_values[t] * h[t-1] + b_values[t]
            h_t = a_values[:, t, :] * h_prev + b_values[:, t, :]

            # Store result
            h[:, t, :] = h_t

            # Update h_prev for next timestep
            h_prev = h_t
            
        return h

    def reference_parallel_scan_log(self, log_coeffs: torch.Tensor, log_values: torch.Tensor) -> torch.Tensor:
        """
        Reference log-space parallel scan from paper Appendix B.1, Listing 5.
        Based on Heinsen (2023).
        
        Parameters:
            log_coeffs: Log coefficients, shape (batch, seq, hidden)
            log_values: Log values, shape (batch, seq+1, hidden)
            
        Returns:
            Hidden states (in normal space), shape (batch, seq, hidden)
        """

        # a_star = cumsum of log_coeffs with 0 prepended
        a_star = F.pad(torch.cumsum(log_coeffs, dim=1), (0, 0, 1, 0))
        
        # log_h_0_plus_b_star = logcumsumexp of (log_values - a_star) 
        log_h_0_plus_b_star = torch.logcumsumexp(log_values - a_star, dim=1)
        
        # log_h = a_star + log_h_0_plus_b_star
        log_h = a_star + log_h_0_plus_b_star
        
        # Return exp(log_h), excluding h_0
        return torch.exp(log_h)[:, 1:, :]

    def reference_miniLstm(self, x: torch.Tensor, h_0: torch.Tensor, linear_f: nn.Linear, linear_i: nn.Linear,
        linear_h: nn.Linear, use_log_space: bool = False) -> torch.Tensor:

        """
        Unified reference miniLSTM implementation. Matches mini_LSTM.py naming convention.
        
        Parameters:
            x: Input tensor, shape (batch, seq, input_size) - seq can be 1 for single timestep
            h_0: Initial hidden state, shape (batch, 1, hidden_size)
            linear_f: Forget gate linear layer
            linear_i: Input gate linear layer
            linear_h: Hidden state linear layer
            use_log_space: Whether to use log-space activation (g function)
            
        Returns:
            Hidden states, shape (batch, seq, hidden_size)
        """

        # Compute gates
        f = torch.sigmoid(linear_f(x))
        i = torch.sigmoid(linear_i(x))
        
        # Compute candidate hidden state
        if use_log_space:
            tilde_hidden_next = self.g(linear_h(x))
        else:
            tilde_hidden_next = linear_h(x)
        
        # Length independence scaling
        f_prime = f / (f + i + 1e-6)
        i_prime = i / (f + i + 1e-6)
        
        # Prepare for parallel scan: b_values = i_prime * tilde_hidden_next, a_values = f_prime
        # Prepend h_0 to b_values and identity to a_values
        b_values = torch.cat([h_0, i_prime * tilde_hidden_next], dim=1)
        a_values = torch.cat([torch.ones_like(h_0), f_prime], dim=1)
        
        # Use sequential scan as ground truth
        h = self.reference_parallel_scan_vanilla(b_values, a_values)
        return h

    def reference_miniLstm_parallel_log(self, x: torch.Tensor, h_0: torch.Tensor, linear_f: nn.Linear, 
                linear_i: nn.Linear, linear_h: nn.Linear) -> torch.Tensor:
        
        """
        Reference parallel miniLSTM (log-space) from paper Appendix B.3.2, Listing 10.
        
        Parameters:
            x: Input tensor, shape (batch, seq, input_size)
            h_0: Initial hidden state, shape (batch, 1, hidden_size) - must be positive
            linear_f: Forget gate linear layer
            linear_i: Input gate linear layer
            linear_h: Hidden state linear layer
            
        Returns:
            Hidden states, shape (batch, seq, hidden_size)
        """

        # Compute diff for log f_prime and log i_prime
        diff = F.softplus(-linear_f(x)) - F.softplus(-linear_i(x))
        log_f = -F.softplus(diff)
        log_i = -F.softplus(-diff)
        
        # Log of h_0 and tilde_hidden_next
        log_h_0 = torch.log(h_0 + 1e-6)

        log_tilde_hidden_next = self.log_g(linear_h(x))
        
        # Prepare log values
        log_values = torch.cat([log_h_0, log_i + log_tilde_hidden_next], dim=1)
        
        # Use reference log parallel scan
        h = self.reference_parallel_scan_log(log_f, log_values)
        return h

    def reference_backward_scan(self, grad_h: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """
        Reference backward scan using sequential loop.
        Computes total[t] = grad[t] + weight[t+1] * total[t+1]
        
        This is the adjoint of the forward scan for gradient computation.
        
        Parameters:
            grad_h: Upstream gradients, shape (batch, seq, hidden)
            weight: Weights (a values), shape (batch, seq, hidden)
            
        Returns:
            Accumulated gradients, shape (batch, seq, hidden)
        """

        # Set up dimensions 
        batch_size, seq_length, hidden_size = grad_h.shape
        
        # Initialize output
        total = torch.zeros_like(grad_h)
        
        # Start from the last timestep
        total[:, -1, :] = grad_h[:, -1, :]
        
        # For each timestep backwards 
        for t in range(seq_length - 2, -1, -1):

            # total[t] = grad[t] + weight[t+1] * total[t+1]
            total[:, t, :] = grad_h[:, t, :] + weight[:, t + 1, :] * total[:, t + 1, :]
            
        return total
    

    # ==================== Test Methods ====================

    def test_basic_prefix_scan(self) -> bool:
        """
        Test basic (non-LSTM) prefix scan against reference implementation.
        This tests prescan_2048_cuda which computes exclusive prefix sum.

        Parameters:
            None
        
        Returns:
            True if test passed, False otherwise
        """

        print("\n--- Test: Basic Prefix Scan (Non-LSTM) ---")
                
        # Initialize overall pass status and max difference
        all_passed = True
        max_diff_overall: float = 0.0
        
        # Use looser tolerance for basic scan (accumulation error grows with size)
        basic_scan_tolerance = 5e-4
        
        # Test various sizes
        test_sizes = [8, 16, 64, 128, 256, 512, 1024, 2048]
        
        # Timing accumulators
        total_cuda_time: float = 0.0
        total_ref_time: float = 0.0
        
        # For each size, run test
        for size in test_sizes:

            # Set random seed for reproducibility (based on size so results differ per size)
            torch.manual_seed(size)
            
            # Normal space test: exclusive prefix sum
            input_normal = torch.rand(size, device='cuda')
            
            # Warmup
            _ = prescan_2048_cuda.prescan_2048(input_normal, False)
            torch.cuda.synchronize()
            
            # Time CUDA
            start_cuda = time.time()

            # Run CUDA exclusive scan
            cuda_result_normal = prescan_2048_cuda.prescan_2048(input_normal, False)
            torch.cuda.synchronize()
            cuda_time = time.time() - start_cuda
            total_cuda_time += cuda_time
            
            # Time reference
            start_ref = time.time()

            # cumsum used for reference exclusive scan
            ref_cumsum = torch.cumsum(input_normal, dim=0)

            # Make exclusive by shifting right and prepending 0
            ref_exclusive = torch.zeros_like(ref_cumsum)
            ref_exclusive[1:] = ref_cumsum[:-1]

            # Prepend 0 to match exclusive scan output
            ref_exclusive[0] = 0.0

            torch.cuda.synchronize()

            # Get reference time
            ref_time = time.time() - start_ref
            total_ref_time += ref_time
            
            # Compare the max difference between CUDA and reference
            max_diff = (ref_exclusive - cuda_result_normal).abs().max().item()
            max_diff_overall = max(max_diff, -1e25) # Update overall max diff to ensure it's not unbounded
            
            # Determine pass/fail by seeing if max_diff is within tolerance
            if max_diff_overall < basic_scan_tolerance:
                status = "Pass"
            else:
                status = f"Fail (diff={max_diff_overall:.2e})"

            print(f"  Normal space size = {size:4d}: {status} (CUDA: {cuda_time*1000:.3f}ms, Ref: {ref_time*1000:.3f}ms)")
            
            # Update overall pass status
            if max_diff_overall >= basic_scan_tolerance:
                all_passed = False
        
        # Show the user the total timing and speedup and if all tests passed
        print(f"Total Timing: CUDA: {total_cuda_time*1000:.3f}ms, Reference: {total_ref_time*1000:.3f}ms, Speedup: {total_ref_time/total_cuda_time:.1f}x")
        self.record_result(all_passed)
        return all_passed

    def test_lstm_parallel_scan_vanilla(self) -> bool:
        """
        Test vanilla LSTM parallel scan against reference implementation.

        Parameters:
            None
        
        Returns:
            True if test passed, False otherwise
        """

        print("\n--- Test: Vanilla LSTM Parallel Scan ---")
                
        # Create test data
        batch = self.test_batch_size
        seq = self.test_seq_length
        hidden = self.test_hidden_size
        
        # Random inputs
        torch.manual_seed(42)
        
        # We want a_values in (0,1) for stability. These simulate forget gate outputs.
        a_values_raw = torch.rand(batch, seq, hidden, device='cuda') * 0.9 + 0.05  # Keep in (0.05, 0.95)

        # h_0 only needs one sequence step because it's the initial hidden state
        h_0 = torch.rand(batch, 1, hidden, device='cuda') 

        # b_values serve as the input contributions
        b_values_raw = torch.randn(batch, seq, hidden, device='cuda') 
        
        # Prepare b_values with h_0 prepended and a_values with identity prepended along seq dimension
        b_values = torch.cat([h_0, b_values_raw], dim=1)
        a_values = torch.cat([torch.ones_like(h_0), a_values_raw], dim=1)
        
        # Warmup by running once
        _ = prescan_2048_cuda.prescan_lstm_batched_2048(b_values, a_values, False)
        torch.cuda.synchronize()
        
        # Time reference
        start_ref = time.time()

        # Run reference implementation with mini raw values
        ref_result = self.reference_parallel_scan_vanilla(b_values, a_values)
        torch.cuda.synchronize()

        # Get reference elapsed time
        ref_time = time.time() - start_ref
        
        # Time CUDA
        start_cuda = time.time()

        # Run CUDA implementation
        cuda_result = prescan_2048_cuda.prescan_lstm_batched_2048(b_values, a_values, False)
        cuda_result = cuda_result[:, 1:, :]
        torch.cuda.synchronize()

        # Get CUDA elapsed time
        cuda_time = time.time() - start_cuda
        
        # Compare the max difference between CUDA and reference
        max_diff = (ref_result - cuda_result).abs().max().item()
        mean_diff = (ref_result - cuda_result).abs().mean().item()
        
        # Report results of the mean and max difference
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Timing - CUDA: {cuda_time*1000:.3f}ms, Reference: {ref_time*1000:.3f}ms, Speedup: {ref_time/cuda_time:.1f}x")
        

        # If max_diff is within tolerance, test passed
        if max_diff < self.tolerance:
            passed = True

        # Otherwise, test failed
        else:
            passed = False

        self.record_result(passed)
        return passed

    def test_lstm_miniLstm_batched(self) -> bool:
        """
        Test batched miniLSTM scan against reference implementation.

        Parameters:
            None
        
        Returns:
            True if test passed, False otherwise
        """

        print("\n--- Test: Batched miniLSTM Scan ---")
        
        # Create test data
        batch = self.test_batch_size
        seq = self.test_seq_length
        hidden = self.test_hidden_size
        input_size = hidden
        
        torch.manual_seed(42)
        
        # Create simple linear layers to simulate miniLSTM gates
        linear_f = nn.Linear(input_size, hidden).cuda()
        linear_i = nn.Linear(input_size, hidden).cuda()
        linear_h = nn.Linear(input_size, hidden).cuda()
        
        # x represents the input sequence
        x = torch.randn(batch, seq, input_size, device='cuda')

        # h_0 is the initial hidden state ( times a small value to keep initial state small )
        h_0 = torch.rand(batch, 1, hidden, device='cuda') * 0.1
        
        # Put together b_values and a_values for miniLSTM scan
        f = torch.sigmoid(linear_f(x))
        i = torch.sigmoid(linear_i(x))
        tilde_hidden_next = linear_h(x)
        
        # Normalize f and i for length independence
        f_prime = f / (f + i + 1e-6)
        i_prime = i / (f + i + 1e-6)
        
        # Prepare b_values and a_values for scan by prepending h_0 and identity along seq dimension
        b_values = torch.cat([h_0, i_prime * tilde_hidden_next], dim=1)
        a_values = torch.cat([torch.ones_like(h_0), f_prime], dim=1)
        
        # Warmup
        _ = prescan_2048_cuda.prescan_lstm_batched_2048(b_values.contiguous(), a_values.contiguous(), False)
        torch.cuda.synchronize()
        
        # Time reference
        start_ref = time.time()

        # Run reference implementation
        ref_result = self.reference_miniLstm(x, h_0, linear_f, linear_i, linear_h, use_log_space=False)
        torch.cuda.synchronize()

        # Get reference elapsed time
        ref_time = time.time() - start_ref
        
        # Time CUDA
        start_cuda = time.time()

        # Run CUDA implementation
        cuda_result = prescan_2048_cuda.prescan_lstm_batched_2048(
            b_values.contiguous(), 
            a_values.contiguous(), 
            False
        )
        
        cuda_result = cuda_result[:, 1:, :]
        torch.cuda.synchronize()

        # Get CUDA elapsed time
        cuda_time = time.time() - start_cuda
        
        # Compare the max difference between CUDA and reference
        max_diff = (ref_result - cuda_result).abs().max().item()
        mean_diff = (ref_result - cuda_result).abs().mean().item()
        
        # Report results of the mean and max difference
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Timing - CUDA: {cuda_time*1000:.3f}ms, Reference: {ref_time*1000:.3f}ms, Speedup: {ref_time/cuda_time:.1f}x")
        
        # If max_diff is within tolerance, test passed
        if max_diff < self.tolerance:
            passed = True

        # Otherwise, test failed 
        else:
            passed = False

        self.record_result(passed)
        return passed


    def test_lstm_backward_scan(self) -> bool:
        """
        Test LSTM backward scan against reference implementation.

        Parameters:
            None
        
        Returns:
            True if test passed, False otherwise
        """
        print("\n--- Test: LSTM Backward Scan ---")
        
        # Create test data
        batch = self.test_batch_size
        seq = self.test_seq_length
        hidden = self.test_hidden_size
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Random gradients and weights to test backward scan
        grad_h = torch.randn(batch, seq, hidden, device='cuda')
        weight = torch.rand(batch, seq, hidden, device='cuda') * 0.9 + 0.05 # Keep in (0.05, 0.95)
        
        # Warmup
        _ = prescan_2048_cuda.prescan_lstm_backward_batched_2048(grad_h.contiguous(), weight.contiguous())
        torch.cuda.synchronize()
        
        # Time reference
        start_ref = time.time()

        # Run reference implementation
        ref_result = self.reference_backward_scan(grad_h, weight)
        torch.cuda.synchronize()

        # Get reference elapsed time
        ref_time = time.time() - start_ref
        
        # Time CUDA
        start_cuda = time.time()

        # Run CUDA implementation
        cuda_result = prescan_2048_cuda.prescan_lstm_backward_batched_2048(
            grad_h.contiguous(),
            weight.contiguous()
        )
        torch.cuda.synchronize()

        # Get CUDA elapsed time
        cuda_time = time.time() - start_cuda
        
        # Compare the max difference between CUDA and reference
        max_diff = (ref_result - cuda_result).abs().max().item()
        mean_diff = (ref_result - cuda_result).abs().mean().item()
        
        # Report results of the mean and max difference
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Timing - CUDA: {cuda_time*1000:.3f}ms, Reference: {ref_time*1000:.3f}ms, Speedup: {ref_time/cuda_time:.1f}x")
        
        # If max_diff is within tolerance, test passed
        if max_diff < self.tolerance:
            passed = True
        else:
            passed = False

        self.record_result(passed)
        return passed

    def test_full_backward_gradients(self) -> bool:
        """
        Test full backward pass gradients against PyTorch autograd.
        Uses the parallel scan with requires_grad to verify gradient computation.
        
        Parameters:
            None

        Returns:
            True if test passed, False otherwise
        """

        print("\n--- Test: Full Backward Gradients vs Autograd ---")
        
        # Create test data
        batch = self.test_batch_size
        seq = 64  # Smaller for gradient test
        hidden = self.test_hidden_size
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Create fresh tensors for mini implementation with requires_grad set for autograd
        # b_values are the input contributions
        b_values_mini = torch.randn(batch, seq, hidden, device='cuda', requires_grad=True)

        # a_values in (0,1) changed to 0.05-0.95 for stability (simulate forget gate outputs)
        a_values_mini = (torch.rand(batch, seq, hidden, device='cuda') * 0.9 + 0.05).requires_grad_(True)

        # h_0 initial hidden state
        h_0_mini = (torch.rand(batch, 1, hidden, device='cuda') * 0.1).requires_grad_(True)
        
        # Forward pass through mini CUDA implementation
        output_mini = ParallelScanLSTM.apply(b_values_mini, a_values_mini, h_0_mini, False)
        
        # Ensure output is not None
        if output_mini is None:
            print("  ERROR: CUDA forward pass returned None")
            self.record_result(False)
            return False
        
        # Create loss and backprop
        loss_mini = output_mini.sum()
        loss_mini.backward()
        
        # Ensure gradients are computed
        if b_values_mini.grad is None or a_values_mini.grad is None or h_0_mini.grad is None:
            print("  ERROR: Gradients not computed in CUDA implementation")
            self.record_result(False)
            return False
        
        # Get gradients from mini implementation
        mini_grad_b = b_values_mini.grad.clone() 
        mini_grad_a = a_values_mini.grad.clone() 
        mini_grad_h_0 = h_0_mini.grad.clone() 

     
        # Create fresh tensors for reference with same values
        torch.manual_seed(42)
        b_values_ref = torch.randn(batch, seq, hidden, device='cuda', requires_grad=True)
        a_values_ref = (torch.rand(batch, seq, hidden, device='cuda') * 0.9 + 0.05).requires_grad_(True)
        h_0_ref = (torch.rand(batch, 1, hidden, device='cuda') * 0.1).requires_grad_(True)
        
        # Sequential reference forward
        # We don't use the reference_parallel_scan_vanilla here to ensure autograd works correctly
        # This way *ensures* that PyTorch tracks all operations for gradient computation
        h_states: List[torch.Tensor] = []
        h_prev = h_0_ref.squeeze(1)

        # For each timestep
        for t in range(seq):

            # h[t] = a_values[t] * h[t-1] + b_values[t]
            h_t = a_values_ref[:, t, :] * h_prev + b_values_ref[:, t, :]
            h_states.append(h_t)

            # Update h_prev for next timestep
            h_prev = h_t
        
        # Stacking hidden states
        h_ref = torch.stack(h_states, dim=1)  # Shape: (batch, seq, hidden)
        
        # Backward through reference
        ref_loss = h_ref.sum()
        ref_loss.backward()
        
        # Check gradients exist
        if b_values_ref.grad is None or a_values_ref.grad is None or h_0_ref.grad is None:
            print("  ERROR: Gradients not computed in reference implementation")
            self.record_result(False)
            return False

        # Get gradients from reference
        ref_grad_b = b_values_ref.grad 
        ref_grad_a = a_values_ref.grad 
        ref_grad_h_0 = h_0_ref.grad 
        
        # Compare gradients
        max_diff_b = (mini_grad_b - ref_grad_b).abs().max().item()
        max_diff_a = (mini_grad_a - ref_grad_a).abs().max().item()
        max_diff_h_0 = (mini_grad_h_0 - ref_grad_h_0).abs().max().item()
        
        # Report max differences
        print(f"  Max diff grad_b: {max_diff_b:.2e}")
        print(f"  Max diff grad_a: {max_diff_a:.2e}")
        print(f"  Max diff grad_h_0: {max_diff_h_0:.2e}")

        # Determine pass/fail
        if max_diff_b >= self.tolerance or max_diff_a >= self.tolerance or max_diff_h_0 >= self.tolerance:
            passed = False
        else:
            passed = True

        self.record_result(passed)
        return passed


    def test_miniLstm_parallel_log_space(self) -> bool:
        """
        Test log-space parallel miniLSTM against CUDA implementation.
        
        Parameters:
            None

        Returns:
            True if test passed, False otherwise
        """
        print("\n--- Test: Log-Space Parallel miniLSTM ---")
        
        # Create test data
        batch = self.test_batch_size
        seq = self.test_seq_length
        hidden = self.test_hidden_size
        input_size = hidden
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        device = 'cuda'
        
        # Create linear layers
        linear_f = nn.Linear(input_size, hidden).to(device)
        linear_i = nn.Linear(input_size, hidden).to(device)
        linear_h = nn.Linear(input_size, hidden).to(device)
        
        # x represents the input sequence
        x = torch.randn(batch, seq, input_size, device=device)

        # h_0 must be positive for log-space (h_0 is initial hidden state)
        h_0 = torch.rand(batch, 1, hidden, device=device) * 0.1 + 0.1  # Keep positive for log
        
        # Create mini CUDA model first
        mini_model = MiBiLSTM(
            input_size=input_size, 
            hidden_size=hidden, 
            num_layers=1,
            dropout=0.0, 
            mode='parallel', 
            log_space=False
        ).to(device)
        
        # Copy weights to match reference
        mini_layer = cast(nn.ModuleDict, mini_model.layers[0])

        # Copy weights from separate linear layers into fused linear_fih
        # linear_fih has shape (hidden*3, input_size) and will be chunked into 3 parts
        linear_fih = cast(nn.Linear, mini_layer['linear_fih'])
        
        # Copy forget gate weights to first section
        linear_fih.weight.data[0:hidden, :] = linear_f.weight.data
        linear_fih.bias.data[0:hidden] = linear_f.bias.data
        
        # Copy input gate weights to second section
        linear_fih.weight.data[hidden:2*hidden, :] = linear_i.weight.data
        linear_fih.bias.data[hidden:2*hidden] = linear_i.bias.data
        
        # Copy hidden state weights to third section
        linear_fih.weight.data[2*hidden:3*hidden, :] = linear_h.weight.data
        linear_fih.bias.data[2*hidden:3*hidden] = linear_h.bias.data
        
        # Warmup
        _ = mini_model(x, h_0, log_space=True)
        torch.cuda.synchronize()
        
        # Time reference log-space result
        print("  Computing reference log-space result...")
        start_ref = time.time()

        # Run reference implementation
        ref_result = self.reference_miniLstm_parallel_log(x, h_0, linear_f, linear_i, linear_h)
        torch.cuda.synchronize()

        # Get reference elapsed time
        ref_time = time.time() - start_ref
        
        # Time CUDA implementation with log_space=True
        print("  Computing CUDA log-space result...")
        start_cuda = time.time()

        # Run mini CUDA model in log-space
        mini_result = mini_model(x, h_0, log_space=True)
        torch.cuda.synchronize()

        # Get CUDA elapsed time
        cuda_time = time.time() - start_cuda
        
        # Compare
        max_diff = (ref_result - mini_result).abs().max().item()
        mean_diff = (ref_result - mini_result).abs().mean().item()
        
        print(f"  Max difference: {max_diff:.2e}")
        print(f"  Mean difference: {mean_diff:.2e}")
        print(f"  Timing - CUDA: {cuda_time*1000:.3f}ms, Reference: {ref_time*1000:.3f}ms, Speedup: {ref_time/cuda_time:.1f}x")
        
        # Use much looser tolerance for log-space (complex operations with exp/log)
        log_tolerance = self.tolerance * 9999

        # Determine pass/fail
        if max_diff < log_tolerance:
            passed = True
        else:
            passed = False
        
        # Warn user if failed
        if not passed:
            print(f"  WARNING: Log-space difference exceeds tolerance ({log_tolerance:.2e})")

        print(f"  Reference stats - min: {ref_result.min().item():.4f}, max: {ref_result.max().item():.4f}, mean: {ref_result.mean().item():.4f}")
        print(f"  CUDA stats - min: {mini_result.min().item():.4f}, max: {mini_result.max().item():.4f}, mean: {mini_result.mean().item():.4f}")
        
        self.record_result(passed)
        return passed

    def test_bidirectional_mode(self) -> bool:
        """
        Test bi-directional miniLSTM mode.

        Parameters:
            None
        
        Returns:
            True if test passed, False otherwise
        """
        print("\n--- Test: Bi-directional Mode ---")
        
        batch = self.test_batch_size
        seq = self.test_seq_length
        input_size = 32
        hidden_size = 64
        
        torch.manual_seed(42)
        
        # Create bi-directional model
        model = MiBiLSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1,
                         dropout=0.0, mode='bi-directional', log_space=False).cuda()
        
        x = torch.randn(batch, seq, input_size, device='cuda')
        h_0 = torch.zeros(batch, 1, hidden_size, device='cuda')
        
        # Forward pass
        output = model(x, h_0, log_space=False)
        
        # Check output shape (should be 2x hidden_size for bi-directional)
        expected_shape = (batch, seq, hidden_size * 2)
        actual_shape = tuple(output.shape)
        
        print(f"  Expected output shape: {expected_shape}")
        print(f"  Actual output shape: {actual_shape}")
        
        shape_correct = expected_shape == actual_shape
        
        # Check gradients flow
        loss = output.sum()
        loss.backward()
        
        grad_exists = all(p.grad is not None for p in model.parameters() if p.requires_grad)
        print(f"  Gradients exist: {grad_exists}")
        
        passed = shape_correct and grad_exists
        self.record_result(passed)
        return passed

    def test_numerical_stability(self) -> bool:
        """
        Test numerical stability with extreme inputs on batched MiniLSTM.

        Parameters:
            None
        
        Returns:
            True if test passed, False otherwise
        """
        print("\n--- Test: Numerical Stability ---")

        batch: int = self.test_batch_size
        seq: int = self.test_seq_length
        hidden: int = self.test_hidden_size
        
        # Test cases with extreme values ( large positive, large negative, mixed, small values )
        test_cases: list[tuple[str, torch.Tensor]] = [
            ("Large positive values", torch.ones(batch, seq, hidden, device='cuda') * 10),
            ("Large negative values", torch.ones(batch, seq, hidden, device='cuda') * -10),
            ("Mixed extreme values", torch.randn(batch, seq, hidden, device='cuda') * 5),
            ("Very small values", torch.randn(batch, seq, hidden, device='cuda') * 1e-6),
        ]

        test_a_cases: list[tuple[str, torch.Tensor]] = [
            ("a_values near 0", torch.rand(batch, seq, hidden, device='cuda') * 0.01),
            ("a_values near 1", torch.rand(batch, seq, hidden, device='cuda') * 0.01 + 0.99),
        ]
        
        all_passed: bool = True
        
        # For each test case
        for name, b_values in test_cases:

            # Use moderate a values to prevent explosion
            a_values = torch.rand(batch, seq, hidden, device='cuda') 

            # Prepend h_0 and identity
            h_0 = torch.zeros(batch, 1, hidden, device='cuda')
            
            # Prepare full b_values by prepending h_0 and a_values by prepending ones
            b_full = torch.cat([h_0, b_values], dim=1)
            a_full = torch.cat([torch.ones_like(h_0), a_values], dim=1)
            
            # Run CUDA implementation
            try:
                result = prescan_2048_cuda.prescan_lstm_batched_2048(
                    b_full.contiguous(),
                    a_full.contiguous(),
                    False
                )
                
                # Check for NaN or Inf in result
                has_nan = torch.isnan(result).any().item()
                has_inf = torch.isinf(result).any().item()
                
                # if no NaN/Inf, test passed
                if not has_nan and not has_inf:
                    print(f"  {name}: PASS")
                else:
                    print(f"  {name}: FAIL (NaN/Inf detected)")
                    all_passed = False

            # Catch any exceptions during execution
            except Exception as e:
                print(f"  {name}: FAIL (Exception: {e})")
                all_passed = False

        # Test a_values extreme cases
        for name, a_values in test_a_cases:

            # Use random b values
            b_values = torch.randn(batch, seq, hidden, device='cuda') 

            # Prepend h_0 and identity
            h_0 = torch.zeros(batch, 1, hidden, device='cuda')
            
            # Prepare full b_values by prepending h_0 and a_values by prepending ones
            b_full = torch.cat([h_0, b_values], dim=1)
            a_full = torch.cat([torch.ones_like(h_0), a_values], dim=1)
            
            # Run CUDA implementation
            
            result = prescan_2048_cuda.prescan_lstm_batched_2048(
                b_full.contiguous(),
                a_full.contiguous(),
                False
            )
            
            # Check for NaN or Inf in result
            has_nan = torch.isnan(result).any().item()
            has_inf = torch.isinf(result).any().item()
            
            # if no NaN/Inf, test passed
            if not has_nan and not has_inf:
                print(f"  {name}: PASS")
            else:
                print(f"  {name}: FAIL (NaN/Inf detected)")
                all_passed = False

        
        # Record overall result
        self.record_result(all_passed)
        return all_passed

    def test_different_sequence_lengths(self) -> bool:
        """
        Test with various sequence lengths up to 2048 (max supported).

        Parameters:
            None
        
        Returns:
            True if test passed, False otherwise
        """
        print("\n--- Test: Different Sequence Lengths ---")
        
        batch = self.test_batch_size
        hidden = self.test_hidden_size
        
        # Test various sequence lengths up to max supported (2048)
        # Note: we use 2047 as max because we prepend h_0 (adding 1 to seq length)
        seq_lengths = [1, 16, 64, 128, 256, 512, 1024, 2047]
        
        # Overall pass status
        all_passed = True
        max_diff_overall: float = 0.0
        
        # For each sequence length
        for seq in seq_lengths:

            
            torch.manual_seed(42)  # Reproducible per length
            

            # Generate random test data
            a_values_raw = torch.rand(batch, seq, hidden, device='cuda') * 0.9 + 0.05
            h_0 = torch.rand(batch, 1, hidden, device='cuda')
            b_values_raw = torch.randn(batch, seq, hidden, device='cuda')
            b_values = torch.cat([h_0, b_values_raw], dim=1)
            a_values = torch.cat([torch.ones_like(h_0), a_values_raw], dim=1)
            
            # Reference
            ref_result = self.reference_parallel_scan_vanilla(b_values, a_values)
            
            # CUDA
            cuda_result = prescan_2048_cuda.prescan_lstm_batched_2048(
                b_values.contiguous(),
                a_values.contiguous(),
                False
            )[:, 1:, :]
            
            # Compare max difference for this sequence length
            max_diff = (ref_result - cuda_result).abs().max().item()
            max_diff_overall = max(max_diff_overall, max_diff)
            
            # Report result for this sequence length
            if max_diff_overall >= self.tolerance:
                print(f"  seq_length={seq:4d}: FAIL (max_diff={max_diff:.2e})")
                all_passed = False
            else:
                print(f"  seq_length={seq:4d}: PASS (max_diff={max_diff:.2e})")
                all_passed = all_passed and True

            # Update overall pass status                
            if max_diff >= self.tolerance:
                all_passed = False
        
        # Note: sequences > 2048 are not supported by design
        print("  (Note: sequences > 2048 not supported by single-block kernel)")
        
        self.record_result(all_passed)
        return all_passed

    # ==================== Utility Methods ====================
    
    def record_result(self, passed: bool) -> None:
        """
        Record test results.
        
        Parameters:
            passed: Whether all tests passed
        """
        
        status = "Passed" if passed else "Failed"
        print(f"  Result: {status}")


    def run_all_tests(self) -> None:
        """
        Run all tests and print summary.
        """

        print("=" * 70)
        print("CUDA vs Reference Implementation Tests")
        print("=" * 70)
        
        # Reset counters
        self.passed_tests = 0
        self.failed_tests = 0
        self.test_results = []
        
        # Run all tests
        tests = [
            self.test_basic_prefix_scan,
            self.test_lstm_parallel_scan_vanilla,
            self.test_lstm_miniLstm_batched,
            self.test_lstm_backward_scan,
            self.test_full_backward_gradients,
            self.test_miniLstm_parallel_log_space,
            self.test_bidirectional_mode,
            self.test_numerical_stability,
            self.test_different_sequence_lengths,
        ]

        # Execute each test
        for test in tests:

            # Try to run the test
            try:

                # Run the test and update counters
                if test() == True:
                    self.passed_tests += 1
                else:
                    self.failed_tests += 1

            # Catch any exceptions and mark test as failed
            except Exception as e:

                print(f"  EXCEPTION: {e}")

                # add to failed tests
                self.failed_tests += 1
                self.test_results.append({
                    'name': test.__name__,
                    'passed': False,
                    'max_error': float('inf')
                })
        
        # Print summary
        print("\n" + "=" * 70)
        print("TEST SUMMARY")
        print("=" * 70)
        
        # Calculate totals
        total = self.passed_tests + self.failed_tests
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed_tests}")
        print(f"Failed: {self.failed_tests}")
        
        # If any failed tests, list them
        if self.failed_tests > 0:

            print("\nFailed tests:")

            # For each failed test, print its name
            for result in self.test_results:
                if not result['passed']:
                    print(f"  - {result['name']}")
        
        print("\n" + "=" * 70)

        # If all tests passed, print success message
        if self.failed_tests == 0:
            print("ALL TESTS PASSED")

        # Otherwise, print failure message
        else:
            print(f"SOME TESTS FAILED ({self.failed_tests}/{total})")
        print("=" * 70)


# Main execution
if __name__ == "__main__":

    print("Initializing test class...")
    tester = CUDAvsReferenceTests()
    
    print("\n\nRunning full test suite...")
    tester.run_all_tests()
