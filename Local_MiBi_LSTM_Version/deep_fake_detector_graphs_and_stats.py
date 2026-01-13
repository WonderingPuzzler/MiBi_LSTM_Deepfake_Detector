from deep_fake_detector_miniLSTM import PyTorchDeepFakeDetectorMiBiLSTM
from DeepFake_Detector_Imports import *

class DeepFakeDetectorGraphsAndStats(PyTorchDeepFakeDetectorMiBiLSTM):
    """
    A class to generate graphs and statistics for the DeepFake Detector.
    Inherits from PyTorchDeepFakeDetectorMiBiLSTM, BatchLossAndOptimization, nn.Module, Dataset, and ClassesFilesDictionarySetUp.
    """

    def __init__(self, directory: str = '', file_extension: str = '.wav', loss: str = 'CrossEntropyLoss', optim: str = 'Adam', DL_type: str = 'RNN', data_type: str = 'mfcc', two_class: bool = False, random_seed: int = 42) -> None:
        """
        Initialize the DeepFakeDetectorGraphsAndStats class.
        Also send directory, file_extension, loss, optim, DL_type, data_type, two_class, and random_seed to the parent class.

        Parameters:
            directory (str): Directory containing the dataset.
            file_extension (str): File extension of the audio files.
            loss (str): Loss function to use.
            optim (str): Optimizer to use.
            DL_type (str): Type of deep learning model to use.
            data_type (str): Type of data to use.
            two_class (bool): Whether to use two-class classification.
            random_seed (int): Random seed number for reproducibility.

        Returns:
            None
        """

        # Initialize the parent class and send parameters up the chain
        super(DeepFakeDetectorGraphsAndStats, self).__init__(directory, file_extension, loss, optim, DL_type, data_type, two_class, random_seed)

        # Initialize attributes for storing testing data and predictions
        self.all_labels: List[int] = []
        self.all_predictions: List[int] = []
        self.all_probabilities: np.ndarray = np.array([])


    def plot_training_curves(self) -> None:
        """
        Plot training and validation loss and accuracy curves in 4 subplots.

        Parameters:
            None

        Returns:
            None
        """

        print("\n=============================== Plotting Training Curves ===============================\n")

        # Check if training metrics exist
        if not hasattr(self, 'train_loss_list') or not self.train_loss_list:
            print("No training metrics found. Please train the model first.")
            return

        epochs: range = range(1, len(self.train_loss_list) + 1)

        # Create figure with 4 subplots (2x2 grid)
        plt.figure(figsize=(14, 10))

        # Subplot 1: Training Loss
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.train_loss_list, label='Training Loss', color='blue')
        plt.title('Training Loss', fontsize=18, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.grid(True)

        # Subplot 2: Validation Loss
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.val_loss_list, label='Validation Loss', color='orange')
        plt.title('Validation Loss', fontsize=18, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.grid(True)

        # Subplot 3: Training Accuracy
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.train_acc_list, label='Training Accuracy', color='green')
        plt.title('Training Accuracy', fontsize=18, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.legend()
        plt.grid(True)

        # Subplot 4: Validation Accuracy
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.val_acc_list, label='Validation Accuracy', color='red')
        plt.title('Validation Accuracy', fontsize=18, fontweight='bold')
        plt.xlabel('Epoch', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        plt.legend()
        plt.grid(True)

        # Adjust layout and show plot
        plt.tight_layout()
        plt.show()

        print("\n---------------------------- Training Curves Plotted ----------------------------\n")

    def plot_class_counts(self) -> None:
        """
        Plot the distribution of samples per class in the dataset.

        Parameters:
            None

        Returns:
            None
        """

        print(" \n=============================== Sample Count Diagramming =============================== \n")

        # Get class counts
        class_counts: Dict[str, int] = self.get_class_counts()

        # Plotting the class distribution
        plt.figure(figsize=(10, 6))
        # Make sure the classes are the x-axis labels, and the values are the heights of the bars in the y-axis
        plt.bar(list(class_counts.keys()), list(class_counts.values()))
        plt.xlabel("Classes")
        plt.ylabel("Number of Samples")
        plt.title("Class Distribution", fontsize=18, fontweight='bold')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

        print(" \n---------------------------- Sample Count Diagramming Complete ---------------------------- \n")


    def test_batch_equality(self, num_batches_to_test: int) -> None:
        """
        Test function to verify that a subset of batches from the training DataLoader are balanced by showing class distributions.

        Parameters:
            num_batches_to_test (int): The number of batches to test for balance.

        Returns:
            None
        """

        print("\n=============================== Testing Batch Equality ===============================\n")

        # Get the training DataLoader
        training_loader: DataLoader = self.get_training_loader()

        # Ensure the training loader is available
        if training_loader is None:
            raise ValueError("Training loader is not set up.")

        # Iterate through the specified number of batches
        for batch_idx, (_, labels) in enumerate(training_loader):

            # Limit to the specified number of batches
            if batch_idx >= num_batches_to_test:
                break

            # Count class distribution in this batch
            label_counts: Counter = Counter(labels.tolist())

            print(f"\nBatch {batch_idx + 1}:")
            print(f"  Total samples: {len(labels)}")
            print(f"  Class distribution:")

            # Print counts and percentages for each class
            for class_idx in sorted(label_counts.keys()):

                # Get count and percentage for this class
                count: int = label_counts[class_idx]
                percentage: float = (count / len(labels)) * 100
                print(f"    Class {class_idx}: {count:3d} samples ({percentage:5.1f}%)")

            # Create a bar plot for this batch's class distribution
            plt.figure(figsize=(8, 4))

            # Only plot if class indices are integers
            if len(label_counts) > 0 and isinstance(next(iter(label_counts.keys())), int):
                # Make sure the class indices are the x-axis labels, and the values are the heights of the bars in the y-axis
                plt.bar(list(label_counts.keys()), list(label_counts.values()))

            plt.xlabel("Class Index", fontsize=16, fontweight='bold')
            plt.ylabel("Number of Samples", fontsize=16, fontweight='bold')
            plt.title(f"Class Distribution in Batch {batch_idx + 1}")
            plt.show()

        print("\n---------------------------- Batch Equality Test Complete ----------------------------\n")


    def plot_random_mfcc_samples(self, samples: int = 3) -> None:
        """
        Plot random MFCC samples from each class using raw audio files.

        Parameters:
            samples (int): Number of random samples to plot for each class. Defaults to 3.

        Returns:
            None
        """

        print("\n=============================== Plotting Random MFCC Samples ===============================\n")

        # Get the file dictionary
        file_dict: dict[int, list[str]] = self.get_file_dictionary()

        # Iterate through each class
        for class_name in self.get_classes():

            # Get class index and corresponding files
            class_index: int = self.get_classes_index()[class_name]
            class_files: list[str] = file_dict[class_index]

            # Randomly sample files from this class
            num_samples: int = min(samples, len(class_files))
            rng = np.random.default_rng(self.get_random_seed())
            random_indices: np.ndarray = rng.choice(len(class_files), size=num_samples, replace=False)

            print(f"\nPlotting {num_samples} MFCC samples from class '{class_name}'...")

            # For each randomly selected file
            for i, file_idx in enumerate(random_indices):

                # Get the file path
                file_path: str = class_files[file_idx]

                try:
                    # Load audio file
                    signal, sr = librosa.load(file_path, sr=self.get_sample_rate(), duration=self.get_duration())

                except Exception as e:
                    print(f"\nWarning: Failed to load audio file: {file_path}")
                    print(f"Error: {e}")
                    print("Returning zeros as fallback.\n")

                    # Return zeros with expected shape as fallback
                    signal: np.ndarray = np.zeros(self.get_sample_rate() * self.get_duration())

                    sr: int | float = self.get_sample_rate()

                # Ensure fixed length
                target_length: int = self.get_sample_rate() * self.get_duration()

                # If wave is shorter than target length, pad it; if longer, truncate it
                if len(signal) < target_length:
                    signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
                else:
                    signal = signal[:target_length]

                # Compute MFCCs
                mfccs: np.ndarray = librosa.feature.mfcc(
                    y=signal,
                    sr=sr,
                    n_mfcc=self.get_n_mfcc(),
                    n_fft=self.get_n_fft(),
                    hop_length=self.get_hop_length()
                )

                # Plot MFCCs
                fig, ax = plt.subplots(figsize=(10, 4))
                img = librosa.display.specshow(
                    mfccs,
                    x_axis='time',
                    ax=ax
                )

                # Add color bar
                fig.colorbar(img, ax=ax)
                ax.set(title=f'MFCC - Class: {class_name} - Sample {i+1}', ylabel='MFCC Coefficients')
                ax.title.set_fontsize(18)
                ax.title.set_fontweight('bold')
                plt.tight_layout()
                plt.show()

        print("\n---------------------------- MFCC Plotting Complete ----------------------------\n")

    def mean_and_std_stats(self) -> None:
        """
        Calculate and print the mean and standard deviation of not just the raw data,
        but also the scaled data if scaling is applied.
        Finally, we print out the mean and std for a set of 5 batches from the training DataLoader.

        Parameters:
            None

        Returns:
            None
        """

        print("\n=============================== Calculating Mean and Standard Deviation Statistics ===============================\n")

        # Collect raw data (before scaling and batching) from 100 random samples
        raw_data: List[np.ndarray] = []
        scaled_data: List[np.ndarray] = []
        num_samples: int = min(100, len(self))
        rng = np.random.default_rng(self.get_random_seed())
        sample_indices: np.ndarray = rng.choice(len(self), size=num_samples, replace=False)

        # Temporarily disable scaling to get raw data
        use_scaler_original: bool = self.get_use_scaler()
        self.set_use_scaler(False)

        # For each randomly selected sample
        for idx in sample_indices:
            # Get data without scaling
            data, _ = self.__getitem__(idx)

            # If data is a tensor, convert to numpy and flatten
            if isinstance(data, torch.Tensor):
                raw_data.append(data.numpy().flatten())

        # Re-enable scaling to get scaled data
        self.set_use_scaler(use_scaler_original)

        # For each randomly selected sample, if scaling is enabled
        if use_scaler_original and self.get_scaler() is not None:
            for idx in sample_indices:

                # Get data with scaling
                data, _ = self.__getitem__(idx)

                # If data is a tensor, convert to numpy and flatten
                if isinstance(data, torch.Tensor):
                    scaled_data.append(data.numpy().flatten())

        # Print raw data stats
        if raw_data:
            raw_concat: np.ndarray = np.concatenate(raw_data)
            print(f"Raw Data Stats ({len(raw_data)} samples):")
            print(f"  Mean: {np.mean(raw_concat):.6f}")
            print(f"  Std:  {np.std(raw_concat):.6f}\n")

        # Print scaled data stats
        if scaled_data:
            scaled_concat: np.ndarray = np.concatenate(scaled_data)
            print(f"Scaled Data Stats ({len(scaled_data)} samples):")
            print(f"  Mean: {np.mean(scaled_concat):.6f}")
            print(f"  Std:  {np.std(scaled_concat):.6f}\n")

        # Collect data from first 5 batches
        training_loader: DataLoader = self.get_training_loader()

        # Ensure the training loader is available
        if training_loader is not None:
            batch_data = []

            # Iterate through first 5 batches
            for batch_idx, (data, _) in enumerate(training_loader):

                # If we've collected 5 batches, stop
                if batch_idx >= 5:
                    break

                # If data is a tensor, convert to numpy
                if isinstance(data, torch.Tensor):
                    batch_data.append(data.numpy())

            # If we have batch data, compute and print stats
            if batch_data is not None and len(batch_data) > 0:
                # Concatenate all batch data
                all_data: np.ndarray = np.concatenate([b.reshape(b.shape[0], -1) for b in batch_data], axis=0)

                # Show the length and shape of the data
                print(f"Batched Data Stats ({len(batch_data)} batches, {all_data.shape[0]} samples):")

                # Print mean and std
                print(f"  Mean: {np.mean(all_data):.6f}")
                print(f"  Std:  {np.std(all_data):.6f}")

        print("\n---------------------------- Statistics Complete ----------------------------\n")

    def print_optimizer_loss_architecture(self, past_model: str = 'None') -> None:
        """
        Print the chosen optimizer, loss function, and model architecture.

        Parameters:
            past_model (str): Path to a saved model to load before evaluation. Defaults to 'None'.

        Returns:
            None
        """

        print("\n=============================== Optimizer, Loss Function, and Model Architecture ===============================\n")

        # Load past model if provided
        if past_model != 'None':
            print(f"Loading model from: {past_model}")
            self.load_state_dict(torch.load(past_model, map_location=self.get_device()), strict=False)
            print("Model loaded successfully.\n")

        # If we don't have optimizer or loss set, set them to defaults
        if self.get_optim() is None or not isinstance(self.get_optim(), torch.optim.Optimizer) or self.get_loss() is None or not isinstance(self.get_loss(), nn.Module):
            optimizer: type[torch.optim.Optimizer] = self.set_optim('Adam')
            loss_function: nn.Module = self.set_loss('CrossEntropyLoss')

        # Otherwise, get the current optimizer and loss
        else:
            optimizer: type[torch.optim.Optimizer] = self.get_optim()
            loss_function: nn.Module = self.get_loss()

        # Get the number of parameters of the model
        num_param_grad: int = sum(p.numel() for p in self.parameters() if p.requires_grad)
        num_param_all: int = sum(p.numel() for p in self.parameters())


        print(f"Chosen Optimizer: {optimizer}")
        print(f"Chosen Loss Function: {loss_function}\n")

        print(f"Number of Parameters (Gradients): {num_param_grad}")
        print(f"Number of Parameters (All): {num_param_all}\n")

        # Print model architecture
        print(f"Model Architecture: {self}")



        print("\n---------------------------- Optimizer, Loss Function, and Architecture Complete ----------------------------\n")


    def create_confusion_matrix(self, past_model: str = 'None') -> None:
        """
        Create and display a confusion matrix for the model's predictions on the test set.

        Parameters:
            past_model (str): Path to a saved model to load before evaluation. Defaults to 'None'.

        Returns:
            None
        """

        print("\n=============================== Creating Confusion Matrix ===============================\n")

        # Load past model and use it if provided
        if past_model != 'None':
            print(f"Loading model from: {past_model}")
            self.load_state_dict(torch.load(past_model, map_location=self.get_device()), strict=False)
            print("Model loaded successfully.\n")


        # Generate predictions if not already done and if a past model is provided
        if (self.get_testing_loader() is None or len(self.all_labels) == 0 or len(self.all_predictions) == 0) and past_model != 'None':
            self.setup_data_loaders()

            # Ensure the testing loader is available
            if self.get_testing_loader() is None:
                raise ValueError("Testing DataLoader is not available. Please call setup_data_loaders() first.")

            self.eval()  # Set model to evaluation mode

            self.all_labels: list[int] = []
            self.all_predictions: list[int] = []

            # Generate predictions
            with torch.no_grad() :

                # Iterate through the testing data
                for data, labels in tqdm(self.get_testing_loader(), desc="Generating Confusion Matrix", unit="batch"):

                    # Move data to the appropriate device
                    data: torch.Tensor = data.to(self.get_device())
                    labels: torch.Tensor = labels.to(self.get_device())

                    # Get model outputs
                    outputs: torch.Tensor = self(data)

                    # Get predicted classes
                    _, predicted = torch.max(outputs, dim=1)

                    # Store true labels and predictions
                    self.all_labels.extend(labels.cpu().numpy())
                    self.all_predictions.extend(predicted.cpu().numpy())

        # Compute confusion matrix
        cm: np.ndarray = confusion_matrix(self.all_labels, self.all_predictions)

        # Create figure with two subplots side by side
        _, (ax_counts, ax_percentage) = plt.subplots(1, 2, figsize=(20, 8))

        # Plot raw counts confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.get_classes(), yticklabels=self.get_classes(), ax=ax_counts)
        ax_counts.set_xlabel('Predicted Label')
        ax_counts.set_ylabel('True Label')
        ax_counts.set_title('Confusion Matrix (Counts)', fontsize=18, fontweight='bold')

        # Create percentage-based confusion matrix
        cm_percentage: np.ndarray = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Plot percentage-based confusion matrix
        sns.heatmap(cm_percentage, annot=True, fmt='.2%', cmap='Greens',
                    xticklabels=self.get_classes(), yticklabels=self.get_classes(), ax=ax_percentage)
        ax_percentage.set_xlabel('Predicted Label')
        ax_percentage.set_ylabel('True Label')
        ax_percentage.set_title('Confusion Matrix (Percentage)', fontsize=18, fontweight='bold')


        plt.tight_layout()
        plt.subplots_adjust(left=0.1) # Add extra margin on the left to prevent labels from being cut off
        plt.show()

        print ("\n---------------------------- Confusion Matrix Creation Complete ----------------------------\n")


    def create_classification_report(self, past_model: str = 'None') -> None:
        """
        Create and display a classification report for the model's predictions on the test set.

        Parameters:
            past_model (str): Path to a saved model to load before evaluation. Defaults to 'None'.

        Returns:
            None
        """

        print("\n=============================== Creating Classification Report ===============================\n")

        # Load past model if provided
        if past_model != 'None':
            print(f"Loading model from: {past_model}")
            self.load_state_dict(torch.load(past_model, map_location=self.get_device()), strict=False)
            print("Model loaded successfully.\n")

        # Check if predictions are already generated and generate them if not
        if (self.get_testing_loader() is None or len(self.all_labels) == 0 or len(self.all_predictions) == 0) and past_model != 'None':

            # Generate predictions if not already done
            self.setup_data_loaders()

            # Ensure the testing loader is available
            if self.get_testing_loader() is None:
                raise ValueError("Testing DataLoader is not available. Please call setup_data_loaders() first.")

            self.eval()  # Set model to evaluation mode

            # Set up lists to store true labels and predictions
            self.all_labels: list[int] = []
            self.all_predictions: list[int] = []

            # Generate predictions
            with torch.no_grad():

                # For each batch in the testing loader
                for data, labels in tqdm(self.get_testing_loader(), desc="Generating Classification Report", unit="batch"):

                    # Move data to the appropriate device
                    data: torch.Tensor = data.to(self.get_device())
                    labels: torch.Tensor = labels.to(self.get_device())

                    # Get model outputs and predicted classes
                    outputs: torch.Tensor = self(data)
                    _, predicted = torch.max(outputs, dim=1)

                    # Store true labels and predictions
                    self.all_labels.extend(labels.cpu().numpy())
                    self.all_predictions.extend(predicted.cpu().numpy())

        # Generate classification report
        report: str | Dict[str, float] = classification_report(self.all_labels, self.all_predictions, target_names=self.get_classes())
        print("Classification Report:\n")
        print(report)

        # Make classification report into matplotlib table
        report_dict: str | Dict[str, float] = classification_report(self.all_labels, self.all_predictions, target_names=self.get_classes(), output_dict=True)
        report_df: pd.DataFrame = pd.DataFrame(report_dict).transpose()
        plt.figure(figsize=(12, 10))
        plt.axis('off')
        # Create table with formatted font sizes and weights
        table = plt.table(cellText=np.round(report_df.values, 2), colLabels=report_df.columns, rowLabels=report_df.index, loc='center')
        table.scale(1.1, 1.1)
        table.auto_set_font_size(False)
        table.set_fontsize(14)

        plt.show()

        print ("\n---------------------------- Classification Report Creation Complete ----------------------------\n")


    def sns_scatter_plot(self, past_model: str = 'None') -> None:
        """
        Create and display a seaborn scatter plot using t-SNE for the model's predictions on the test set.

        Parameters:
            past_model (str): Path to a saved model to load before evaluation. Defaults to 'None'.

        Returns:
            None
        """

        print("\n=============================== Creating t-SNE Scatter Plot ===============================\n")

        # Load past model if provided
        if past_model != 'None':
            print(f"Loading model from: {past_model}")
            self.load_state_dict(torch.load(past_model, map_location=self.get_device()), strict=False)
            print("Model loaded successfully.\n")

        # Check if testing loader is available
        if self.get_testing_loader() is None:
            self.setup_data_loaders()

            # If not, raise error
            if self.get_testing_loader() is None:
                self.setup_data_loaders()
                raise ValueError("Testing DataLoader is not available. Please call setup_data_loaders() first.")

        # Start evaluation of model
        self.eval()  # Set model to evaluation mode

        # Set up separate lists to store batches of features and labels
        features_list: list = []
        labels_list: list = []


        # Generate features and calculate accuracy
        with torch.no_grad():
            # Iterates through data
            for samples, labels in tqdm(self.get_testing_loader(), desc="Generating Features for t-SNE Plot", unit="batch"):

                # Move data to the appropriate device
                samples: torch.Tensor = samples.to(self.get_device())
                labels: torch.Tensor = labels.to(self.get_device())

                # Forward pass to calculate accuracy
                outputs: torch.Tensor = self(samples)

                # Extract features using the model's feature extractor
                features: torch.Tensor = self.extract_features(samples)

                # Store features and labels (move to CPU)
                labels_list.append(labels.cpu().numpy())
                features_list.append(features.cpu().numpy())


        # Concatenate all feature outputs and labels
        X_output: np.ndarray = np.concatenate(features_list, axis=0)
        y_output: np.ndarray = np.concatenate(labels_list)

        print(f"Feature shape: {X_output.shape}")
        print(f"Applying t-SNE to reduce {X_output.shape[1]} dimensions to 2D...")

        # Initialize t-SNE
        tsne: TSNE = TSNE(n_components=2, perplexity=30, random_state=self.get_random_seed(), init='pca')

        # Fit and transform the feature outputs
        X_embedded: np.ndarray = tsne.fit_transform(X_output)

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Use seaborn's colorblind-friendly palette
        scatter = sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y_output, palette='colorblind', s=60)

        # Set plot titles and labels
        plt.title("t-SNE Clustering of DeepFake Samples (Feature Space)", fontsize=16, fontweight='bold')
        plt.xlabel("t-SNE Component 1", fontsize=12)
        plt.ylabel("t-SNE Component 2", fontsize=12)

        # Update legend with class names
        handles, _ = scatter.get_legend_handles_labels()
        class_names: list[str] = self.get_classes()
        plt.legend(handles, class_names, title="Audio Class", bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()
        plt.show()

        print("\n---------------------------- t-SNE Scatter Plot Creation Complete ----------------------------\n")


    def plot_roc_curve_with_eer(self, past_model: str = 'None') -> None:
        """
        Plot ROC curve for binary classification (gt vs deepfake) and compute Equal Error Rate (EER).
        Combines all 6 deepfake classes into one 'deepfake' category vs ground truth 'real' category.

        Parameters:
            past_model (str): Path to a saved model to load before evaluation. Defaults to 'None'.

        Returns:
            None
        """

        print("\n=============================== Plotting ROC Curve with EER ===============================\n")

        # Load past model if specified
        if past_model != 'None':
            print(f"Loading model from: {past_model}")
            self.load_state_dict(torch.load(past_model, map_location=self.get_device()), strict=False)
            print("Model loaded successfully.\n")

        # Check if predictions are already generated and generate them if not (Note: we need probabilities here so we regenerate if needed)
        if (self.get_testing_loader() is None or len(self.all_labels) == 0 or self.all_probabilities.size == 0) and past_model != 'None':

            # Generate predictions if not already done
            self.setup_data_loaders()

            # Ensure the testing loader is available
            if self.get_testing_loader() is None:
                raise ValueError("Testing DataLoader is not available. Please call setup_data_loaders() first.")

            self.eval()  # Set model to evaluation mode

            # Set up lists to store true labels
            self.all_labels: List[int] = []

            # Store probability scores for ROC curve
            all_probabilities: List[np.ndarray] = []

            # Generate predictions
            with torch.no_grad():

                # For each batch in the testing loader
                for data, labels in tqdm(self.get_testing_loader(), desc="Generating ROC Predictions", unit="batch"):

                    # Move data to the appropriate device
                    data: torch.Tensor = data.to(self.get_device())
                    labels: torch.Tensor = labels.to(self.get_device())

                    # Get model outputs and predicted classes
                    outputs: torch.Tensor = self(data)

                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(outputs, dim=1)

                    # Store true labels, predictions, and probabilities
                    self.all_labels.extend(labels.cpu().numpy())
                    all_probabilities.append(probabilities.cpu().numpy())

            # Concatenate all probabilities
            self.all_probabilities = np.concatenate(all_probabilities, axis=0)

        # Get the index for 'gt' class (ground truth/real)
        gt_class_index: int | None = self.get_classes_index().get('gt')

        # If it doesn't exist, raise error
        if gt_class_index is None:
            raise ValueError("Ground truth class 'gt' not found in classes.")

        # Prepare binary labels and deepfake probabilities
        all_labels_binary: np.ndarray = np.array([])

        # Convert multi-class labels to binary: gt (real) = 0, all others (deepfake) = 1
        for label in self.all_labels:
            if label == gt_class_index:
                all_labels_binary = np.append(all_labels_binary, [0])
            else:
                all_labels_binary = np.append(all_labels_binary, [1])

        # Get deepfake probability scores: P(deepfake) = 1 - P(gt)
        deepfake_probabilities: np.ndarray = 1.0 - self.all_probabilities[:, gt_class_index]

        # Set up variables for ROC computation
        fpr: np.ndarray = np.ndarray([])
        tpr: np.ndarray = np.ndarray([])
        thresholds: np.ndarray = np.ndarray([])

        # Compute ROC curve and AUC
        fpr, tpr, thresholds = roc_curve(all_labels_binary, deepfake_probabilities)
        roc_auc: float = float(auc(fpr, tpr))

        # Compute EER and ERR threshold using the two-line approach (Thanks to Changjiang at https://yangcha.github.io/EER-ROC/)!
        eer: tuple = (brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.))
        eer_threshold: float = float(interp1d(fpr, thresholds)(eer))


        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='orange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (FPR)', fontsize=14)
        plt.ylabel('True Positive Rate (TPR)', fontsize=14)
        plt.title('ROC Curve: Real vs Deepfake Classification', fontsize=18, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

        # Print ROC and EER statistics
        print(f"ROC Curve Statistics:")
        print(f"--------------------------------------------")
        print(f"  Area Under Curve (AUC): {roc_auc:.4f}")
        print(f"  Equal Error Rate (EER): {eer:.4f} ({eer*100:.2f}%)")
        print(f"  EER Threshold: {eer_threshold:.4f}")
        print(f"--------------------------------------------")

        print("\n---------------------------- ROC Curve with EER Complete ----------------------------\n")


    def find_optimal_threshold(self, past_model: str, target_recall: float = 0.99) -> float:
        """
        Finds the probability threshold for the 'GT' class that achieves the target recall 
        (e.g., accepting 99% of real users), even if it means slightly increasing false positives.
        """

        print("\n=============================== Finding Optimal Threshold for Target Real User Recall ===============================\n")

         # Load past model if specified
        if past_model != 'None':
            print(f"Loading model from: {past_model}")
            self.load_state_dict(torch.load(past_model, map_location=self.get_device()), strict=False)
            print("Model loaded successfully.\n")

       
        self.eval()
        val_loader = self.get_validation_loader()
        
        gt_probs = []
        true_labels = []
        
        # 2. Collect Probabilities (Inference)
        with torch.no_grad():
            for data, labels in val_loader:
                data = data.to(self.get_device())
                outputs = self(data)
                
                # Convert logits to probabilities
                probs = torch.softmax(outputs, dim=1)
                
                # Get probability of the GT class (Index 1 based on your logs)
                # CAUTION: Verify 'gt' is actually index 1 in your self.class_names
                gt_index = 1 
                batch_gt_probs = probs[:, gt_index].cpu().numpy()
                
                gt_probs.extend(batch_gt_probs)
                true_labels.extend(labels.cpu().numpy())

        gt_probs = np.array(gt_probs)
        true_labels = np.array(true_labels)

        # 3. Create a binary mask (1 for Real, 0 for Fake)
        is_gt = (true_labels == 1)

        # 4. Sweep Thresholds
        best_threshold = 0.0
        best_fake_acc = 0.0
        
        # Check thresholds from 0.01 to 0.99
        thresholds = np.linspace(0.01, 0.99, 100)
        
        print(f"\n--- Searching for Threshold to achieve {target_recall*100}% Real Accuracy ---")
        
        for t in thresholds:
            # If Prob(Real) > t, we classify as Real. Otherwise Fake.
            predicted_real = gt_probs > t
            
            # Calculate Real Accuracy (Recall)
            # How many Real samples did we correctly accept?
            real_acc = np.mean(predicted_real[is_gt])
            
            # Calculate Fake Accuracy (Specificity)
            # How many Fakes did we correctly reject? (i.e. predicted_real is False)
            fake_acc = np.mean(~predicted_real[~is_gt])
            
            # We want the highest Fake Accuracy that still meets our Real Accuracy target
            if real_acc >= target_recall:
                best_threshold = t
                best_fake_acc = fake_acc
                print(f"Thresh {t:.2f}: Real Acc: {real_acc:.4f} | Fake Detect: {fake_acc:.4f}")

        print(f"\nOptimal Threshold found: {best_threshold:.4f}")
        print(f"At this threshold:")
        print(f"  - Real Users Accepted: {target_recall*100}%")
        print(f"  - Deepfakes Detected:  {best_fake_acc*100:.2f}%")
        
        return best_threshold

    def save_model(self, save_path: str = '/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V10.pth') -> None:
        """
        Save the trained model to disk or Google Drive.

        Parameters:
            save_path (str): Path where the model should be saved.

        Returns:
            None
        """

        print(f"\n=============================== Saving Model ===============================\n")

        # Save the model state dictionary
        torch.save(self.state_dict(), save_path)
        print(f"Model saved successfully to: {save_path}")

        print("\n---------------------------- Model Save Complete ----------------------------\n")

# For testing
# if __name__ == "__main__":
#     detector = DeepFakeDetectorGraphsAndStats(directory='/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc', file_extension='.wav', loss='CrossEntropyLoss', optim='Adam', DL_type='RNN', random_seed=42)
#     detector.set_batch_size(16) # Make it 16 samples per batch
#     detector.set_learning_rate(0.0005) # Lower learning rate for stability
#     detector.set_non_gt_subset_count(5000)
#     detector.set_gt_subset_count(5000)
#     detector.setup_file_dictionary(".wav")


#     detector.setup_data_loaders()
#     detector.plot_class_counts()

#     detector.mean_and_std_stats()
#     detector.test_batch_equality(8)
#     detector.plot_random_mfcc_samples(samples=2) # Plot 2 random MFCC samples per class

#     detector.evaluate_model('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
#     detector.print_optimizer_loss_architecture('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
#     detector.sns_scatter_plot('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
#     detector.create_confusion_matrix('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
#     detector.create_classification_report('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')
#     detector.plot_roc_curve_with_eer('/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/Deep_Fake_Detector_LSTM_V15.pth')

