from classes_files_dictionary_set_up import *
from DeepFake_Detector_Imports import *

class BatchLossAndOptimization(ClassesFilesDictionarySetUp, nn.Module):
    """
    A class to manage batch size, learning rate, and data splitting for training, validation, and testing sets,
    as well as set up Loss and Optimization for a DeepFake Detector.
    Inherits from ClassesFilesDictionarySetUp and nn.Module.
    """

    def __init__(self, directory: str = '', optim: str = 'Adam', loss: str = 'CrossEntropyLoss', file_extension: str = '.wav', DL_type: str = 'RNN', data_type: str = 'mfcc', two_class: bool = False, random_seed: int = 42) -> None:
        """
        Desc:
            Initialize the BatchLossAndOptimization class, including the class variables.
            Also, sets up the loss function and optimizer.

        Parameters:
            directory (str): The root directory containing the dataset files.
            optim (str): The name of the optimizer to use for training.
            loss (str): The name of the loss function to use for training.
            file_extension (str): The file extension of the dataset files.
            DL_type (str): The type of deep learning model ('RNN', '1DCNN', '2DCNN').
            two_class (bool): Whether to use a two-class system or not.
            data_type (str): The type of data ('mfcc', 'mel-spectrogram, 'waveform', 'bytes').

        Returns:
            None
        """

        # Initialize parent classes, including nn.Module, also send up directory, file_extension, and DL_type to ClassesFilesDictionarySetUp
        super(BatchLossAndOptimization, self).__init__(directory, file_extension, DL_type, data_type, two_class, random_seed)

        # Initialize batch size, and learning rate
        self.__batch_size: int = 32
        self.__learning_rate: float = 0.001

        # Initialize data split proportions
        self.__train_size: float = 0.7 # 70% for training (what the model learns from)
        self.__valid_size: float = 0.15 # 15% for validation (tuning hyperparameters and early stopping)
        self.__test_size: float = 0.15 # 15% for testing (final unbiased evaluation)

        # Initialize DataLoaders
        self.__training_loader: Optional[DataLoader] = None # DataLoader for training set (the model learns from this data)
        self.__validation_loader: Optional[DataLoader] = None # DataLoader for validation set (used for hyperparameter tuning and early stopping)
        self.__testing_loader: Optional[DataLoader] = None # DataLoader for testing set (final unbiased evaluation of the model)

        # Initialize class weights for loss function
        self.__class_weights: Optional[torch.Tensor] = None

        # Set up optimizer and loss function
        self.__optim: type[torch.optim.Optimizer] = self.set_optim(optim)
        self.__loss_name: str = 'None'
        self.__loss: nn.Module = self.set_loss(loss)

    def get_batch_size(self) -> int:
        """
        Get the batch size, or the number of samples/files per batch
        e.g. if batch size is 32, each batch will contain 32 samples/files.

        Parameters:
            None

        Returns:
            int: The batch size.
        """

        return self.__batch_size

    def set_batch_size(self, batch_size: int) -> None:
        """
        Set the batch size, or the number of samples/files per batch.
        e.g. if batch size is 32, each batch will contain 32 samples/files

        Parameters:
            batch_size (int): The batch size to set.
        Returns:
            None
        """

        if batch_size <= 0:
            self.__batch_size = 1  # Ensure batch size is positive
        else:
            self.__batch_size = batch_size

    def get_learning_rate(self) -> float:
        """
        Get the learning rate, or the step size at each iteration while moving toward a minimum of a loss function.
        e.g. a learning rate of 0.001 means the model weights are updated by 0.1% of the value of the gradient at each step.

        Parameters:
            None

        Returns:
            float: The learning rate.
        """

        return self.__learning_rate

    def set_learning_rate(self, learning_rate: float) -> None:
        """
        Set the learning rate, or the step size at each iteration while moving toward a minimum of a loss function.
        e.g. a learning rate of 0.001 means the model weights are updated by 0.1% of the value of the gradient at each step.

        Parameters:
            learning_rate (float): The learning rate to set.

        Returns:
            None
        """

        if learning_rate <= 0.0:
            self.__learning_rate = 0.001  # Ensure learning rate is positive
        else:
            self.__learning_rate = learning_rate

    def get_train_size(self) -> float:
        """
        Get the training set size proportion, or the proportion of the dataset used for training.
        e.g. a train size of 0.7 means 70% of the dataset is used for training.

        Parameters:
            None

        Returns:
            float: The training set size proportion.
        """

        return self.__train_size

    def set_train_size(self, train_size: float) -> None:
        """
        Set the training set size proportion, or the proportion of the dataset used for training.
        e.g. a train size of 0.7 means 70% of the dataset is used for training.

        Parameters:
            train_size (float): The training set size proportion to set.

        Returns:
            None
        """

        # Ensure training is a "reasonable" proportion
        if 0.0 < train_size < 1.0:
            self.__train_size = train_size
        else:
            self.__train_size = 0.7  # Default to 70% if invalid

    def get_validation_size(self) -> float:
        """
        Get the validation set size proportion, or the proportion of the dataset used for validation.
        e.g. a validation size of 0.15 means 15% of the dataset is used for validation.

        Parameters:
            None

        Returns:
            float: The validation set size proportion.
        """

        return self.__valid_size

    def set_validation_size(self, validation_size: float) -> None:
        """
        Set the validation set size proportion, or the proportion of the dataset used for validation.
        e.g. a validation size of 0.15 means 15% of the dataset is used for validation.

        Parameters:
            validation_size (float): The validation set size proportion to set.

        Returns:
            None
        """

        # Ensure validation is a "reasonable" proportion
        if 0.0 < validation_size < 1.0:
            self.__valid_size = validation_size
        else:
            self.__valid_size = 0.15  # Default to 15% if invalid

    def get_test_size(self) -> float:
        """
        Get the testing set size proportion, or the proportion of the dataset used for testing.
        e.g. a test size of 0.15 means 15% of the dataset is used for testing.

        Parameters:
            None

        Returns:
            float: The testing set size proportion.
        """

        return self.__test_size

    def set_test_size(self, test_size: float) -> None:
        """
        Set the testing set size proportion, or the proportion of the dataset used for testing.
        e.g. a test size of 0.15 means 15% of the dataset is used for testing.

        Parameters:
            test_size (float): The testing set size proportion to set.

        Returns:
            None
        """

        # Ensure testing is a "reasonable" proportion
        if 0.0 < test_size < 1.0:
            self.__test_size = test_size
        else:
            self.__test_size = 0.15  # Default to 15% if invalid


    def get_training_loader(self) -> DataLoader:
        """
        Get the training data loader, which provides batches of training data during model training.
        The information in the DataLoader looks like: (batch_size, channels, time_steps, features) for RNN

        Parameters:
            None

        Returns:
            DataLoader: The training data loader.
        """

        # Check if the training loader is valid
        if self.__training_loader is None or not isinstance(self.__training_loader, DataLoader):
            raise ValueError("Training loader has not been set.")

        return self.__training_loader

    def set_training_loader(self, training_loader: DataLoader) -> None:
        """
        Set the training data loader, which provides batches of training data during model training.
        The information in the DataLoader looks like: (batch_size, time_steps, features) for RNN

        Parameters:
            training_loader (DataLoader): The training data loader to set.
            If None or not a DataLoader, raises an error.

        Returns:
            None
        """

        # Make sure we have a valid DataLoader
        if training_loader is None or not isinstance(training_loader, DataLoader):
            raise ValueError("Training loader cannot be None.")
        else:
            self.__training_loader = training_loader

    def get_validation_loader(self) -> DataLoader:
        """
        Get the validation data loader, which provides batches of validation data during model training.
        The information in the DataLoader looks like: (batch_size, time_steps, features) for RNN

        Parameters:
            None

        Returns:
            DataLoader: The validation data loader.
        """

        # Check if the validation loader is valid
        if self.__validation_loader is None or not isinstance(self.__validation_loader, DataLoader):
            raise ValueError("Validation loader has not been set.")

        return self.__validation_loader

    def set_validation_loader(self, validation_loader: DataLoader) -> None:
        """
        Set the validation data loader, which provides batches of validation data during model training.
        The information in the DataLoader looks like: (batch_size, time_steps, features) for RNN

        Parameters:
            validation_loader (DataLoader): The validation data loader to set.
            If None or not a DataLoader, raises an error.

        Returns:
            None
        """

        # Make sure we have a valid DataLoader
        if validation_loader is None or not isinstance(validation_loader, DataLoader):
            raise ValueError("Validation loader cannot be None.")
        else:
            self.__validation_loader = validation_loader

    def get_testing_loader(self) -> DataLoader:
        """
        Get the testing data loader, which provides batches of testing data during model evaluation.
        The information in the DataLoader looks like: (batch_size, time_steps, features) for RNN

        Parameters:
            None

        Returns:
            DataLoader: The testing data loader.
        """

        # Check if the testing loader is valid
        if self.__testing_loader is None or not isinstance(self.__testing_loader, DataLoader):
            raise ValueError("Testing loader has not been set.")

        return self.__testing_loader

    def set_testing_loader(self, testing_loader: DataLoader) -> None:
        """
        Set the testing data loader, which provides batches of testing data during model evaluation.
        The information in the DataLoader looks like: (batch_size, time_steps, features) for RNN

        Parameters:
            testing_loader (DataLoader): The testing data loader to set.
            If None or not a DataLoader, raises an error.

        Returns:
            None
        """

        # Make sure we have a valid DataLoader
        if testing_loader is None or not isinstance(testing_loader, DataLoader):
            raise ValueError("Testing loader cannot be None.")
        else:
            self.__testing_loader = testing_loader

    def get_optim(self) -> type[torch.optim.Optimizer]:
        """
        Get the optimizer for training.
        The optimizer updates the model weights based on the computed gradients during backpropagation.
        Backpropagation is the process of calculating how much each weight in the network contributed to the overall error,
        allowing efficient computation of gradients for deep networks.
        e.g. Adam optimizer adapts the learning rate for each parameter.

        Parameters:
            None

        Returns:
            type: The optimizer class.
        """

        return self.__optim

    def set_optim(self, optimizer_name: str) -> type[torch.optim.Optimizer]:
        """
        Lets the user decide what optimizer they wish to use for training.
        The optimizer updates the model weights based on the computed gradients during backpropagation.
        Backpropagation is the process of calculating how much each weight in the network contributed to the overall error,
        allowing efficient computation of gradients for deep networks.
        e.g. Adam optimizer adapts the learning rate for each parameter.

        Parameters:
            optimizer_name (str): The name of the optimizer. Includes 'SGD, 'Adam, 'NAdam', 'RAdam', 'AdamW', 'Adagrad', 'Adamax, 'Rprop', 'Rmsprop', and 'ASGD'.
            (Adam likely is the best choice for deep fake detection due to its adaptive learning rate capabilities.)

        Returns:
            type[torch.optim.Optimizer]: The chosen optimizer class.
        """

        # Define available optimizers
        optimizers: dict[str, type] = {
            'SGD': optim.SGD,
            'Adam': optim.Adam,
            'NAdam': optim.NAdam,
            'RAdam': optim.RAdam,
            'AdamW': optim.AdamW,
            'Adagrad': optim.Adagrad,
            'Adamax': optim.Adamax,
            'Rprop': optim.Rprop,
            'RMSprop': optim.RMSprop,
            'ASGD': optim.ASGD,
        }

        # If we do not have a valid optimizer, raise an error
        if optimizer_name not in optimizers:
            raise ValueError(f"Optimizer '{optimizer_name}' is not supported. Choose from: {list(optimizers.keys())}")
        else:
            return optimizers[optimizer_name]

    def get_class_weights(self) -> torch.Tensor:
        """
        Get the class weights tensor for loss function.
        Class weights are used to balance the importance of different classes in the loss function.
        It is a tensor where each element corresponds to the weight for a specific class.
        e.g. torch.tensor([1.0, 2.0, 0.5]) means class 0 has weight 1.0, class 1 has weight 2.0, and class 2 has weight 0.5.

        Parameters:
            None

        Returns:
            Optional[torch.Tensor]: The class weights tensor.
        """

        # Ensure class weights are valid
        if self.__class_weights is None or not isinstance(self.__class_weights, torch.Tensor):
            raise ValueError("Class weights have not been set.")

        return self.__class_weights

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        """
        Set the class weights tensor for loss function.
        Class weights are used to balance the importance of different classes in the loss function.
        It is a tensor where each element corresponds to the weight for a specific class.
        e.g. torch.tensor([1.0, 2.0, 0.5]) means class 0 has weight 1.0, class 1 has weight 2.0, and class 2 has weight 0.5.

        Parameters:
            class_weights (Optional[torch.Tensor]): The class weights tensor to set.

        Returns:
            None
        """

        # Ensure class weights are valid
        if class_weights is None and not isinstance(class_weights, torch.Tensor):
            raise ValueError("Class weights have not been set.")

        self.__class_weights = class_weights

    def get_loss_name(self) -> str:
        """
        Get the name of the loss function.
        e.g. 'CrossEntropyLoss' is commonly used for multi-class classification tasks.

        Parameters:
            None

        Returns:
            str: The name of the loss function.
        """
        return self.__loss_name

    def get_loss(self) -> nn.Module:
        """
        Get the loss function for training.
        The loss function measures how well the model's predictions match the true labels.
        The higher the loss, the worse the model is performing.
        For example, 2.0 is usually worse, while 0.3 is usually good
        e.g. CrossEntropyLoss is commonly used for multi-class classification tasks.

        Parameters:
            None

        Returns:
            nn.Module: The loss function instance.
        """
        return self.__loss

    def set_loss(self, loss_name: str, class_weights: Optional[torch.Tensor] = None) -> nn.Module:
        """
        Lets the user decide what loss function they wish to use for training.
        The loss function measures how well the model's predictions match the true labels.
        The higher the loss, the worse the model is performing.
        For example, 2.0 is usually worse, while 0.3 is usually
        e.g. CrossEntropyLoss is commonly used for multi-class classification tasks.

        Parameters:
            loss_name (str): The name of the loss function. Includes 'L1Loss', 'MSELoss', 'CrossEntropyLoss', 'NLLLoss', 'BCELoss', 'BCEWithLogitsLoss', 'HingeEmbeddingLoss', and 'SmoothL1Loss'.
            class_weights (Optional[torch.Tensor]): Class weights for CrossEntropyLoss. Ignored for other loss functions.

        Returns:
            nn.Module: The chosen loss function.
        """

        # Define available loss functions
        # Label smoothing (0.1) helps prevent overconfident predictions without limiting learning too much
        losses: dict[str, nn.Module] = {
            'L1Loss': nn.L1Loss(),
            'MSELoss': nn.MSELoss(),
            'CrossEntropyLoss': nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2) if class_weights is not None else nn.CrossEntropyLoss(label_smoothing=0.2),
            'NLLLoss': nn.NLLLoss(),
            'BCELoss': nn.BCELoss(),
            'BCEWithLogitsLoss': nn.BCEWithLogitsLoss(),
            'HingeEmbeddingLoss': nn.HingeEmbeddingLoss(),
            'SmoothL1Loss': nn.SmoothL1Loss(),
        }

        if loss_name not in losses:
            raise ValueError(f"Loss function '{loss_name}' is not supported. Choose from: {list(losses.keys())}")

        # Store the loss name and return the loss function
        self.__loss_name = loss_name
        self.__loss = losses[loss_name]
        return self.__loss


    def split_data(self, train_size: float = 0.7, validation_size: float = 0.15, test_size: float = 0.15) -> None:
        """
        Split the dataset into training, validation, and testing sets.

        Parameters:
            train_size (float): Proportion of the dataset to include in the training set.
            validation_size (float): Proportion of the dataset to include in the validation set.
            test_size (float): Proportion of the dataset to include in the testing set.

        Returns:
            None
        """

        # Ensure the sizes sum to 1.0
        total_size: float = train_size + validation_size + test_size
        if total_size != 1.0:
            raise ValueError("Train, validation, and test sizes must sum to 1.0")

        # Store the sizes
        self.set_train_size(train_size)
        self.set_validation_size(validation_size)
        self.set_test_size(test_size)

    def setup_data_loaders(self) -> None:
        """
        Set up the data loaders for training, validation, and testing datasets.
        Uses sklearn's train_test_split for stratified splitting to ensure balanced classes.

        Parameters:
            None

        Returns:
            None
        """

        print("\n=============================== Setting Up Data Loaders ===============================\n")


        # Initialize DataLoader temporaries
        training_loader: DataLoader
        validation_loader: DataLoader
        testing_loader: DataLoader

        # Get total dataset size
        dataset_size: int = len(self)
        print(f"Total dataset size: {dataset_size} samples")
        print(f"Split ratios - Train: {self.get_train_size()}, Valid: {self.get_validation_size()}, Test: {self.get_test_size()}\n")

        # Build label mapping for all samples in the dataset
        # This creates a list where labels[i] = class_index for dataset sample i
        all_labels: list[int] = []
        all_indices: list[int] = list(range(dataset_size))


        # The end result of this loop is that all_labels contains the class index for each sample in the dataset
        # e.g. all_labels = [0, 0, 1, 1, 2, 0, 1, ...] where each number corresponds to the class index of the sample at that position
        for class_name in self.get_classes():
            # Get the count and index for the current class
            count: int = self.get_class_counts()[class_name]
            class_index: int = self.get_classes_index()[class_name]

            print(f"Class '{class_name}' (index {class_index}) has {count} samples.")

            # Extend the labels list with the class index repeated 'count' times
            all_labels.extend([class_index] * count)


        # Now we need to split the data into train, validation, and test sets and try to keep the class distribution similar in each set

        # First split: Train vs Temp (Validation + Test)
        # By giving it all the indices and labels, we can stratify the split to maintain class distribution
        # Stratifiying means we try to keep the same proportion of each class in each split as in the overall dataset

        train_indices: list[int] = [0]  # Initialize variables
        temp_indices: list[int] = [0]
        train_labels: list[int] = [0]
        temp_labels: list[int] = [0]

        train_indices, temp_indices, train_labels, temp_labels = train_test_split(
            all_indices,
            all_labels,
            train_size=self.get_train_size(),
            stratify=all_labels,
            random_state=self.get_random_seed()
        )


        # Second split: Validation vs Test
        # We need to calculate the proportion of validation size relative to the temp size

        # We need to calculate the size of temp relative to the whole dataset
        test_val_size: float = self.get_validation_size() + self.get_test_size()

        # valid_size / (valid_size + test_size)
        val_prop: float = self.get_validation_size() / test_val_size

        # Initialize variables
        val_indices: list[int] = [0]
        test_indices: list[int] = [0]
        val_labels: list[int] = [0]
        test_labels: list[int] = [0]

        # This time, we use temp_indices and temp_labels (what's left after the first split) to split into validation and test sets
        # We stratifiy again to maintain class distribution
        val_indices, test_indices, val_labels, test_labels = train_test_split(
            temp_indices,
            temp_labels,
            train_size=val_prop,
            stratify=temp_labels,
            random_state=self.get_random_seed()
        )

        print(f"\nSplit sizes - Train: {len(train_indices)}, Valid: {len(val_indices)}, Test: {len(test_indices)}")
        print(f"Training labels distribution: {np.bincount(train_labels)}")
        print(f"Validation labels distribution: {np.bincount(val_labels)}")
        print(f"Testing labels distribution: {np.bincount(test_labels)}\n")

        # Create Subsets
        train_dataset: Subset = Subset(self, train_indices)
        valid_dataset: Subset = Subset(self, val_indices)
        test_dataset: Subset = Subset(self, test_indices)

        # Compute balanced class weights using sklearn
        unique_classes = np.unique(train_labels)
        sklearn_weights = compute_class_weight(
            class_weight='balanced',
            classes=unique_classes,
            y=train_labels
        )

        # Convert to torch tensor
        class_weights_tensor: torch.Tensor = torch.tensor(sklearn_weights, dtype=torch.float32)
        class_weights_tensor = class_weights_tensor.to(self.get_device())  # Move to the same device as the model

        # Print class weights for user information
        print(f"Class weights (computed from training data):")
        for i, (cls, weight) in enumerate(zip(self.get_classes(), sklearn_weights)):
            print(f"  Class {i} ({cls}, {np.sum(np.array(train_labels) == i)} samples): {weight:.4f}")
        print()

        # Store class weights
        self.set_class_weights(class_weights_tensor)

        # Set loss function with class weights (set_loss updates both __loss and __loss_name)
        self.set_loss(self.get_loss_name(), class_weights_tensor)

        # Fit StandardScaler on training data before creating DataLoaders
        self.fit_scaler_on_training_data(train_indices)

        # Create a generator with the random seed for reproducible shuffling
        g = torch.Generator()
        g.manual_seed(self.get_random_seed())

        # Create DataLoaders
        training_loader: DataLoader = DataLoader(
            train_dataset,
            batch_size=self.get_batch_size(),
            shuffle=True,  # Shuffle training data
            generator=g,  # Use seeded generator for reproducibility
        )

        self.set_training_loader(training_loader)

        validation_loader: DataLoader = DataLoader(
            valid_dataset,
            batch_size=self.get_batch_size(),
            shuffle=False,  # Don't shuffle validation
        )

        self.set_validation_loader(validation_loader)

        testing_loader: DataLoader = DataLoader(
            test_dataset,
            batch_size=self.get_batch_size(),
            shuffle=False,  # Don't shuffle test
        )

        self.set_testing_loader(testing_loader)

        # Validate our DataLoaders and make sure our logic worked correctly
        if not isinstance(self.get_training_loader(), DataLoader) or not isinstance(self.get_validation_loader(), DataLoader) or not isinstance(self.get_testing_loader(), DataLoader) or any(loader is None for loader in [self.get_training_loader(), self.get_validation_loader(), self.get_testing_loader()]):
            raise ValueError("One or more DataLoaders were not set up correctly.")

        # Print summary if it worked
        else:
            print(f"DataLoaders created successfully!")
            print(f"  Training batches: {len(training_loader)}")
            print(f"  Validation batches: {len(validation_loader)}")
            print(f"  Testing batches: {len(testing_loader)}")
            print("\n----------------------------- Data Loaders Setup Complete -----------------------------\n")