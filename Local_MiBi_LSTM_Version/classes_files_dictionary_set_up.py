# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from DeepFake_Detector_Imports import *


class ClassesFilesDictionarySetUp(Dataset):
    """
    A class to set up and manage dataset properties such as device, directory, classes, class counts,
    data length, transform flag, class indices, and file dictionary.
    If the file we're working with is a .wav file, we use librosa to load it properly and create a 2d array (spectrogram).
    Other file types are loaded as raw byte data (works well for 1d data).

    """
    def __init__(self, directory: str = '', file_extension: str = '.wav', DL_type: str = 'RNN', data_type: str = 'mfcc', two_class: bool = False, random_seed: int = 42) -> None:
        """
        Initialize the dataset setup with directory and file extension.
        All other parameters use default values from private attributes.

        Parameters:
            directory (str): The directory path of the dataset. Defaults to an empty string.
            file_extension (str): The file extension to filter files. Defaults to '.wav'.
            DL_type (str): The type of deep learning model to use. Defaults to 'RNN'.
            data_type (str): The type of data to use. Defaults to 'MFCC'.
            two_class (bool): Whether to set up for two-class classification. Defaults to False.
            Will eventually fully support '1DCNN' and '2DCNN' as well.

        Returns:
            None
        """

        super().__init__() # Initialize the parent Dataset class

        # Initialize private attributes with default values
        self.__device: torch.device = torch.device("cpu")  # Default to CPU
        self.__directory: str = 'None' # Default to 'None' indicating no directory set
        self.__two_class: bool = False # Whether to use two-class classification
        self.__classes: List[str] = [] # List of class names
        self.__classes_index: Dict[str, int] = {} # Mapping of class names to indices
        self.__class_counts: Dict[str, int] = {} # Mapping of class names to their respective counts
        self.__file_extension: str = 'None' # File extension to filter files
        self.__data_type: str = 'None'

        self.__random_seed: int = random_seed # Random seed for reproducibility

        # Set global random seeds for reproducibility
        torch.manual_seed(self.__random_seed)
        np.random.seed(self.__random_seed)
        random.seed(self.__random_seed)

        # if using CUDA, set CUDA random seeds as well
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.__random_seed)
            torch.cuda.manual_seed_all(self.__random_seed)  # For multi-GPU

        # Mapping of class indices to lists of file paths
        # Example: {0: ['path/to/class0/file1.wav', 'path/to/class0/file2.wav'], 1: ['path/to/class1/file1.wav']}
        self.__file_dictionary = {}

        self.__gt_subset_count: int = 650 # Number of 'gt' class samples to use
        self.__non_gt_subset_count: int = 500 # Number of non-'gt' class samples to use

        self.__data_length: int = 1048576 # Fixed length for data samples (in bytes) for non-.wav files

        self.__sample_rate: int = 44100 # The sample rate for audio files (basically a measure of quality in Hz)
        self.__duration: int = 8 # Duration of audio clips in seconds
        # self.__n_mels: int = 128 # Number of Mel bands to generate (used for 2DCNN spectrograms)
        self.__n_mfcc: int = 40 # Number of MFCCs to generate (needed if we're dealing with MFCC features)
        self.__n_fft: int = 2048 # Size of the FFT window
        self.__hop_length: int = 512 # Number of samples between successive frames

        self.__DL_type: str = DL_type.strip() # Type of deep learning model to use (RNN, 1DCNN, 2DCNN)

        # Add StandardScaler for feature normalization
        self.__scaler: Optional[StandardScaler | RobustScaler] = StandardScaler()  # Default to StandardScaler
        self.__use_scaler: bool = True  # Flag to enable/disable scaling



        # Choose device: prefer CUDA when available
        # Perform setup actions (these methods set the private attributes internally)
        self.setupDevice()

        # Prefer provided directory when calling set_directory
        if directory is not None and directory != '':
            self.set_directory(directory.strip())
        else:
            self.set_directory()


        # Set gt and non-gt subset counts
        self.set_gt_subset_count(self.__gt_subset_count)
        self.set_non_gt_subset_count(self.__non_gt_subset_count)

        # Use setters to ensure our private attributes are set correctly
        self.set_data_length(self.__data_length)
        self.set_sample_rate(self.__sample_rate)
        self.set_duration(self.__duration)
        self.set_n_mfcc(self.__n_mfcc)
        self.set_n_fft(self.__n_fft)
        self.set_hop_length(self.__hop_length)


        # Set two_class attribute
        self.__two_class = two_class


        # Ensure classes and counts are initialized
        self.set_classes()
        self.set_class_counts()

        # Initialize class index mapping and create classes and counts
        self.create_classes_and_class_counts()

        if two_class == True:
            self.setup_file_dictionary(self.get_directory())
            self.seven_class_to_two()


        # Set the deep learning model type (RNN, 1DCNN, 2DCNN)
        self.set_DL_type(DL_type)


        # Set the file extension
        self.set_file_extension(file_extension)

        # Set the data type
        self.set_data_type(data_type)

    def get_device(self) -> torch.device:
        """
        Get the device used for PyTorch computations.

        Parameters:
            None

        Returns:
            torch.device: The device (CPU or GPU). If not set, defaults to CPU.
        """

        # Ensure device is set and available
        if not self.__device or not torch.cuda.is_available():
            self.__device = torch.device("cpu")

        return self.__device

    def set_device(self, device: torch.device) -> None:
        """
        Set the device for PyTorch computations.

        Parameters:
            device (torch.device): The device to set (CPU or GPU). If None or CUDA not available, defaults to CPU.

        Returns:
            None
        """

        # Ensure device is valid and available
        if device == None or not torch.cuda.is_available():
            self.__device = torch.device("cpu")

        self.__device = device

    def get_directory(self) -> str:
        """
        Get the directory path of the dataset.

        Parameters:
            None

        Returns:
            str: The directory path. If empty, raises an error.
        """

        # Ensure directory is set
        if not self.__directory or not os.path.isdir(self.__directory):
            raise ValueError("Directory is not set or does not exist.")

        return self.__directory

    def set_directory(self, directory: str = '') -> None:
        """
        Set the directory path of the dataset.

        Parameters:
            directory (str): The directory path to set. If empty, raises an error.

        Returns:
            None
        """

        # Ensure directory is valid by checking if it exists and is a directory
        if not directory or not os.path.isdir(directory):
            raise ValueError("Provided directory is not valid or does not exist.")

        self.__directory = directory


    def get_two_class(self) -> bool:
        """
        Get whether the dataset is set up for two-class classification.

        Parameters:
            None

        Returns:
            bool: True if two-class classification is set, False otherwise.

        """

        # Ensure two_class is set
        if not isinstance(self.__two_class, bool):
            raise ValueError("Two-class attribute is not set properly.")

        return self.__two_class

    def set_two_class(self, two_class: bool = False) -> None:
        """
        Set whether the dataset is set up for two-class classification.

        Parameters:
            two_class (bool): True to set up for two-class classification, False otherwise.

        Returns:
            None
        """

        # Ensure two_class is a boolean value
        if not isinstance(two_class, bool):
            raise ValueError("Two-class attribute must be a boolean value.")

        self.__two_class = two_class



    def get_classes(self) -> List[str]:
        """
        Get the list of classes in the dataset. If empty, raises an error.
        e.g.: ['class0', 'class1', 'class2']

        Parameters:
            None

        Returns:
            List[str]: The list of class names.
        """

        # Ensure classes at least has a default value
        if not self.__classes:
           raise ValueError("Classes list is empty.")

        return self.__classes

    def set_classes(self, classes: List[str] = []) -> None:
        """
        Set the list of classes in the dataset by scanning the directory structure.
        e.g.: ['class0', 'class1', 'class2']

        Parameters:
            classes (List[str], optional): The list of class names to set. If empty, raises an error.

        Returns:
            None
        """

        # Ensure classes is a list of strings
        if classes is not None:
            self.__classes = classes
        elif not self.__classes:
            raise ValueError("Classes list is empty.")

    def get_class_counts(self) -> Dict[str, int]:
        """
        Get the counts of samples per class in the dataset.
        eg.: {'class0': 100, 'class1': 150, 'class2': 200}

        Parameters:
            None

        Returns:
            Dict[str, int]: A dictionary with class names as keys and their respective counts as values.
            If empty, raises an error.
        """

        # Ensure class counts at least has a default value
        if not self.__class_counts:
            raise ValueError("Class counts dictionary is empty.")

        return self.__class_counts

    def set_class_counts(self, class_counts: Dict[str, int] = {}) -> None:
        """
        Set the counts of samples per class in the dataset by scanning the directory structure.
        eg.: {'class0': 100, 'class1': 150, 'class2': 200}

        Parameters:
            class_counts (Dict[str, int], optional): A dictionary with class names as keys and their respective counts as values.
            If empty, raises an error.

        Returns:
            None
        """

        # Ensure class counts is a dictionary with a tuple of (string, integer)
        if class_counts is not None:
            self.__class_counts = class_counts
        elif not self.__class_counts:
            raise ValueError("Class counts dictionary is empty.")


    def get_gt_subset_count(self) -> int:
        """
        Get the number of 'gt' class samples to use.
        e.g.: 700 samples

        Parameters:
            None

        Returns:
            int: The number of 'gt' class samples to use.
        """

        return self.__gt_subset_count

    def set_gt_subset_count(self, gt_subset_count: int) -> None:
        """
        Set the number of 'gt' class samples to use.
        e.g.: 700 samples

        Parameters:
            gt_subset_count (int): The number of 'gt' class samples to set.

        Returns:
            None
        """

        # If gt_subset_count is positive, set it; otherwise, raise an error
        if gt_subset_count > 0:
            self.__gt_subset_count = gt_subset_count
        else:
            raise ValueError("GT subset count must be a positive integer.")


    def get_non_gt_subset_count(self) -> int:
        """
        Get the number of non-'gt' class samples to use.
        e.g.: 475 samples

        Parameters:
            None

        Returns:
            int: The number of non-'gt' class samples to use.
        """

        return self.__non_gt_subset_count


    def set_non_gt_subset_count(self, non_gt_subset_count: int) -> None:
        """
        Set the number of non-'gt' class samples to use.
        e.g.: 475 samples

        Parameters:
            non_gt_subset_count (int): The number of non-'gt' class samples to set.

        Returns:
            None
        """

        # If non_gt_subset_count is positive, set it; otherwise, raise an error
        if non_gt_subset_count > 0:
            self.__non_gt_subset_count = non_gt_subset_count
        else:
            raise ValueError("Non-GT subset count must be a positive integer.")


    def get_random_seed(self) -> int:
        """
        Get the random seed for reproducibility.
        e.g.: 42

        Parameters:
            None

        Returns:
            int: The random seed.
        """

        # Ensure random seed is set properly and is an integer
        if not isinstance(self.__random_seed, int):
            raise ValueError("Random seed is not set properly.")
        else:
            return self.__random_seed


    def set_random_seed(self, random_seed: int) -> None:
        """
        Set the random seed for reproducibility.
        e.g.: 42

        Parameters:
            random_seed (int): The random seed to set.

        Returns:
            None
        """

        # Ensure random seed is an integer
        if isinstance(random_seed, int):
            self.__random_seed = random_seed
        else:
            raise ValueError("Random seed must be an integer.")


    def get_data_length(self) -> int:
        """
        Get the fixed length for data samples.
        Only needed if doing raw data.
        e.g.: 1048576 (in bytes)

        Parameters:
            None
        Returns:
            int: The fixed length for data samples.
        """

        return self.__data_length

    def set_data_length(self, data_length: int) -> None:
        """
        Set the fixed length for data samples.
        Only needed if doing raw data.
        e.g.: 1048576 (in bytes)

        Parameters:
            data_length (int): The fixed length to set for data samples.

        Returns:
            None
        """

        # If data_length is positive, set it; otherwise, raise an error
        if data_length > 0:
            self.__data_length = data_length
        else:
            raise ValueError("Data length must be a positive integer.")

    def get_classes_index(self) -> Dict[str, int]:
        """
        Get the mapping of class names to their respective indices.
        e.g.: {'class0': 0, 'class1': 1, 'class2': 2}

        Parameters:
            None

        Returns:
            Dict[str, int]: A dictionary mapping class names to their respective indices.
            If empty, raises an error.
        """

        # Ensure classes index at least is a dictionary with a tuple of (string, integer)
        if not self.__classes_index:
            raise ValueError("Classes index dictionary is empty.")

        return self.__classes_index

    def set_classes_index(self, classes_index: Dict[str, int] = {}) -> None:
        """
        Set the mapping of class names to their respective indices.
        e.g.: {'class0': 0, 'class1': 1, 'class2': 2}

        Parameters:
            classes_index (Dict[str, int], optional): A dictionary mapping class names to their respective indices.
            If empty, raises an error.

        Returns:
            None
        """

        # Ensure classes index is at least a dictionary with a tuple of (string, integer)
        if classes_index is not None:
            self.__classes_index = classes_index
        elif not self.__classes_index:
            raise ValueError("Classes index dictionary is empty.")


    def get_file_dictionary(self) -> Dict[int, List[str]]:
        """
        Get the dictionary of file paths along with their corresponding class indices.
        e.g.: {0: ['file1.wav', 'file2.wav'], 1: ['file3.wav', 'file4.wav']}

        Parameters:
            None

        Returns:
            Dict[int, List[str]]: A dictionary mapping class indices to lists of file paths.
            If empty, raises an error.
        """

        # Return the file dictionary mapping class_index -> list[str].
        # If empty, raise an error.
        if not self.__file_dictionary:
            raise ValueError("File dictionary is empty.")
        else:
            return self.__file_dictionary

    def set_file_dictionary(self, file_dictionary: Dict[int, List[str]]) -> None:
        """
        Set the dictionary of file paths along with their corresponding class indices.
        e.g.: {0: ['file1.wav', 'file2.wav'], 1: ['file3.wav', 'file4.wav']}

        Parameters:
            file_dictionary (Dict[int, List[str]]): A dictionary mapping class indices to lists of file paths.

        Returns:
            None
        """

        # Accept a dictionary mapping class_index -> list[str]
        if file_dictionary is not None and isinstance(file_dictionary, dict):
            self.__file_dictionary = file_dictionary
        else:
            raise ValueError("File dictionary cannot be None.")

    def get_sample_rate(self) -> int:
        """
        Get the sample rate for audio files.
        e.g.: 44100 Hz

        Parameters:
            None

        Returns:
            int: The sample rate in Hz.
        """
        return self.__sample_rate

    def set_sample_rate(self, sample_rate: int) -> None:
        """
        Set the sample rate for audio files.
        e.g.: 44100 Hz

        Parameters:
            sample_rate (int): The sample rate in Hz.

        Returns:
            None
        """

        # If sample_rate is positive and reasonable, set it.
        if sample_rate > 10000 and isinstance(sample_rate, int):
            self.__sample_rate = sample_rate

        # Otherwise, raise an error
        else:
            raise ValueError("Sample rate must be a positive integer and of a reasonable value above 10000 Hz.")

    def get_duration(self) -> int:
        """
        Get the duration of audio clips.
        e.g.: 8 seconds

        Parameters:
            None

        Returns:
            int: The duration in seconds.
        """
        return self.__duration

    def set_duration(self, duration: int) -> None:
        """
        Set the duration of audio clips.
        e.g.: 8 seconds

        Parameters:
            duration (int): The duration in seconds.

        Returns:
            None
        """
        if duration > 0 or isinstance(duration, int):
            self.__duration = duration
        else:
            raise ValueError("Duration must be a positive integer and not a float.")

    def get_n_mfcc(self) -> int:
        """
        Get the number of MFCCs to generate.
        e.g.: 40 MFCCs

        Parameters:
            None

        Returns:
            int: The number of MFCCs.
        """
        return self.__n_mfcc

    def set_n_mfcc(self, n_mfcc: int) -> None:
        """
        Set the number of MFCCs to generate.
        e.g.: 40 MFCCs

        Parameters:
            n_mfcc (int): The number of MFCCs.

        Returns:
            None
        """

        # Ensure n_mfcc is a positive integer between 1 and 512
        if 512 >= n_mfcc > 0 and isinstance(n_mfcc, int):
            self.__n_mfcc = n_mfcc
        else:
            raise ValueError("Number of MFCCs must be a positive integer between 1 and 512 and not a float.")

    def get_n_mels(self) -> int:
        """
        Get the number of Mel bands to generate.
        e.g.: 128 Mel bands

        Parameters:
            None

        Returns:
            int: The number of Mel bands.
        """

        return self.__n_mels

    def set_n_mels(self, n_mels: int) -> None:
      """
      Set the number of Mel bands to generate.
      e.g.: 128 Mel bands

      Parameters:
          n_mels (int): The number of Mel bands.

      Returns:
          None
      """

      # Ensure n_mels is a positive number between 1 and 1024
      if 1024 >= n_mels > 0 and isinstance(n_mels, int):
          self.__n_mels = n_mels
      else:
          raise ValueError("Number of Mel bands must be a positive integer between 1 and 1024 and not a float.")

    def get_n_fft(self) -> int:
        """
        Get the size of the FFT window.
        The FFT (Fast Fourier Transform) window size determines the number of samples/points
        in the window.
        e.g.: 2048 samples/points

        Parameters:
            None

        Returns:
            int: The size of the FFT window.
        """
        return self.__n_fft

    def set_n_fft(self, n_fft: int) -> None:
        """
        Set the size of the FFT window.
        The FFT (Fast Fourier Transform) window size determines the number of samples/points
        in the window.
        e.g.: 2048 samples/points

        Parameters:
            n_fft (int): The size of the FFT window.

        Returns:
            None
        """

        # Ensure n_fft is a positive integer
        if n_fft > 0 and isinstance(n_fft, int):
            self.__n_fft = n_fft
        else:
            raise ValueError("Size of FFT window must be a positive integer and not a float.")

    def get_hop_length(self) -> int:
        """
        Get the number of samples between successive frames.
        The hop length determines how much the window shifts between successive frames.
        By window, we mean the segment of audio data being analyzed at a time.
        e.g.: 512 samples

        Parameters:
            None

        Returns:
            int: The hop length.
        """
        return self.__hop_length

    def set_hop_length(self, hop_length: int) -> None:
        """
        Set the number of samples between successive frames.
        The hop length determines how much the window shifts between successive frames.
        By window, we mean the segment of audio data being analyzed at a time.
        e.g.: 512 samples

        Parameters:
            hop_length (int): The hop length.

        Returns:
            None
        """

        # Ensure hop_length is a positive integer
        if hop_length > 0 and isinstance(hop_length, int):
            self.__hop_length = hop_length
        else:
            raise ValueError("Hop length must be a positive integer and not a float.")

    def get_DL_type(self) -> str:
        """
        Get the type of deep learning model to use.
        e.g.: 'RNN', '1DCNN', '2DCNN'

        Parameters:
            None

        Returns:
            str: The deep learning model type.
        """

        return self.__DL_type

    def set_DL_type(self, DL_type: str) -> None:
        """
        Set the type of deep learning model to use.
        e.g.: 'RNN', '1DCNN', '2DCNN'

        Parameters:
            DL_type (str): The deep learning model type.

        Returns:
            None
        """

        # Make sure DL_type is stripped of whitespace and uppered
        DL_type = DL_type.strip().upper()


        # Ensure DL_type is one of the supported types
        if DL_type in ['RNN', '1DCNN', '2DCNN']:
            self.__DL_type = DL_type
        else:
            raise ValueError("Deep learning model type must be one of: 'RNN', '1DCNN', '2DCNN'.")

    def get_file_extension(self) -> str:
        """
        Get the file extension of the audio files.
        e.g.: '.wav'

        Parameters:
            None

        Returns:
            str: The file extension.
        """

        return self.__file_extension

    def set_file_extension(self, file_extension: str) -> None:
      """
      Set the file extension of the audio files.
      e.g.: '.wav'

      Parameters:
          file_extension (str): The file extension.

      Returns:
          None
      """

      # Ensure file_extension is a proper string
      if isinstance(file_extension, str):
          self.__file_extension = file_extension
      else:
          raise ValueError("File extension must be a string.")

    def get_data_type(self) -> str:
      """
      Get the data type of the audio files.
      e.g.: 'mfcc', 'Mel-Spectrogram', etc...

      Parameters:
          None

      Returns:
          str: The data type.
      """

      return self.__data_type

    def set_data_type(self, data_type: str) -> None:
      """
      Set the data type of the audio files.
      e.g.: 'mfcc', 'mel-spectrogram', etc...

      Parameters:
          data_type (str): The data type.

      Returns:
          None
      """

      data_type = data_type.strip().lower()

      try:

        # Check if data type is in list of allowed data_types
        if data_type in ['mfcc', 'mel-Spectrogram', 'bytes', 'waveform']:

            # Check if model correlates correctly with data_type
            if data_type in ['mfcc', 'mel-Spectrogram', 'waveform'] and self.get_DL_type() in ['RNN', '2DCNN']:
                self.__data_type = data_type

            elif data_type in ['bytes'] and self.get_DL_type() in ['1DCNN']:
                self.__data_type = data_type

            else:
                raise ValueError

        else:
            raise ValueError

      except:
           raise ValueError('Data type must be one of: \'mfcc\', \'mel-Spectrogram\', \'bytes\', \'waveform\'.\n If using mfcc, mel-spectrogram, or waveform, you must use RNN or 2DCNN DL_type.\n Otherwise, you can use bytes only with 1DCNN')



    def get_scaler(self) -> StandardScaler | RobustScaler:
        """
        Get the Scaler used for feature normalization.
        The Scaler is used to normalize the features in the dataset.
        e.g.: StandardScaler() or RobustScaler()
        StandardScaler is used when the dataset does not contain or contains few outliers.
        RobustScaler can be used when the dataset contains outliers.

        Parameters:
            None

        Returns:
            Optional[StandardScaler | RobustScaler]: The fitted StandardScaler or RobustScaler.
        """

        if self.__scaler is None:
            raise ValueError("Scaler has not been set.")

        return self.__scaler

    def set_scaler(self, scaler: StandardScaler | RobustScaler) -> None:
        """
        Set the Scaler used for feature normalization.
        The Scaler is used to normalize the features in the dataset.
        e.g.: StandardScaler() or RobustScaler()
        StandardScaler is used when the dataset does not contain or contains few outliers.
        RobustScaler can be used when the dataset contains outliers.


        Parameters:
            scaler (StandardScaler | RobustScaler): The fitted StandardScaler or RobustScaler to set.

        Returns:
            None
        """

        # Ensure scalar is either StandardScaler or RobustScaler
        if not isinstance(scaler, (StandardScaler, RobustScaler)):
            raise ValueError("Scaler must be an instance of StandardScaler or RobustScaler.")

        self.__scaler = scaler

    def get_use_scaler(self) -> bool:
        """
        Get whether to use Scaler for feature normalization.
        e.g.: True or False

        Parameters:
            None

        Returns:
            bool: Whether to use scaling.
        """

        return self.__use_scaler

    def set_use_scaler(self, use_scaler: bool) -> None:
        """
        Set whether to use Scaler for feature normalization.
        e.g.: True or False

        Parameters:
            use_scaler (bool): Whether to use scaling.

        Returns:
            None
        """

        # Ensure use_scaler is a boolean
        if not isinstance(use_scaler, bool):
            raise ValueError("use_scaler must be a boolean value (True or False).")

        self.__use_scaler = use_scaler


    def setupDevice(self) -> None:
        """
        Setup device for PyTorch computations by checking for CUDA availability.
        If CUDA is available, it sets the device to GPU; otherwise, it defaults to CPU.

        Parameters:
            None

        Returns:
            None
        """

        print(" =============================== Device Setup =============================== \n")

        self.set_device(torch.device("cuda")) # Try to set to CUDA
        print(f"Using device: {self.get_device()}") # Print the device being used

        # Check if CUDA is available and print relevant information (get_device will automatically default to CPU if not)
        if self.get_device().type == "cuda":

            try:
                # Print basic GPU info
                print("GPU:", torch.cuda.get_device_name(0))
                print("CUDA available:", torch.cuda.is_available())
            except Exception:
                print("Error accessing CUDA device information.")

        # Otherwise, default to CPU
        else:
            print("CUDA not available, using CPU.")
        print ("\n ----------------------------- Setup Complete ------------------------------ \n")


    def create_classes_and_class_counts(self) -> None:
        """
        This function sets up the logic needed to not only find the classes in the dataset directory
        and find their counts, but also to create a mapping from class names to indices.

        Parameters:
            None

        Returns:
            None
        """

        print(" \n=============================== Creating Classes, Class Counts, and Class Indices =============================== \n")

        # Go through the dataset directory, finding the name of each file within the directory, and ensure it is a directory
        # If it is, add it to the classes list
        classes: List[str] = sorted([d for d in os.listdir(self.get_directory()) if os.path.isdir(os.path.join(self.get_directory(), d))])

        # Set the classes found and print them
        self.set_classes(classes)
        print (" Classes found: ", list(self.get_classes()))

        # Now, create the class counts and class indices
        class_counts: Dict[str, int] = {}
        classes_index: Dict[str, int] = {}

        # For each class found, count the number of files in its directory and assign an index
        for index, class_name in enumerate(self.get_classes()):

            # Get the directory path for the current class by joining the base directory with the class name
            class_dir: str = os.path.join(self.get_directory(), class_name)

            # The class count is simply the number of files in that directory,
            # which can be found by getting a length of the list of files in a directory
            class_counts[class_name] = len(os.listdir(class_dir))

            # The index is simply the enumeration order (0, 1, 2, ...)
            # Basically we're saying: key: class_name -> value: index
            classes_index[class_name] = index

        # Store the class counts and indices and print them
        self.set_class_counts(class_counts)
        self.set_classes_index(classes_index)
        print (" Class counts: ", self.get_class_counts())
        print (" Class indices: ", self.get_classes_index())

        print (" \n----------------------------- Creation of Classes, Class Counts, and Class Indices Complete ------------------------------ \n")


    def setup_file_dictionary(self, file_type: str = 'None') -> None:
        """
        This function sets up the file lists by scanning the dataset directory
        and pairing each file path with its corresponding class index.
        Basically, it creates a dictionary mapping (int) class index -> ([str]) list of file paths for that class.

        Parameters:
            file_type (str): The file extension to filter files (e.g., '.wav').

        Returns:
            None
        """

        print(" \n=============================== Setting up File Dictionary =============================== \n")

        if file_type == 'None':
            file_type = self.get_file_extension()

        # Placeholder to build the file dictionary
        file_dict: Dict[int, List[str]] = {}

        # Placeholder for class directory path
        class_dir: str = ''

        # Placeholder to store filenames temporarily
        files: List[str] = []

        # Placeholder to store full file paths
        full_paths: List[str] = []

        # Placeholder for class index
        class_index: int = 0

        # Build a mapping from class index -> list of full file paths for that class
        for class_name in self.get_classes():

            # Get the directory path for the current class by joining the base directory with the class name
            class_dir: str = os.path.join(self.get_directory(), class_name)

            # Get all files with the correct extension first
            all_files = sorted([f for f in os.listdir(class_dir) if f.endswith(file_type)])

            # Create a seeded random generator for reproducibility
            rng = random.Random(self.get_random_seed())

            if class_name != 'gt':
                # Randomly sample up to 475 (or the specified non-GT subset count) files from this class
                max_samples = min(self.get_non_gt_subset_count(), len(all_files))
                sampled_files = rng.sample(all_files, max_samples)
            else:
                # Randomly sample up to 700 (or the specified GT subset count) files from this class
                max_samples = min(self.get_gt_subset_count(), len(all_files))
                sampled_files = rng.sample(all_files, max_samples)

            files = sampled_files
            
            # Build full file paths
            full_paths = [os.path.join(class_dir, f) for f in files]

            # Get the class index from the class name using the previously created mapping
            class_index_temp: int | None = self.get_classes_index()[class_name]

            # Safety check in case class name not found in index mapping
            if class_index_temp is None:
                raise ValueError(f"Class index for class '{class_name}' not found.")
            else:
                class_index = class_index_temp

            # Store the list of full paths under the class index
            file_dict[class_index] = full_paths

            # Print files for this class immediately with only basenames for readability
            basenames: List[str] = [os.path.basename(p) for p in full_paths]
            print(f"\n Class {class_name}: Files: {basenames}\n\n")

        # Store the dictionary
        self.set_file_dictionary(file_dict)

        # Update class counts to reflect actual sampled files
        updated_class_counts: Dict[str, int] = {}

        for class_name in self.get_classes():
            class_index = self.get_classes_index()[class_name]
            updated_class_counts[class_name] = len(file_dict[class_index])
            print(f"Class {class_name} has {updated_class_counts[class_name]} files.")

        self.set_class_counts(updated_class_counts)

        # Print total number of files found
        total_files: int = sum(len(v) for v in file_dict.values())
        print(f" Total files found: {total_files} ")
        print(f" Updated class counts: {self.get_class_counts()} ")
        print (" \n----------------------------- File Dictionary Setup Complete ------------------------------ \n")


    def seven_class_to_two(self) -> None:
        """
        This class coverts the six classes plus 'gt' into a two-class system: 'gt' and 'non-gt'.


        Parameters:
            None

        Returns:
            None
        """

        print(" \n=============================== Converting Seven-Class to Two-Class System =============================== \n")

        # Create a new file dictionary for two-class system
        new_file_dict: Dict[int, List[str]] = {0: [], 1: []}

        new_file_index: Dict[str, int] = {'non-gt': 0, 'gt': 1}


        # Update classes to only 'gt' and 'non-gt'
        for index, class_name in enumerate(self.get_classes()):
            if class_name != 'gt':

                new_file_dict[new_file_index['non-gt']].extend(self.get_file_dictionary()[self.get_classes_index()[class_name]])
            else:
                new_file_dict[new_file_index['gt']].extend(self.get_file_dictionary()[self.get_classes_index()[class_name]])

        # Update the file dictionary
        self.set_file_dictionary(new_file_dict)

        # Update classes and class indices
        self.set_classes(['non-gt', 'gt'])

        # Update class counts
        new_class_counts: Dict[str, int] = {'non-gt': len(new_file_dict[new_file_index['non-gt']]),
                                    'gt': len(new_file_dict[new_file_index['gt']])}

        self.set_class_counts(new_class_counts)

        # Update class indices
        self.set_classes_index(new_file_index)

        print(" Updated classes: ", self.get_classes())
        print(" Updated class counts: ", self.get_class_counts())
        print(" Updated class indices: ", self.get_classes_index())

        print(" \n----------------------------- Conversion to Two-Class System Complete ------------------------------ \n")



    def setup_data(self, class_index: int, file_index_within_class: int) -> Tuple[torch.Tensor | np.ndarray, int]:
        """
        This function will get a specific file in a class based on the provided class index and file index within that class,
        process data from it, and return the data along with its class index.

        Parameters:
            class_index (int): The index of the class.
            file_index_within_class (int): The index of the file within the specified class.

        Returns:
            Tuple[torch.Tensor | bytes | np.ndarray, int]: A tuple containing the processed data and its corresponding class index.

            The exact format depends on DL_type:

            - RNN mode (DL_type='RNN' and file_type = '.wav'):
              Returns (torch.Tensor, int) where:
                - Tensor shape: (time_steps, n_mfcc)
                - time_steps = (sample_rate * duration) // hop_length + 1
                - Features are MFCC coefficients extracted via librosa
                - n_mfcc is the number of MFCCs, where MFCCs are Mel-Frequency Cepstral Coefficients
                - Mel-Frequency Cepstral Coefficients (MFCCs) are a compact representation of the spectral
                  envelope of audio signals. They capture high-frequency characteristics while being robust
                  to pitch variations, making them effective features for distinguishing audio patterns.
                - Audio is loaded with fixed duration and padded/truncated to exact length

            TODO:
            - 2DCNN mode (DL_type='2DCNN' and file_type = '.wav'):
              Returns (torch.Tensor, int) where:
                - Tensor shape: (1, n_mels, time_steps)
                - First dimension is channel (1 for grayscale spectrogram)
                - n_mels is the number of mel frequency bins
                - time_steps = (sample_rate * duration) // hop_length + 1
                - Data is mel-spectrogram in dB scale
                - Audio is loaded with fixed duration and padded/truncated to exact length

            - 1DCNN mode (DL_type='1DCNN' and file_type != '.wav'):
              Returns (np.ndarray, int) where:
                - Array contains raw byte data from non-.wav files
                - Length is data_length bytes, zero-padded if file is shorter
        """


        signal: np.ndarray # The audio signal
        sr: int | float # The sample rate

        # Retrieve the dictionary of files
        file_dict: Dict[int, List[str]] = self.get_file_dictionary()

        # Validate class index
        if class_index not in file_dict:
            raise IndexError(f"Class index {class_index} not found in file dictionary.")

        # Validate file index within the class
        class_files: List[str] = file_dict[class_index]
        if file_index_within_class < 0 or file_index_within_class >= len(class_files):
            raise IndexError(f"File index {file_index_within_class} out of range for class index {class_index}.")

        # Get the file path
        file_path: str = class_files[file_index_within_class]

        # If the file is a .wav file, use librosa to load it properly
        if os.path.splitext(file_path)[1] == '.wav' and self.get_DL_type() in ['RNN'] and self.get_data_type() in ['mfcc']:

            # Use librosa to load the audio file
            try:
                signal: np.ndarray
                sr: int | float

                signal, sr = librosa.load(file_path, sr=self.get_sample_rate(), duration=self.get_duration())

            # Catch errors during audio loading
            except Exception as e:

                print(f"\nWarning: Failed to load audio file: {file_path}")
                print(f"Error: {e}")
                print("Returning zeros as fallback.\n")

                # Return zeros with expected shape as fallback
                signal: np.ndarray = np.zeros(self.get_sample_rate() * self.get_duration())

                # Get the sample rate for consistency
                sr: int | float = self.get_sample_rate()

            # Ensure signal has fixed length
            target_length: int = self.get_sample_rate() * self.get_duration()

            # Pad the signal to the target length
            if len(signal) < target_length:
                signal: np.ndarray = np.pad(signal, (0, target_length - len(signal)), mode='constant')

            # Truncate the signal if it's longer than target length
            else:
                signal: np.ndarray = signal[:target_length]

            # Extract MFCC features (output shape: (n_mfcc/features, time_steps))
            mfccs: np.ndarray = librosa.feature.mfcc(
                y=signal,
                sr=sr,
                n_mfcc=self.get_n_mfcc(),
                n_fft=self.get_n_fft(),
                hop_length=self.get_hop_length())

            # Transpose to (time_steps, features) for RNN (RNNs expect time dimension first)
            mfccs = mfccs.T

            # Convert to tensor
            data: torch.Tensor | bytes | np.ndarray = torch.tensor(mfccs, dtype=torch.float32)

            # Apply scaling if scaler is fitted and enabled
            if self.get_use_scaler() is True and self.get_scaler() is not None:
                # Data is already in (time_steps, n_mfcc) format for RNN
                data_np = data.numpy()
                scaler = self.get_scaler()

                if scaler is not None:  # Additional safety check for type checker
                    data_scaled = scaler.transform(data_np)
                    data = torch.tensor(data_scaled, dtype=torch.float32)

        # TODO: If the file is a .wav file and we're using 2DCNN, process accordingly
        elif os.path.splitext(file_path)[1] == '.wav' and self.get_DL_type() in ['2DCNN']:
            data = torch.tensor([])  # Placeholder for 2DCNN processing

        # If the file is not a .wav file and we're using 1DCNN, read raw bytes
        elif os.path.splitext(file_path)[1] != '.wav' and self.get_DL_type() in ['1DCNN']:

            # Load the data from the file
            with open(file_path, 'rb') as f:
                # Read the specified number of bytes
                data = f.read(self.get_data_length())
                data = np.pad(data, (0, self.get_data_length() - len(data)), 'constant')  # Pad with zeros if needed
        else:
            raise ValueError(f"Unsupported file type or DL_type for file: {file_path}")


        return data, class_index


    def __getitem__(self, index: int) -> Tuple[Any, int]:
        """
        Uses setup_data function to retrieve a sample based on a flat index.
        Cumulative count works as a running tally! If we have 3 classes with 5, 10, and 15 samples respectively,
        and we look for index 12, we see that:
        - Class 0 (5 samples): cumulative count is 5, index 12 is greater than 5, move to next class
        - Class 1 (10 samples): cumulative count is 15, index 12 is less than 15, so it belongs to class 1
        - File index within class 1 is 12 - 5 = 7
        - Call setup_data with class_index=1 and file_index_within_class=7

        We know we'll get a valid flat index because PyTorch's DataLoader, train_test_split (stratified splitting),
        and our fit_scaler_on_training_data all use __len__(self) to determine valid index ranges (0 to len(self)-1),
        and len(self) returns the sum of all class counts, ensuring indices are within bounds.
        Therefore, we'll NEVER get the indices based on one class only - we always consider the entire dataset!

        For example, train_test_split gives us stratified indices like [0, 2048, 7, 591, 20, ...] which span across classes
        while maintaining proportional class distribution in train/validation/test splits.

        Parameters:
            index (int): The flat index of the sample.

        Returns:
            Tuple[Any, int]: A tuple containing the data and its corresponding class index.

        """

        # Placeholder for cumulative count of samples
        cumulative_count = 0

        # Iterate through classes to find the correct class for the given index
        for class_name in self.get_classes():

            # Get the count of samples in this class
            count = self.get_class_counts()[class_name]

            # Check if the index falls within this class's range (cumulative_count maintains the running total of samples)
            if index < cumulative_count + count:

                # Found the correct class
                class_index = self.get_classes_index()[class_name]
                file_index_within_class = index - cumulative_count # Calculate the file index within the class by subtracting cumulative count

                return self.setup_data(class_index, file_index_within_class) # Retrieve the item using setup_data

            # Update cumulative count for the next iteration
            cumulative_count += count

        # If we reach here, index is out of bounds
        raise IndexError(f"Index {index} out of range for dataset of size {cumulative_count}")

    def __len__(self) -> int:
        """
        Get the total number of samples in the dataset.
        If we have 3 classes with counts 100, 150, and 200, the total length is 450.
        (Note: If we choose a subsample of files per class during setup, like we did in setup_file_dictionary(), this should reflect that total subsample count.)

        Parameters:
            None

        Returns:
            int: The total number of samples across all classes.
        """

        total_samples: int = 0

        # Sum the counts of all classes to get the total number of samples
        for count in self.get_class_counts().values():
            total_samples += count

        return total_samples

    def fit_scaler_on_training_data(self, train_indices) -> None:
        """
        Fit the StandardScaler on training data features using optimized sampling.
        Uses time-step reduction for fast fitting.

        Parameters:
            train_indices: Indices of training samples to fit scaler on.

        Returns:
            None
        """

        # If scaling is not enabled, skip fitting
        if not self.get_use_scaler():
            return


        print( "\n=============================== Fitting StandardScaler on Training Data ===============================\n")

        # Limit to a maximum of 100 samples for fitting
        max_samples = min(100, len(train_indices))

        # Calculate step size for sampling
        sample_step = max(1, len(train_indices) // max_samples)

        # Sample indices with step size (max samples serves as an upper limit)
        sample_indices = train_indices[::sample_step][:max_samples]

        print(f"Using {len(sample_indices)} samples (out of {len(train_indices)}) for scaler fitting")

        training_features = []

        # Disable scaling temporarily during feature extraction
        # (we want raw features for fitting and if we don't disable, the scaler will try to scale data that hasn't been fitted yet...)
        self.set_use_scaler(False)

        # For each sampled index, extract features and subsample time steps
        for idx in sample_indices:

            try:
                # Get raw data (scaling disabled)
                # For RNN: data comes as (time_steps, features) - already in StandardScaler format
                data, _ = self.__getitem__(idx)

                # Convert to numpy array if it's a tensor (if we're dealing with RNN or 2DCNN, it should be)
                if isinstance(data, torch.Tensor):
                    data = data.numpy()

                # Handle different data shapes
                if self.get_DL_type() == 'RNN':
                    # data shape: (time_steps, features)
                    # Subsample every 7th time step to reduce data volume
                    training_features.append(data[::7])

                elif self.get_DL_type() == '2DCNN':
                    continue  # 2DCNN - spectrograms, skip scaling for now

                else:  # 1DCNN - raw bytes, no scaling needed
                    continue

            # Catch any exceptions during feature extraction
            except Exception as e:
                print(f"Warning: Could not process sample {idx} for scaler fitting: {e}")
                continue

        # Re-enable scaling after feature extraction
        self.set_use_scaler(True)

        # Fit StandardScaler if we have valid features
        if training_features:

            # Concatenate all features into a single array
            # np.concatenate stacks arrays along the first axis (rows)
            all_features: np.ndarray = np.concatenate(training_features, axis=0)

            print(f"Fitting scaler on {all_features.shape[0]} feature vectors (shape: {all_features.shape})")

            # Create a fresh Scaler instance and fit on all collected features at once
            if isinstance(self.get_scaler(), RobustScaler):
                scaler: StandardScaler | RobustScaler = RobustScaler() # Use RobustScaler if previously set

            elif isinstance(self.get_scaler(), StandardScaler):
                scaler: StandardScaler | RobustScaler = StandardScaler() # Use StandardScaler if previously set

            else:
                raise ValueError("Scaler must be set to either StandardScaler or RobustScaler before fitting.")

            scaler.fit(all_features)  # Fit on all collected feature vectors
            self.set_scaler(scaler)  # Store the newly fitted scaler

            print(f"StandardScaler fitted on {len(training_features)} training samples")

        # Error out if no valid features found
        else:
            print("Warning: No valid training features found for scaler fitting (disregard warning if using 1DCNN with raw byte data).")
            self.set_use_scaler(False)

        print ( "\n----------------------------- StandardScaler Fitting Complete ------------------------------\n")