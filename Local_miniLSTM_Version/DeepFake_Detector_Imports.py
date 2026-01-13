# Import standard libraries
import time
import os
import os.path
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union

# Import pickle for object serialization
import pickle

# Import standard data science libraries
import random
import numpy as np
import pandas as pd

# Import PyTorch and related libraries
import torch
import torchmetrics, mlxtend
from torchmetrics import ConfusionMatrix
from torch.utils.data import DataLoader, Dataset, Subset, random_split, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from mlxtend.plotting import plot_confusion_matrix

# Import tqdm for progress bar
from tqdm.auto import tqdm


# Import scikit-learn for various utilities
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.utils.class_weight import compute_class_weight
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt # Import matplotlib for plotting
import seaborn as sns # Import seaborn for enhanced visualizations

# Import XGBoost and other classifiers
from sklearn.preprocessing import RobustScaler, StandardScaler


# Import audio processing libraries
import scipy
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import scipy.io.wavfile as wavfile
import scipy.signal
import librosa
import librosa.display

# Import collections for Counter
from collections import Counter

# Import CUDA extension for parallel scan operations
try:
    import prescan_2048_cuda # type: ignore[import]
    CUDA_AVAILABLE = True
except ImportError:
    prescan_2048_cuda = None
    CUDA_AVAILABLE = False

# Import additional utilities
import shutil

from torch.utils.flop_counter import FlopCounterMode

from pathlib import Path
