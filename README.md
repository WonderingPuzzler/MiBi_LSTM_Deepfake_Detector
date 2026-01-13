
# **MiBi-LSTM Audio Deepfake Detector**

An updated version of my previous LSTM model! This one is a lot more custom than the older version, so I thought I'd give it its own repository for now! I plan to use it for a lot more than just this dataset or just use it by
itself, however, so this repository is subject to major changes in the next few months! In any case, MiBi-LSTM has a custom forward and backward torch autograd class, a custom parallel scan in C CUDA with a C++ wrapper for speed, and
a custom model creation class entirely unreliant on Pytorch's nn.LSTM() module! Finetuning and tweaks have lead this model to acheive much better performance than the previous LSTM model, along with a much tinier model state dictionary and 
way fewer parameters!

The LibreSeVoc dataset and inspiration for this project came from this project from Chengzhe Sun, Shan Jia, Shuwei Hou, and Siwei Lyu: https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts.
For the model, MiBi-LSTM is *heavily* based off of minLSTM by Leo Feng, Frederick Tung, Mohamed Osama Ahmed, Yoshua Bengio, and Hossein Hajimirsadeghi in their paper "Were RNNs All We Needed?" which can be found here: https://arxiv.org/abs/2410.01201
Both of these groups have been a major help and inspiration for this starter project I've been working on, so I can't thank them enough!

Here are a few images showing one of the better model's statistics: 

![](https://github.com/WonderingPuzzler/MiBi_LSTM/blob/main/v9/MiBi_curves_v9.png)


![](https://github.com/WonderingPuzzler/MiBi_LSTM/blob/main/v9/MiBi_confusion_matrix_v9.png)


![](https://github.com/WonderingPuzzler/MiBi_LSTM/blob/main/v9/MiBI_t_SNE_v9.png)


![](https://github.com/WonderingPuzzler/MiBi_LSTM/blob/main/v9/MiBi_classification_v9.png)


![](https://github.com/WonderingPuzzler/MiBi_LSTM/blob/main/v9/MiBi_ROC_v9.png)

Finally, I've included some model state dictionaries of a couple of the best versions I've trained, in case you want to mess around with them yourself!

## Usage

To run locally, you will likely need to run this command in Python:

```bash
# Downloads requirement libraries
pip install -r requirements.txt
```

Then, you will need to download and unzip the LibreSeVoc Dataset (Warning: *very* large): https://zenodo.org/records/15127251

Finally, you will need to change the directory for main.py to scan to whatever directory you put the extracted folders in (Warning: it MUST be the directory which shows the folders for each individual class. Otherwise, the program WON'T know what to scan)

However, I've also created a Google Colab Notebook (called MiBi-LSTM_Deepfake_Detector_Notebook.ipynb, coming in the next couple days, have a few edits to make to it) in case you want to run it without taking up any space on your computer (Note: The dataset and model state dictionary *will* need to take up space somewhere. I used Google Drive.)

Again, in both cases, you'll need to change the directory of wherever you decide to put the extracted LibriSeVoc Dataset and Model State Dictionary for the program to function properly.

## Credits

Huge thanks again to Chengzhe Sun, Shan Jia, Shuwei Hou, and Siwei Lyu, who created the dataset I used to train my model, as well as for being the major inspiration behind this project! You can view their paper and original project here:

C. Sun, S. Jia, S. Hou, and S. Lyu, “AI-synthesized voice detection using neural vocoder artifacts,” 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pp. 904–912, Jun. 2023. doi:10.1109/cvprw59228.2023.00097 

C. Sun, S. Hou, and S. Shi, “Synthetic-Voice-Detection-Vocoder-Artifacts,” GitHub, https://github.com/csun22/Synthetic-Voice-Detection-Vocoder-Artifacts (accessed Dec. 4, 2025). 

Additionally, thank you again to Leo Feng, Frederick Tung, Mohamed Osama Ahmed, Yoshua Bengio, and Hossein Hajimirsadeghi for their work in "Were RNNs All We Needed?" for their minLSTM model being the main inspiration for MiBi-LSTM! Once more, their paper can be found here:

https://arxiv.org/abs/2410.01201


-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

Other important sources I used include the following (Citations in IEEE format):

A. Bhandari, “Guide to AUC ROC Curve in Machine Learning,” Analytics Vidhya, https://www.analyticsvidhya.com/blog/2020/06/auc-roc-curve-machine-learning/ (accessed Dec. 4, 2025).

A. Parisi, Hands-on Artificial Intelligence for Cybersecurity: Implement Smart AI Systems for Preventing Cyber Attacks and Detecting Threats and Network Anomalies. Birmingham, UK: Packt Publishing, 2019. 

A. Rawat, “The VOICE- Real-Time Voice Cloning,” Medium, https://medium.com/analytics-vidhya/the-voice-real-time-voice-cloning-b7815afd6869 (accessed Dec. 5, 2025). 

A. Yadav, “Multiclass receiver operating characteristic (ROC) in Scikit Learn,” GeeksforGeeks, https://www.geeksforgeeks.org/machine-learning/multiclass-receiver-operating-characteristic-roc-in-scikit-learn/ (accessed Dec. 4, 2025). 

Almutairi, Z., & Elgibreen, H. (2022). A Review of Modern Audio Deepfake Detection Methods: Challenges and Future Directions. Algorithms, 15(5), 155. https://doi.org/10.3390/a15050155 

B. Zhang, H. Cui, V. Nguyen, and M. Whitty, “Audio deepfake detection: What has been achieved and what lies ahead,” Sensors, vol. 25, no. 7, p. 1989, Mar. 2025. doi:10.3390/s25071989 

C. Hughes, “Demystifying pytorch’s Weightedrandomsampler by example,” Medium, https://towardsdatascience.com/demystifying-pytorchs-weightedrandomsampler-by-example-a68aceccb452/ (accessed Dec. 4, 2025). 

C. Sun, LibriSeVoc Dataset, (May 02, 2023). doi: 10.5281/zenodo.15127251.  

C. Y. Wijaya, “Breaking Down the Classification Report from Scikit-Learn - NBD Lite #6,” Non-Brand Data, https://www.nb-data.com/p/breaking-down-the-classification (accessed Dec. 4, 2025). 

C. Yang, “How to compute equal error rate (EER) on ROC curve,” Changjiang’s blog, https://yangcha.github.io/EER-ROC/ (accessed Dec. 4, 2025). 

D. Anikin, “Don’t believe your ears: voice deepfakes,” Kaspersky, https://www.kaspersky.com/blog/audio-deepfake-technology/48586/ (accessed Dec. 4, 2025). 

D. Bourke, “00. PyTorch Fundamentals,” Zero to Mastery Learn PyTorch for Deep Learning, https://www.learnpytorch.io/00_pytorch_fundamentals/ (accessed Dec. 4, 2025). 

D. Bourke, “01. PyTorch Workflow Fundamentals,” Zero to Mastery Learn PyTorch for Deep Learning, https://www.learnpytorch.io/01_pytorch_workflow/ (accessed Dec. 4, 2025). 

D. Bourke, “02. PyTorch Neural Network Classification,”  Zero to Mastery Learn PyTorch for Deep Learning, https://www.learnpytorch.io/02_pytorch_classification/ (accessed Dec. 4, 2025). 

D. Bourke, “03. PyTorch Computer Vision ,” Zero to Mastery Learn PyTorch for Deep Learning, https://www.learnpytorch.io/03_pytorch_computer_vision/ (accessed Dec. 4, 2025). 

E. Deruty, “Intuitive understanding of mfccs,” Medium, https://medium.com/@derutycsl/intuitive-understanding-of-mfccs-836d36a1f779 (accessed Dec. 5, 2025). 

F. T. Winata, N. J. Tanuwijaya, R. Setiawan, and R. Y. Rumagit, “Comparison of deepfake detection using CNN and Hybrid models,” Procedia Computer Science, vol. 269, pp. 1556–1564, 2025. doi:10.1016/j.procs.2025.09.097 

Google Deepmind, “WaveNet,” Google DeepMind, https://deepmind.google/research/wavenet/ (accessed Dec. 4, 2025). 

H. Joshi, “Understanding Feature Scaling in Machine Learning: Techniques, Implementation, and Advantages,” Medium, https://python.plainenglish.io/understanding-feature-scaling-in-machine-learning-techniques-implementation-and-advantages-fd9065a349aa (accessed Dec. 5, 2025). 

H. Nguyen, M. Harris, S. Sengupta, and J. D. Owens, “Chapter 39. Parallel Prefix Sum (Scan) with CUDA,” NVIDIA Developer. https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-39-parallel-prefix-sum-scan-cuda (accessed Jan. 12, 2026).

H. Shandilya, “Training Neural Networks with validation using pytorch,” GeeksforGeeks, https://www.geeksforgeeks.org/machine-learning/training-neural-networks-with-validation-using-pytorch/ (accessed Dec. 3, 2025). 

I. Goodfellow, Y. Bengio, and A. Courville, Deep Learning. Cambridge, MA: The MIT Press, 2017. 

J. Brownlee, “How to use ROC curves and precision-recall curves for classification in Python,” MachineLearningMastery, https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/ (accessed Dec. 4, 2025). 

J. Gavande, “Bar plot in Matplotlib,” GeeksforGeeks, https://www.geeksforgeeks.org/pandas/bar-plot-in-matplotlib/ (accessed Dec. 4, 2025). 

J. Hunter, D. Dale, E. Firing, M. Droettboom, and Matplotlib development team, “Create multiple subplots using plt.subplots#,” Matplotlib, https://matplotlib.org/stable/gallery/subplots_axes_and_figures/subplots_demo.html (accessed Dec. 4, 2025).

J. Yi, C. Wang, J. Tao, X. Zhang, C. Y. Zhang, and Y. Zhao, ‘Audio Deepfake Detection: A Survey’, arXiv [cs.SD]. 2023. 

K. Erdem, “T-SNE clearly explained. an intuitive explanation of T-sne… | by Kemal Erdem (Burnpiro) | TDS archive | medium,” Medium, https://medium.com/data-science/t-sne-clearly-explained-d84c537f53a (accessed Dec. 5, 2025). 

K. Kumar et al., ‘MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis’, arXiv [eess.AS]. 2019. 

K. Schäfer, J.-E. Choi, and S. Zmudzinski, “Explore the world of audio deepfakes: A guide to detection techniques for Non-Experts,” 3rd ACM International Workshop on Multimedia AI against Disinformation, pp. 13–22, Jun. 2024. doi:10.1145/3643491.3660289 

Kvpratama, “Audio classification with LSTM and torchaudio,” Kaggle, https://www.kaggle.com/code/kvpratama/audio-classification-with-lstm-and-torchaudio/notebook (accessed Dec. 4, 2025). 

L. Feng, F. Tung, M. O. Ahmed, Y. Bengio, and H. Hajimirsadeghi, ‘Were RNNs All We Needed?’, arXiv [cs.LG]. 2024.

L. Roberts, “Understanding the Mel Spectrogram | by Leland Roberts | Analytics Vidhya | Medium,” Medium, https://medium-com.translate.goog/analytics-vidhya/understanding-the-mel-spectrogram-fca2afa2ce53?_x_tr_sl=en&_x_tr_tl=pt&_x_tr_hl=pt-PT&_x_tr_pto=tc (accessed Dec. 5, 2025). 

librosa development team, “librosa.feature.melspectrogram,” librosa, https://librosa.org/doc/0.11.0/generated/librosa.feature.melspectrogram.html (accessed Dec. 4, 2025). 

librosa development team, “librosa.feature.mfcc,” librosa, https://librosa.org/doc/main/generated/librosa.feature.mfcc.html (accessed Dec. 4, 2025). 

librosa development team, “librosa.stft,” librosa, https://librosa.org/doc/main/generated/librosa.stft.html (accessed Dec. 4, 2025).

librosa development team, “Using display.specshow,” librosa, https://librosa.org/doc/0.11.0/auto_examples/plot_display.html (accessed Dec. 4, 2025). 

M. A. I. Khan, “Training and validation data in pytorch,” MachineLearningMastery, https://machinelearningmastery.com/training-and-validation-data-in-pytorch/ (accessed Dec. 3, 2025). 

M. Alahmid, “FAR, FRR and EER with python,” Medium, https://becominghuman.ai/face-recognition-system-and-calculating-frr-far-and-eer-for-biometric-system-evaluation-code-2ac2bd4fd2e5 (accessed Dec. 5, 2025). 

M. Li, Y. Ahmadiadli, and X.-P. Zhang, “A Survey on Speech Deepfake Detection,” ACM Computing Surveys, vol. 57, no. 7, pp. 1–38, Feb. 2025. doi:10.1145/3714458 

N. Shah, “Voice Conversion using Generative Techniques,” CS230 - Stanford, http://cs230.stanford.edu/projects_fall_2020/reports/55721255.pdf (accessed Dec. 4, 2025). 

NumPy Developers, “numpy.concatenate,” NumPy, https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html (accessed Dec. 4, 2025). 

NumPy Developers, “numpy.ndarray,” NumPy, https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html (accessed Dec. 4, 2025). 

NumPy Developers, “numpy.std,” NumPy, https://numpy.org/doc/stable/reference/generated/numpy.std.html (accessed Dec. 4, 2025). 

NVIDIA Corporation, “CUDA Programming Guide Release 13.1 NVIDIA Corporation,” NVIDIA Corporation & Affiliates, Dec. 2025. Accessed: Jan. 12, 2026. [Online]. Available: https://docs.nvidia.com/cuda/cuda-programming-guide/pdf/cuda-programming-guide.pdf

P. Belagatti, “Understanding the softmax activation function: A comprehensive guide,” SingleStore, https://www.singlestore.com/blog/a-guide-to-softmax-activation-function/#:~:text=The%20softmax%20function%2C%20often%20used%20in%20the,by%20the%20sum%20of%20all%20the%20exponentials (accessed Dec. 4, 2025). 

PyTorch Contributers, “CrossEntropyLoss,” PyTorch, https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html (accessed Dec. 4, 2025). 

PyTorch Contributors, “Custom C++ and CUDA Operators,” Pytorch.org, Jan. 28, 2025. https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html (accessed Jan. 12, 2026).

PyTorch Contributers, “LSTM,” PyTorch, https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html (accessed Dec. 3, 2025). 

PyTorch Contributers, “torch.uitils.data,” PyTorch, https://docs.pytorch.org/docs/stable/data.html#torch.utils.data.WeightedRandomSampler (accessed Dec. 4, 2025). 

S. Raheja, “Train-test-validation split in 2025,” Analytics Vidhya, https://www.analyticsvidhya.com/blog/2023/11/train-test-validation-split/ (accessed Dec. 4, 2025). 

Saidrasool, “Speech recognition using LSTM: A step-by-step guide | by Saidrasool | Medium,” Medium, https://medium.com/@saidrasool402/speech-recognition-using-lstm-a-step-by-step-guide-78e3ee7e7d5f (accessed Dec. 5, 2025). 

Saturn Cloud, “Using weights in Crossentropyloss and BCELoss (pytorch),” Saturn Cloud Blog, https://saturncloud.io/blog/using-weights-in-crossentropyloss-and-bceloss-pytorch/ (accessed Dec. 4, 2025). 

scikit-learn developers, “classification_report,” scikit, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html (accessed Dec. 4, 2025). 

scikit-learn developers, “confusion_matrix,” scikit-learn, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html (accessed Dec. 4, 2025). 

scikit-learn developers, “roc_auc_score,” scikit-learn, https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html (accessed Dec. 4, 2025). 

scikit-learn developers, “StandardScaler,” scikit-learn, https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html (accessed Dec. 4, 2025). 

SciPy Community, “Softmax,” SciPy, https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.softmax.html (accessed Dec. 4, 2025). 

T. Segura, “‘Train, Validation, Test Split’ explained in 200 words.,” Data Science - One Question at a time, https://thaddeus-segura.com/train-test-split/ (accessed Dec. 4, 2025). 

V. Trevisan, “Multiclass classification evaluation with ROC curves and Roc Auc,” Towards Data Science, https://towardsdatascience.com/multiclass-classification-evaluation-with-roc-curves-and-roc-auc-294fd4617e3a/ (accessed Dec. 4, 2025). 

Vishal, “Compute the mean, standard deviation, and variance of a given Numpy Array,” GeeksforGeeks, https://www.geeksforgeeks.org/python/compute-the-mean-standard-deviation-and-variance-of-a-given-numpy-array/ (accessed Dec. 4, 2025). 

W3Schools, “NumPy Joining Array,” W3Schools, https://www.w3schools.com/python/numpy/numpy_array_join.asp (accessed Dec. 4, 2025). 

Y. A. Li, A. Zare, and N. Mesgarani, ‘StarGANv2-VC: A Diverse, Unsupervised, Non-parallel Framework for Natural-Sounding Voice Conversion’, arXiv [cs.SD]. 2021. 

Z. Ahmed and Resemble.AI, “Understanding AI voice cloning: What, why, and how,” Resemble AI, https://www.resemble.ai/understanding-ai-voice-cloning/#:~:text=TTS%20models%20generate%20speech%20from,a%20few%20seconds%20of%20speech. (accessed Dec. 4, 2025). 

Z. Almutairi and H. Elgibreen, “A review of modern audio deepfake detection methods: Challenges and future directions,” Algorithms, vol. 15, no. 5, p. 155, May 2022. doi:10.3390/a15050155 

Z. Khanjani, G. Watson, and V. P. Janeja, “Audio Deepfakes: A survey,” Frontiers in Big Data, vol. 5, Jan. 2023. doi:10.3389/fdata.2022.1001063
