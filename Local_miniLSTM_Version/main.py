from deep_fake_detector_graphs_and_stats import *

def main() -> None:
    try:
        # Set up new path for dataset
        new_path = Path('E:')

        # If the new path exists, change to that directory
        if new_path.exists():
            os.chdir(new_path)
            print(f"Changed working directory to: {new_path}")

            detector = DeepFakeDetectorGraphsAndStats(directory='/LibriSeVoc', file_extension='.wav', loss='CrossEntropyLoss',  optim='Adam', DL_type='RNN', data_type='mfcc', two_class=False, random_seed=42)

            detector.set_non_gt_subset_count(1000)  # 1000 non-ground truth samples
            detector.set_gt_subset_count(1250)      # 1250 ground truth samples

            detector.set_batch_size(32)             # 32 batch size
            detector.set_sample_rate(22050)
            detector.set_duration(7)                # 7 second audio clips

            # Set up file dictionary by scanning the dataset directory
            detector.setup_file_dictionary('.wav')

            detector.set_learning_rate(0.0008)  # Learning rate of 0.001
            detector.set_optim("Adam")          # Using Adam optimizer

            detector.setup_data_loaders()
            detector.train_MiBiLSTM(num_epochs=300)  # 300 epochs
            detector.save_model('/LibriSeVoc/MiBI_LSTM_V9.pth')

            detector.plot_training_curves()
            detector.plot_class_counts()
            detector.print_optimizer_loss_architecture('/LibriSeVoc/MiBI_LSTM_V9.pth')
            detector.evaluate_model('/LibriSeVoc/MiBI_LSTM_V9.pth')
            detector.sns_scatter_plot('/LibriSeVoc/MiBI_LSTM_V9.pth')
            detector.create_confusion_matrix('/LibriSeVoc/MiBI_LSTM_V9.pth')
            detector.create_classification_report('/LibriSeVoc/MiBI_LSTM_V9.pth')
            detector.plot_roc_curve_with_eer('/LibriSeVoc/MiBI_LSTM_V9.pth')
        else:
            raise FileNotFoundError(f"The specified path does not exist: {new_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()