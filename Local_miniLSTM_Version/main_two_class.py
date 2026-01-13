from DeepFake_Detector_Imports import *
from classes_files_dictionary_set_up import ClassesFilesDictionarySetUp

def main():
    classes = ClassesFilesDictionarySetUp(directory='/content/drive/My Drive/CYBR_4980_Project/Dataset_Extracted/LibriSeVoc_extracted/LibriSeVoc', file_extension='.wav',two_class=True)


if __name__ == "__main__":
    main()

    