# Importing Needed Libraries
import sys

# Importing my own modules
from Modules.Image_loader import data_loader, save_dataset_tf
from Modules.Face_Detector import face_detector
from Modules.Model import SmileDetectionModel


def main():
    if SAVE_DATA_FOR_TF:
        # Loading images and their labels
        images, labels = data_loader(GENKI_IMAGE_ROOT, GENKI_LABEL_ROOT)
        print("|----------------------- Dataset Loaded -----------------------|\n")

        # Detecting face in the images
        images = face_detector(images)
        print("|----------------------- Face Detection Done -----------------------|\n")

        # Saving the images in the new directory for tensorflow
        save_dataset_tf(images, labels, DATA_PATH)
        print("|----------------------- Dataset Saved! -----------------------|\n")

    # Creating an object of the model to work with
    classifier = SmileDetectionModel(DATA_PATH, BATCH_SIZE, SHUFFLE, TEST_RATE, VALID_RATE)
    data_augmentation = classifier.data_augmentation(filp_augment=True, brightness_augment=True)
    classifier.compile_model(data_augmentation)
    classifier.model.summary()
    classifier.train()
    print("\n|----------------------- Model Trained -----------------------|\n")
    loss, accuracy = classifier.test()
    classifier.report()

    # Ask for saving
    user_entry = input("Do you want to save this model?(y,N): ")
    if user_entry == "y":
        # Saving the Model
        classifier.model.save("Model/CNN_Trained.h5")

        print("-- Model Saved Successfully in ./Model/CNN_Trained.pkl --")

    elif user_entry == "N":
        # Ignoring the trained Model
        print("--- Model didn't saved ---")
    return


if __name__ == "__main__":
    # Setting Genki dataset address
    GENKI_IMAGE_ROOT = "../Genki Dataset/genki4k/files/"
    GENKI_LABEL_ROOT = "../Genki Dataset/genki4k/labels.txt"
    DATA_PATH = "./Genki New Dir/"
    BATCH_SIZE = 20
    TEST_RATE = 3
    VALID_RATE = 10
    SHUFFLE = True

    # Asking user to load the raw images
    entry = input("Do you want to load raw images?(Y,n) ")

    # Setting SAVE_DATA_FOR_TF
    if entry == "Y":
        SAVE_DATA_FOR_TF = True

    elif entry == "n":
        SAVE_DATA_FOR_TF = False

    else:
        print("-- Entry is not correct! --")
        sys.exit()

    main()
