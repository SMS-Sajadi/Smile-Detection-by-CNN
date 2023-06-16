# Importing Needed Libraries
import tensorflow as tf
from sklearn.metrics import accuracy_score
import pickle
import sys

# Importing my own modules
from Modules.Image_loader import data_loader, save_dataset_tf
from Modules.Face_Detector import face_detector
from Modules.Feature_Extractor import feature_extract
from Modules.Train_Test_Split import data_split
from Modules.Data_Augmentation import augment


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

    # Training the SVM
    svm_model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0))
    svm_model.fit(train_feature_matrix, train_labels)
    print("|----------------------- Model Trained -----------------------|\n")

    # Finding the train accuracy
    print("|----------------------- Training Result -----------------------|")
    train_output = svm_model.predict(train_feature_matrix)
    train_accuracy = accuracy_score(train_output, train_labels)
    print(f"Accuracy: %{train_accuracy*100} on {len(train_feature_matrix)} train data\n")

    # Testing the Model
    print("|----------------------- Testing Result -----------------------|")
    test_output = svm_model.predict(test_feature_matrix)
    test_accuracy = accuracy_score(test_output, test_labels)
    print(f"Accuracy: %{test_accuracy*100} on {len(test_feature_matrix)} train data\n\n")

    # Ask for saving
    user_entry = input("Do you want to save this model?(y,N): ")
    if user_entry == "y":
        # Saving the Model
        with open("Model/SVM_Trained.pkl", "wb") as f:
            pickle.dump(svm_model, f)

        print("-- Model Saved Successfully in ./Model/SVM_Trained.pkl --")

    elif user_entry == "N":
        # Ignoring the trained Model
        print("--- Model didn't saved ---")
    return


if __name__ == "__main__":
    # Setting Genki dataset address
    GENKI_IMAGE_ROOT = "../Genki Dataset/genki4k/files/"
    GENKI_LABEL_ROOT = "../Genki Dataset/genki4k/labels.txt"
    DATA_PATH = "./Genki New Dir/"

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
