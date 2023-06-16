# Importing Needed Libraries
import cv2
import pickle
import sys

# Importing my own moduls
from Modules.Face_Detector import face_detector_single
from Modules.Feature_Extractor import feature_extract_single


def main():
    # Loading the trained model
    with open("./Model/SVM_Trained.pkl", "rb") as f:
        svm_model = pickle.load(f)
        print("-- Model loaded Successfully --")

    # Accessing Source and writing on it
    capture = cv2.VideoCapture(capture_source)

    # Reading the Camera
    while(True):
        # Reading the frame
        ret, frame = capture.read()

        # Check if frame is read
        if ret:
            # Resizing frame for better showing
            frame = cv2.resize(frame, size_of_source)

            # Detecting all faces in the frame and find their locations
            faces, locs = face_detector_single(frame)

            # Iterating over the faces detected
            for i, face in enumerate(faces):
                # Set the locations of the face detected
                x, y, w, h = locs[i]

                # Extracting the features of the frame
                feature = feature_extract_single(face)

                # Get prediction from the model
                output = svm_model.predict(feature)

                # if smile predicted
                if output[0] == '1':
                    # It will show that smile is detected
                    print(f"Face {i} has smile!")

                    # Color will be Green
                    color = (0, 255, 0)
                    text = 'Smile Detected'
                else:
                    # It will show that there is no smile
                    print(f"Face {i} has no smile!")

                    # if smile is not detected, color will be red
                    color = (0, 0, 255)
                    text = 'No Smile'

                # Setting font properties
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.5
                thickness = 1

                # Setting location of text
                org = (x, y - 10)

                # Creating a rectangle and a text with the given locations and color based on smile detection
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, text, org, font, fontScale, color, thickness, cv2.LINE_AA)

            # Showing the frame with rectangle
            cv2.imshow("Video", frame)
            cv2.waitKey(1)

        else:
            # Frame is not read
            print("-- Frame is not loaded successfully or source is ended --")
            break

    # Closing all windows
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # Source for capturing frames from
    capture_source = "20230521_142847.mp4"

    # Asking user to choose capture source
    entry = input("Do you want to switch to webcam?(Y,n) ")

    # Setting capture source and size of output
    if entry == "Y":
        capture_source = 0
        size_of_source = (1080, 720)

    elif entry == "n":
        size_of_source = (480, 720)

    else:
        print("-- Entry is not correct! --")
        sys.exit()

    main()
