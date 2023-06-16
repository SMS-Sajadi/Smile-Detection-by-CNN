# Importing Needed Libraries
import cv2


def face_detector(images):
    """
    This function gets a list of images and will crop the faces in the images and return the list of them
    :param images:
    :return faces:
    """

    # A list for saving the faces
    faces = []

    # Creating the face detector obj
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Iterating over the images for face detection
    for img in images:
        # Detecting face in image
        face_detected = face_detector.detectMultiScale(img, 1.1, 4)

        # Handling if no face is detected
        if len(face_detected) == 0:
            # Maybe a face will be detected in the flip image
            flip_img = cv2.flip(img, 1)
            face_detected = face_detector.detectMultiScale(flip_img, 1.1, 4)

        # Iterating to reach the last coordinates of the face
        for (x, y, w, h) in face_detected:
            pass

        # Handling if really there is no face detected
        if len(face_detected) == 0:
            x = 0
            y = 0
            h = img.shape[0]
            w = img.shape[1]

        # Save the detected face
        face_crop = img[y: y + h, x: x + w]
        faces.append(cv2.resize(face_crop, (64,64)))

    return faces


def face_detector_single(img):
    """
    This function gets an image and will crop the face in the image and return the list of them
    :param img:
    :return faces, loc:
    """

    # Changing image into gray scale
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # A list for saving the faces and their locations
    faces = []
    loc = []

    # Creating the face detector obj
    face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Detecting face in image
    face_detected = face_detector.detectMultiScale(image, 1.3, 5)

    # Saving the faces' location to loc and crop the image
    for (x, y, w, h) in face_detected:
        # Location of face
        loc.append((x,y,w,h))

        # Save the detected face
        face_crop = image[y: y + h, x: x + w]
        faces.append(cv2.resize(face_crop, (64,64)))

    return faces, loc
