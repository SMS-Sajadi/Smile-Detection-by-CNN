# Importing Needed Libraries
import cv2
import numpy as np


def augment(images, labels, flip=True, light=True):
    """
    This function will create new images based on images provided to it
    flip=True will create new images by flipping
    light=True will create new images by changing light of all images
    :param images:
    :param labels:
    :param flip:
    :param light:
    :return images, labels:
    """

    # Temp lists for storing augmented images
    new_images = []
    new_labels = []

    # if flip=True, it will create images by flipping
    if flip:
        # Flipping each image
        for img, label in zip(images, labels):
            new_img = cv2.flip(img, 1)
            new_images.append(new_img)
            new_labels.append(label)

    # if light=True, it will create images by randomly changing light of images
    if light:
        # Change light of each image
        for img, label in zip(images, labels):
            # Create a random number to sum with each pixel
            rand_num = np.random.uniform(-125, 126)
            new_img = img.astype(np.float64)
            new_img += rand_num

            # Normalizing Image
            new_img = new_img - np.min(new_img)
            new_img /= np.max(new_img)
            new_img *= 255

            # Save new Image
            new_img = new_img.astype(np.uint8)
            new_images.append(new_img)
            new_labels.append(label)

    # append new images and labels
    images = images + new_images
    labels = labels + new_labels

    # Shuffle images and labels
    temp = list(zip(images, labels))
    np.random.shuffle(temp)
    images, labels = zip(*temp)

    return images, labels
