# Importing Needed libraries
import cv2
import os


def data_loader(img_root, label_root):
    """
    This function will read images and their labels and returns them as a list
    :param img_root:
    :param label_root:
    :return dataset, labels:
    """
    data_set = []
    list_of_images = os.listdir(img_root)

    # Loading images
    for i in list_of_images:
        x = cv2.imread(os.path.join(img_root + i), cv2.IMREAD_GRAYSCALE)
        data_set.append(x)

    # Loading labels
    labels = []
    with open(label_root) as f:
        lines = f.readlines()
        for line in lines:
            labels.append(line.split()[0])

    return data_set, labels


def save_dataset_tf(images, labels, dir):
    """
    :param images: The list of images to save in the directory
    :param labels: The list of labels to save in the directory
    :return:
    """
    for i in range(len(images)):
        if labels[i] == '1':
            cv2.imwrite(os.path.join(dir + "/1/" + str(i + 1) + ".jpg"), images[i])
        else:
            cv2.imwrite(os.path.join(dir + "/0/" + str(i + 1) + ".jpg"), images[i])
