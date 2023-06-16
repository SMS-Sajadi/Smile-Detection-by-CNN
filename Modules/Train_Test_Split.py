# Importing Needed Libraries
from sklearn.model_selection import train_test_split


def data_split(images, labels, size, shuffling=False):
    """
    This function will split the data into tst and train based on size provided
    :param images:
    :param labels:
    :param size:
    :param shuffling:
    :return train_img, train_label, test_img, test_label:
    """
    # Splitting dataset
    train_img, test_img, train_label, test_label = train_test_split(images, labels, test_size=size, shuffle=shuffling)

    return train_img, train_label, test_img, test_label

