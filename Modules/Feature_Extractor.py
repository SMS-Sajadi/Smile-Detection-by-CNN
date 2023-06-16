# Importing Needed Libraries
import numpy as np
from skimage.feature import hog, local_binary_pattern


def feature_extract(images, histogram_feature=False):
    """
    This function will extract hog and lbp feature of images and return the feature matrix
    histogram_feature is a flag that will decide the lbp or its histogram will bre returned
    :param images:
    :param histogram_feature:
    :return image_features matrix:
    """
    # A list for saving features
    image_features = []
    radius = 3
    n_points = 8 * radius
    for image in images:
        # Extracting HOG of image
        hog_features = hog(image, orientations=16, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False,
                           feature_vector=True)

        # Extracting LBP of image
        lbp_features = local_binary_pattern(image, n_points, radius, method='default')

        # Creating the histogram of the lbp
        if histogram_feature:
            lbp_features, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

        # Adding features to the matrix
        image_features.append(np.concatenate((hog_features, lbp_features.flatten())))

    # Change list to numpy array
    image_features = np.array(image_features)

    return image_features


def feature_extract_single(image, histogram_feature=False):
    """
    This function will extract hog and lbp feature of image and return the feature matrix
    histogram_feature is a flag that will decide the lbp or its histogram will bre returned
    :param image:
    :param histogram_feature:
    :return image_features matrix:
    """
    # A list for saving features
    image_features = []
    radius = 3
    n_points = 8 * radius

    # Extracting HOG of image
    hog_features = hog(image, orientations=16, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=False, feature_vector=True)

    # Extracting LBP of image
    lbp_features = local_binary_pattern(image, n_points, radius, method='default')

    # Creating the histogram of the lbp
    if histogram_feature:
        lbp_features, _ = np.histogram(lbp_features.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))

    # Adding features to the matrix
    image_features.append(np.concatenate((hog_features, lbp_features.flatten())))

    # Change list to numpy array
    image_features = np.array(image_features)

    return image_features
