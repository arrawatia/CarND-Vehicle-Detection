import glob
import pickle
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import features


def load_data():
    # cars = glob.glob('data/small/vehicles_smallset/**/*.jpeg', recursive=True)
    # notcars = glob.glob('data/small/non-vehicles_smallset/**/*.jpeg', recursive=True)
    # train_image_format = "jpeg"

    cars = glob.glob('data/full/vehicles/**/*.png', recursive=True)[0:100]
    notcars = glob.glob('data/full/non-vehicles/**/*.png', recursive=True)[0:100]

    print("car_images=", len(cars))
    print("notcar_images=", len(notcars))

    return cars, notcars


def extract_features(images, config):
    orient = config["orient"]
    pix_per_cell = config["pix_per_cell"]
    cell_per_block = config["cell_per_block"]
    spatial_size = config["spatial_size"]
    hist_bins = config["hist_bins"]
    train_image_format = config["train_image_format"]
    colorspace = config["colorspace"]
    hog_channel = config["hog_channel"]
    spatial_feat = config["spatial_feat"]
    hist_feat = config["hist_feat"]
    hog_feat = config["hog_feat"]

    return features.extract_features(images,
                                     color_space=colorspace,
                                     spatial_size=spatial_size,
                                     hist_bins=hist_bins,
                                     orient=orient,
                                     pix_per_cell=pix_per_cell,
                                     cell_per_block=cell_per_block,
                                     hog_channel=hog_channel,
                                     spatial_feat=spatial_feat,
                                     hist_feat=hist_feat,
                                     hog_feat=hog_feat)


def process_data(cars, notcars, config):
    car_features = extract_features(cars, config)
    notcar_features = extract_features(notcars, config)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

    # Fit a per-column scaler only on the training data
    X_scaler = StandardScaler().fit(X_train)
    # Apply the scaler to X_train and X_test
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_scaler, X_train, X_test, y_train, y_test


def train_svm(config, X_train, X_test, y_train, y_test):
    orient = config["orient"]
    pix_per_cell = config["pix_per_cell"]
    cell_per_block = config["cell_per_block"]

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with SVC')

    return svc


def train_rf(config, X_train, X_test, y_train, y_test):
    orient = config["orient"]
    pix_per_cell = config["pix_per_cell"]
    cell_per_block = config["cell_per_block"]

    print('Using:', orient, 'orientations', pix_per_cell, 'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))

    # Use a linear SVC
    rf = RandomForestClassifier(n_estimators=50, min_samples_split=7, min_samples_leaf=1)
    # Check the training time for the SVC
    t = time.time()
    rf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train RF...')

    # Check the score of the SVC
    print('Test Accuracy of RF = ', round(rf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t = time.time()
    n_predict = 10
    print('My RF predicts: ', rf.predict(X_test[0:n_predict]))
    print('For these', n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2 - t, 5), 'Seconds to predict', n_predict, 'labels with RF')
    return rf


def main():
    config = {
        # performs under different binning scenarios
        "colorspace": 'RGB2YCrCb',
        "orient": 9,  # HOG orientations
        "pix_per_cell": 8,  # HOG pixels per cell
        "cell_per_block": 2,  # HOG cells per block
        "hog_channel": 0,  # Can be 0, 1, 2, or "ALL"
        "spatial_size": (32, 32),  # Spatial binning dimensions
        "hist_bins": 32,  # Number of histogram bins
        "spatial_feat": True,  # Spatial features on or off
        "hist_feat": True,  # Histogram features on or off
        "hog_feat": True,  # HOG features on or off
        "train_image_format": "png",
    }
    cars, notcars = load_data()
    X_scaler, X_train, X_test, y_train, y_test = process_data(cars, notcars, config)
    svc = train_svm(config, X_train, X_test, y_train, y_test)
    rf = train_rf(config, X_train, X_test, y_train, y_test)
    model = {
        "svc": svc,
        "svc": rf,
        "scaler": X_scaler,
        "config": config
    }
    filename = "svc.sav"
    pickle.dump(model, open(filename, 'wb'))


if __name__ == '__main__':
    main()
