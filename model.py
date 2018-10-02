import glob
import pickle

from skimage.feature import hog

import features

# cars = glob.glob('data/small/vehicles_smallset/**/*.jpeg', recursive=True)
# notcars = glob.glob('data/small/non-vehicles_smallset/**/*.jpeg', recursive=True)
train_image_format = "jpeg"

cars = glob.glob('data/full/vehicles/**/*.png', recursive=True)
notcars = glob.glob('data/full/non-vehicles/**/*.png', recursive=True)
train_image_format = "png"

print("car_images=",len(cars))
print("notcar_images=", len(notcars))


# car_features = extract_features(cars, cspace='RGB', spatial_size=(32, 32),
#                                 hist_bins=32, hist_range=(0, 256))
# notcar_features = extract_features(notcars, cspace='RGB', spatial_size=(32, 32),
#                                    hist_bins=32, hist_range=(0, 256))

# if len(car_features) > 0:
#     # Create an array stack of feature vectors
#     X = np.vstack((car_features, notcar_features)).astype(np.float64)
#     # Fit a per-column scaler
#     X_scaler = StandardScaler().fit(X)
#     # Apply the scaler to X
#     scaled_X = X_scaler.transform(X)
#     car_ind = np.random.randint(0, len(cars))
#     # Plot an example of raw and scaled features
#     fig = plt.figure(figsize=(12, 4))
#     plt.subplot(131)
#     plt.imshow(mpimg.imread(cars[car_ind]))
#     plt.title('Original Image')
#     plt.subplot(132)
#     plt.plot(X[car_ind])
#     plt.title('Raw Features')
#     plt.subplot(133)
#     plt.plot(scaled_X[car_ind])
#     plt.title('Normalized Features')
#     fig.tight_layout()
#     fig.savefig("tmp/features.jpeg")
# else:
#     print('Your function only returns empty feature vectors...')

import matplotlib.image as mpimg
import numpy as np
import cv2
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid
# for scikit-learn version <= 0.17
# if you are using scikit-learn >= 0.18 then use this:
from sklearn.model_selection import train_test_split

# from sklearn.cross_validation import train_test_split

# TODO play with these values to see how your classifier
# performs under different binning scenarios
colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8  # HOG pixels per cell
cell_per_block = 2  # HOG cells per block
hog_channel = 0  # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32)  # Spatial binning dimensions
hist_bins = 32  # Number of histogram bins
spatial_feat = True  # Spatial features on or off
hist_feat = True  # Histogram features on or off
hog_feat = True  # HOG features on or off
y_start_stop = [None, None]  # Min and max in y to search in slide_window()

car_features = features.extract_features(cars, color_space=colorspace,
                                spatial_size=spatial_size, hist_bins=hist_bins,
                                orient=orient, pix_per_cell=pix_per_cell,
                                cell_per_block=cell_per_block,
                                hog_channel=hog_channel, spatial_feat=spatial_feat,
                                hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = features.extract_features(notcars, color_space=colorspace,
                                   spatial_size=spatial_size, hist_bins=hist_bins,
                                   orient=orient, pix_per_cell=pix_per_cell,
                                   cell_per_block=cell_per_block,
                                   hog_channel=hog_channel, spatial_feat=spatial_feat,
                                   hist_feat=hist_feat, hog_feat=hog_feat)

# Create an array stack of feature vectors
X = np.vstack((car_features, notcar_features)).astype(np.float64)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=rand_state)

# Fit a per-column scaler only on the training data
X_scaler = StandardScaler().fit(X_train)
# Apply the scaler to X_train and X_test
X_train = X_scaler.transform(X_train)
X_test = X_scaler.transform(X_test)
print('Using:', orient, 'orientations', pix_per_cell,
      'pixels per cell and', cell_per_block, 'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use
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

model = {
    "svc": svc,
    "scaler": X_scaler,
    "orient": orient,
    "pix_per_cell": pix_per_cell,  # HOG pixels per cell
    "cell_per_block": cell_per_block,  # HOG cells per block
    "hog_channel": hog_channel,  # Can be 0, 1, 2, or "ALL"
    "spatial_size": spatial_size,  # Spatial binning dimensions
    "hist_bins": hist_bins,  # Number of histogram bins
    "spatial_feat": spatial_feat,  # Spatial features on or off
    "hist_feat": hist_feat,  # Histogram features on or off
    "hog_feat": hog_feat,  # HOG features on or off
    "train_image_format": train_image_format,
    "colorspace": colorspace,
    "hog_channel": hog_channel,
}
filename = "svc.sav"
pickle.dump(model, open(filename, 'wb'))
