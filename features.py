import matplotlib.image as mpimg
from skimage.feature import hog


def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'COLOR_BGR2HSV':
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm='L2-Hys',
                                  transform_sqrt=False,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient,
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L2-Hys',
                       transform_sqrt=False,
                       visualise=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:, :, 0], size).ravel()
    color2 = cv2.resize(img[:, :, 1], size).ravel()
    color3 = cv2.resize(img[:, :, 2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def color_hist_plot(img, nbins=32):  # bins_range=(0, 256)
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features, channel1_hist, channel2_hist, channel3_hist


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            feature_image = convert_color(image, color_space)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)

        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)

        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot3d(pixels, colors_rgb,
           axis_labels=list("RGB"), axis_limits=((0, 255), (0, 255), (0, 255)), output=None):
    """Plot pixels in 3D."""

    # Create figure and 3D axes
    fig = plt.figure(figsize=(8, 8))
    ax = Axes3D(fig)

    # Set axis limits
    ax.set_xlim(*axis_limits[0])
    ax.set_ylim(*axis_limits[1])
    ax.set_zlim(*axis_limits[2])

    # Set axis labels and sizes
    ax.tick_params(axis='both', which='major', labelsize=14, pad=8)
    ax.set_xlabel(axis_labels[0], fontsize=16, labelpad=16)
    ax.set_ylabel(axis_labels[1], fontsize=16, labelpad=16)
    ax.set_zlabel(axis_labels[2], fontsize=16, labelpad=16)

    # Plot pixel values with colors given in colors_rgb
    ax.scatter(
        pixels[:, :, 0].ravel(),
        pixels[:, :, 1].ravel(),
        pixels[:, :, 2].ravel(),
        c=colors_rgb.reshape((-1, 3)), edgecolors='none')
    plt.savefig(output)

    return ax  # return Axes3D object for further manipulation


def main():
    import glob
    import matplotlib.pyplot as plt
    import os

    cars = glob.glob('data/full/vehicles/**/*.png', recursive=True)
    notcars = glob.glob('data/full/non-vehicles/**/*.png', recursive=True)

    print("car_images=", len(cars))
    print("notcar_images=", len(notcars))

    fig, axs = plt.subplots(8, 8, figsize=(16, 16))
    fig.subplots_adjust(hspace=.2, wspace=.001)
    axs = axs.ravel()

    # Step through the list and search for chessboard corners
    for i in np.arange(32):
        img = cv2.imread(cars[np.random.randint(0, len(cars))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('car image', fontsize=10)
        axs[i].imshow(img)
    for i in np.arange(32, 64):
        img = cv2.imread(notcars[np.random.randint(0, len(notcars))])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i].axis('off')
        axs[i].set_title('Not-car image', fontsize=10)
        axs[i].imshow(img)

    fig.savefig("output_images/data_sample.png")

    lane_images = glob.glob('test_images/*.jpg')

    output_path = "output_images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path_plots = "output_images/plots/colorspace"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for idx in ["8", "200", "500", "1001"]:
        # Read a color image
        img = cv2.imread("data/full/vehicles/KITTI_extracted/%s.png" % idx)

        # Select a small fraction of pixels to plot by subsampling it
        scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
        img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                               interpolation=cv2.INTER_NEAREST)

        # Convert subsampled image to desired color space(s)
        img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
        img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

        img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        img_small_YCC = cv2.cvtColor(img_small, cv2.COLOR_RGB2YCrCb)
        img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_RGB2LUV)

        # Plot and show
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.savefig("%s/%s" % (output_path_plots, os.path.basename("car-%s.png" % idx)))
        plot3d(img_small_RGB, img_small_rgb, axis_labels=list("RGB"),
               output="%s/%s" % (output_path_plots, os.path.basename("car-rgb-%s.png" % idx)))
        plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"),
               output="%s/%s" % (output_path_plots, os.path.basename("car-hsv-%s.png" % idx)))
        plot3d(img_small_YCC, img_small_rgb, axis_labels=list("YCC"),
               output="%s/%s" % (output_path_plots, os.path.basename("car-ycc-%s.png" % idx)))
        plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"),
               output="%s/%s" % (output_path_plots, os.path.basename("car-luv-%s.png" % idx)))

    for idx in ["8", "200", "500", "4000"]:
        # Read a color image
        img = cv2.imread("data/full/non-vehicles/Extras/extra%s.png" % idx)

        # Select a small fraction of pixels to plot by subsampling it
        scale = max(img.shape[0], img.shape[1], 64) / 64  # at most 64 rows and columns
        img_small = cv2.resize(img, (np.int(img.shape[1] / scale), np.int(img.shape[0] / scale)),
                               interpolation=cv2.INTER_NEAREST)

        # Convert subsampled image to desired color space(s)
        img_small_RGB = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)  # OpenCV uses BGR, matplotlib likes RGB
        img_small_rgb = img_small_RGB / 255.  # scaled to [0, 1], only for plotting

        img_small_HSV = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
        img_small_YCC = cv2.cvtColor(img_small, cv2.COLOR_RGB2YCrCb)
        img_small_LUV = cv2.cvtColor(img_small, cv2.COLOR_RGB2LUV)

        # Plot and show
        fig = plt.figure(figsize=(8, 8))
        plt.imshow(img)
        plt.savefig("%s/%s" % (output_path_plots, os.path.basename("non-car-%s.png" % idx)))
        plot3d(img_small_RGB, img_small_rgb, axis_labels=list("RGB"),
               output="%s/%s" % (output_path_plots, os.path.basename("non-car-rgb-%s.png" % idx)))
        plot3d(img_small_HSV, img_small_rgb, axis_labels=list("HSV"),
               output="%s/%s" % (output_path_plots, os.path.basename("non-car-hsv-%s.png" % idx)))
        plot3d(img_small_YCC, img_small_rgb, axis_labels=list("YCC"),
               output="%s/%s" % (output_path_plots, os.path.basename("non-car-ycc-%s.png" % idx)))
        plot3d(img_small_LUV, img_small_rgb, axis_labels=list("LUV"),
               output="%s/%s" % (output_path_plots, os.path.basename("non-car-luv-%s.png" % idx)))


    output_path_plots = "output_images/plots/hist"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for lane_image in lane_images:
        print(lane_image)
        image = cv2.imread(lane_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Color Features
        result, channel1_hist, channel2_hist, channel3_hist = color_hist_plot(image, nbins=32)
        bin_edges = channel1_hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2
        # Plot it
        f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
        f.tight_layout()
        ax1.bar(bin_centers, channel1_hist[0])
        ax1.set_title('channel1 Histogram')
        ax2.bar(bin_centers, channel2_hist[0])
        ax2.set_title('channel2 Histogram')
        ax3.bar(bin_centers, channel3_hist[0])
        ax3.set_title('channel3 Histogram')
        ax4.plot(result)
        ax4.set_title('Histogram feature vector')
        plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))

    output_path_plots = "output_images/plots/spatial"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for lane_image in lane_images:
        print(lane_image)
        image = cv2.imread(lane_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        size = (32, 32)
        temp_result = cv2.resize(image, size)

        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(temp_result)
        ax1.set_title('Resized 32X32 Image')
        ax2.plot(bin_spatial(image, size=(32, 32)))
        ax2.set_title('spatial distribution')
        plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))

    output_path_plots = "output_images/plots/hog"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for lane_image in lane_images:
        print(lane_image)
        image = cv2.imread(lane_image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        orient = 9
        pix_per_cell = 8
        cell_per_block = 8
        features0, hogimage0 = get_hog_features(image[:, :, 0], orient, pix_per_cell, cell_per_block,
                                              vis=True, feature_vec=True)
        features1, hogimage1 = get_hog_features(image[:, :, 1], orient, pix_per_cell, cell_per_block,
                                              vis=True, feature_vec=True)
        features2, hogimage2 = get_hog_features(image[:, :, 2], orient, pix_per_cell, cell_per_block,
                                              vis=True, feature_vec=True)

        f, ax = plt.subplots(7, 1, figsize=(16, 32))
        f.tight_layout()
        ax[0].imshow(image)
        ax[0].set_title('Original')
        ax[1].imshow(hogimage0)
        ax[1].set_title('Image channel0 with gradients')
        ax[2].plot(features0)
        ax[2].set_title('Features channel1')
        ax[3].imshow(hogimage1)
        ax[3].set_title('Image channel1 with gradients')
        ax[4].plot(features1)
        ax[4].set_title('Features channel1')
        ax[5].imshow(hogimage2)
        ax[5].set_title('Image channel2 with gradients')
        ax[6].plot(features2)
        ax[6].set_title('Features channel2')
        plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))


if __name__ == '__main__':
    main()
