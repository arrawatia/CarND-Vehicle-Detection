import glob
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label

# load a pe-trained svc model from a serialized (pickle) file
import features

dist_pickle = pickle.load(open("svc.sav", "rb"))
print(dist_pickle)
# dist_pickle = pickle.load(open("svc_pickle.p", "rb"))
print(dist_pickle)
# get attributes of our svc object
svc = dist_pickle["svc"]
X_scaler = dist_pickle["scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]

img = mpimg.imread("./test_images/test1.jpg")


def read_image(file):
    return mpimg.imread(file)


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    boxes = []

    # img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = features.convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    ch1 = ctrans_tosearch[:, :, 0]
    ch2 = ctrans_tosearch[:, :, 1]
    ch3 = ctrans_tosearch[:, :, 2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
    nfeat_per_block = orient * cell_per_block ** 2

    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1

    # Compute individual channel HOG features for the entire image
    hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = features.bin_spatial(subimg, size=spatial_size)
            hist_features = features.color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            # print(nxsteps, nysteps, test_prediction)
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                box = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                boxes.append(box)

    return boxes


def remove_false_positives_and_multiple_detections(img, boxes):
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    heat_img = draw_labeled_bboxes(np.copy(img), labels)
    return heat_img, heatmap


ystart = 400
ystop = 656
scale = 1.5


# boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
#                     hist_bins)

# draw_img = np.copy(img)
# for box in boxes:
#     cv2.rectangle(draw_img, box[0],box[1], (0, 0, 255), 6)
# plt.imshow(draw_img)


# heat_img, heatmap = remove_false_positives_and_multiple_detections(img, boxes)

# fig = plt.figure()
# plt.subplot(121)
# plt.imshow(heat_img)
# plt.title('Car Positions')
# plt.subplot(122)
# plt.imshow(heatmap, cmap='hot')
# plt.title('Heat Map')
# fig.tight_layout()

def process_image(img):
    boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                      hist_bins)
    heat_img, _ = remove_false_positives_and_multiple_detections(img, boxes)

    return heat_img


import os

# * Apply a distortion correction to raw images.
lane_images = glob.glob('test_images/*.jpg')

output_path = "output_images"
if not os.path.exists(output_path):
    os.makedirs(output_path)

output_path_plots = "output_images/plots"
if not os.path.exists(output_path_plots):
    os.makedirs(output_path_plots)

for lane_image in lane_images:
    print(image)
    image = read_image(lane_image)

    boxes = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,
                      hist_bins)

    plt.imsave("%s/%s" % (output_path, os.path.basename(lane_image)), process_image(image))

    # Plot the result
    draw_img = np.copy(image)
    for box in boxes:
        cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)

    heat_img, heatmap = remove_false_positives_and_multiple_detections(image, boxes)

    f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image')
    ax2.imshow(draw_img)
    ax2.set_title('All detections')
    ax3.imshow(heat_img)
    ax3.set_title('Car Positions')
    ax4.imshow(heatmap, cmap='hot')
    ax4.set_title('Heat Map')
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)

    plt.savefig("%s/%s" % (output_path_plots, os.path.basename(lane_image)))

# from moviepy.editor import VideoFileClip
#
# output_path = "output_videos"
#
# if not os.path.exists(output_path):
#     os.makedirs(output_path)
# output = 'output_videos/test_video.mp4'
# clip = VideoFileClip("test_video.mp4")
# white_clip = clip.fl_image(process_image)
# white_clip.write_videofile(output, audio=False)
