import glob
import os
import pickle

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label

import features


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
def find_cars(img, scaler, model, config):
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
    ystart = config["y_start"]
    ystop = config["y_stop"]
    scale = config["scale"]

    boxes = []

    # if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    if train_image_format == 'png':
        img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = features.convert_color(img_tosearch, conv=colorspace)

    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

    else:
        ch1 = ctrans_tosearch[:, :, hog_channel]

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
    if hog_channel == 'ALL':
        hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = features.get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = features.get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

    else:
        hog1 = features.get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)

    # import time
    #
    # start = time.time()
    # print("nxsteps=",nxsteps,"nysteps=",nysteps )
    for xb in range(nxsteps):
        for yb in range(nysteps):
            # start_i = time.time()
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            if hog_channel == 'ALL':
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            else:
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                hog_features = np.hstack((hog_feat1))

            # print("xb=",xb,"yb=",yb , "hog = ", time.time() - start_i)
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = features.bin_spatial(subimg, size=spatial_size)
            hist_features = features.color_hist(subimg, nbins=hist_bins)
            # print("xb=", xb, "yb=", yb, "spatial/hist = ", time.time() - start_i)
            # Scale features and make a prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
            # print("xb=", xb, "yb=", yb, "scale = ", time.time() - start_i)
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = model.predict(test_features)
            # print("xb=", xb, "yb=", yb, "pred = ", time.time() - start_i)
            # print(nxsteps, nysteps, test_prediction)
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)

                box = ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                boxes.append(box)
            # print("xb=", xb, "yb=", yb, "total = ", time.time() - start_i)
    # end = time.time()
    # print("pred + sliding window = ", end - start)
    return boxes


def remove_false_positives_and_multiple_detections(img, boxes, threshold=1):
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, threshold)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    heat_img = draw_labeled_bboxes(np.copy(img), labels)
    return heat_img, heatmap


def process_image(img, scaler, model, config):
    boxes = find_cars(img, scaler, model, config)
    heat_img, _ = remove_false_positives_and_multiple_detections(img, boxes, config["threshold"])

    return heat_img


def main():
    dist_pickle = pickle.load(open("svc.sav", "rb"))
    print(dist_pickle)

    svc = dist_pickle["svc"]
    scaler = dist_pickle["scaler"]
    config = dist_pickle["config"]

    img = mpimg.imread("./test_images/test1.jpg")

    config["y_start"] = 400  # Min and max in y to search in slide_window()
    config["y_stop"] = 656  # Min and max in y to search in slide_window()
    config["scale"] = 1.5
    config["threshold"] = 1

    # * Apply a distortion correction to raw images.
    lane_images = glob.glob('test_images/*.jpg')

    output_path = "output_images"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output_path_plots = "output_images/plots"
    if not os.path.exists(output_path_plots):
        os.makedirs(output_path_plots)

    for lane_image in lane_images:
        print(lane_image)
        image = read_image(lane_image)

        boxes = find_cars(img, scaler, svc, config)

        plt.imsave("%s/%s" % (output_path, os.path.basename(lane_image)), process_image(image, scaler, svc, config))

        # Plot the result
        draw_img = np.copy(image)
        for box in boxes:
            cv2.rectangle(draw_img, box[0], box[1], (0, 0, 255), 6)

        heat_img, heatmap = remove_false_positives_and_multiple_detections(image, boxes, config["threshold"])

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
