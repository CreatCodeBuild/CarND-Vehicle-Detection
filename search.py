from feature import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label
from classifier import *
import os


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


def heat_map(image, box_list):
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat, box_list)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 1)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)
    return draw_img, heatmap



# Define a single function that can extract features using hog sub-sampling and make predictions

class CarFinder:

    def __init__(self):
        self.ystart = 400
        self.ystop = 656

    def find_cars(self, img, draw_img, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):

        ystart = self.ystart
        ystop = self.ystop
        # scale = self.scale
        spatial_size = (spatial_size, spatial_size)

        img = img.astype(np.float32) / 255

        img_tosearch = img[ystart:ystop, :, :]
        ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
        if scale != 1:
            imshape = ctrans_tosearch.shape
            ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

        ch1 = ctrans_tosearch[:, :, 0]
        ch2 = ctrans_tosearch[:, :, 1]
        ch3 = ctrans_tosearch[:, :, 2]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) - 1
        nyblocks = (ch1.shape[0] // pix_per_cell) - 1
        nfeat_per_block = orient * cell_per_block ** 2
        # 64 was the original sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)

        _list = []
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
                spatial_features = bin_spatial(subimg, size=spatial_size)
                hist_features = color_hist(subimg, nbins=hist_bins)
                # color_features = np.concatenate((spatial_features, hist_features))

                # print(color_features.shape, hog_features.shape)
                reshaped = np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
                # print(reshaped.shape)

                # Scale features and make a prediction
                test_features = X_scaler.transform(reshaped)
                # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
                test_prediction = svc.predict(test_features)
                # print(test_prediction)

                if test_prediction == 1:
                    # print(test_prediction)
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    cv2.rectangle(draw_img, (xbox_left, ytop_draw + ystart),
                                  (xbox_left + win_draw, ytop_draw + win_draw + ystart), (0, 0, 255), 6)
                    _list.append(((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))

        return draw_img, _list


if __name__ == "__main__":
    with open('classifier.p', 'rb') as f:
        classifier = pickle.load(f)
    finder = CarFinder()
    find_cars = finder.find_cars

    for path in glob.glob('test_images/*.jpg'):
        img = mpimg.imread(path)
        draw_img = np.copy(img)
        box_list = []
        out_img, _list = find_cars(img, draw_img, 1, classifier.svc, classifier.X_scaler, classifier.orient,
                            classifier.pix_per_cell, classifier.cell_per_block, classifier.spatial, classifier.histbin)
        box_list = box_list + _list

        out_img, _list = find_cars(img, draw_img, 1.5, classifier.svc, classifier.X_scaler, classifier.orient,
                            classifier.pix_per_cell, classifier.cell_per_block, classifier.spatial, classifier.histbin)
        box_list = box_list + _list

        out_img, _list = find_cars(img, draw_img, 2, classifier.svc, classifier.X_scaler, classifier.orient,
                            classifier.pix_per_cell, classifier.cell_per_block, classifier.spatial, classifier.histbin)
        box_list = box_list + _list

        mpimg.imsave("output_images/"+os.path.basename(path), out_img)

        draw_img, heatmap = heat_map(img, box_list)
        mpimg.imsave("output_images/" + "draw_" + os.path.basename(path), draw_img)
        mpimg.imsave("output_images/" + "heatmap_" + os.path.basename(path), heatmap)
