import glob
import numpy as np
import time
import pickle

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from feature import extract_hog_features, extract_color_features


class Classifier:
    def __init__(self):
        self.cars = glob.glob('./vehicles/*/*.png')
        self.notcars = glob.glob('./non-vehicles/*/*.png')

        # Set up parameters
        self.colorspace = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 9
        self.pix_per_cell = 8
        self.cell_per_block = 2
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial = 32
        self.histbin = 32
        self.svc = LinearSVC()
        self.X_scaler = None

        # Initialization the classifier
        t = time.time()
        car_hog_features = extract_hog_features(self.cars, cspace=self.colorspace, orient=self.orient,
                                                pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                                hog_channel=self.hog_channel)
        notcar_hog_features = extract_hog_features(self.notcars, cspace=self.colorspace, orient=self.orient,
                                                   pix_per_cell=self.pix_per_cell, cell_per_block=self.cell_per_block,
                                                   hog_channel=self.hog_channel)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to extract HOG features...')

        #
        t = time.time()
        car_color_features = extract_color_features(self.cars, cspace=self.colorspace,
                                                    spatial_size=(self.spatial, self.spatial),
                                                    hist_bins=self.histbin, hist_range=(0, 256))
        notcar_color_features = extract_color_features(self.notcars, cspace=self.colorspace,
                                                       spatial_size=(self.spatial, self.spatial),
                                                       hist_bins=self.histbin, hist_range=(0, 256))
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to extract color features...')

        #
        car_features = list(map(lambda x: np.concatenate(x), zip(car_color_features, car_hog_features)))
        notcar_features = list(map(lambda x: np.concatenate(x), zip(notcar_color_features, notcar_hog_features)))

        # Create an array stack of feature vectors
        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        self.X_scaler = StandardScaler().fit(X)  # Fit a per-column scaler
        scaled_X = self.X_scaler.transform(X)    # Apply the scaler to X

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell, 'pixels per cell and',
              self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))

        # Use a linear SVC
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')

        print(accuracy_score(self.svc.predict(X_test), y_test))


if __name__ == "__main__":
    with open("classifier.p", 'wb') as f:
        pickle.dump(Classifier(), f)
