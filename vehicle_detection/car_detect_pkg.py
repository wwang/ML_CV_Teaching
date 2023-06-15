# car detection helper functions
import os
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import pickle
from collections import deque
from scipy.ndimage.measurements import label
import io
import base64
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
import subprocess
import random
import sys
from tqdm import tqdm
from moviepy.editor import VideoFileClip
from IPython.display import HTML


# load all images
def load_images(basedir):
    """
    Images are divided up into vehicles and non-vehicles folders, 
    each of which contains subfolders. Different folders represent 
    different sources for images (eg. GTI, kitti, generated by me,...).
    """
    image_types = os.listdir(basedir)
    res = []
    for imtype in image_types:
      res.extend(glob.glob(basedir + imtype + '/*'))
    return res
      
def visualize(fig, rows, cols, imgs, titles):
    """
    Function for plotting multiple images.
    """
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i + 1)
        plt.title(i + 1)
        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
        else:
            plt.imshow(img)
        plt.title(titles[i])
    #plt.show()

def convert_color(img, conv='RGB2YCrCb'):
    """
    Function to convert an image from a 
    space color to another one.
    """
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == 'HSV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    elif conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif conv == 'YUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)

def draw_boxes(img, bboxes, color=(0, 255, 0), thick=6):
    """
    Function to draw bounding boxes
      1. Make a copy of the image.
      2. Iterate through the bounding boxes and draw a 
         rectangle given bbox coordinates.
      3. Return the image copy with boxes drawn.
    """
    imcopy = np.copy(img)
    for bbox in bboxes:
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    return imcopy


def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                     vis=False, feature_vec=True):
    """
    Function to return HOG features and visualization.
    
    Call with two outputs if vis==True, otherwise call with one output
    
    The histogram of oriented gradients (HOG) is a feature 
    descriptor used in computer vision and image processing 
    for the purpose of object detection. The technique counts 
    occurrences of gradient orientation in localized portions 
    of an image. This method is similar to that of edge 
    orientation histograms, scale-invariant feature transform 
    descriptors, and shape contexts, but differs in that it is 
    computed on a dense grid of uniformly spaced cells and uses 
    overlapping local contrast normalization for improved accuracy.
    
    Source: https://en.wikipedia.org/wiki/Histogram_of_oriented_gradients
    """
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block,
                                                   cell_per_block),
                                  block_norm='L1', # added due to Python warning
                                  transform_sqrt=True, 
                                  visualize=vis, feature_vector=feature_vec)
        return features, hog_image
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L1', # added due to Python warning
                       transform_sqrt=True, 
                       visualize=vis, feature_vector=feature_vec)
        return features


def bin_spatial(img, size=(32, 32)):
    """
    Function to compute binned color features.
    It takes an image, a color space, and a new image size
    and returns a feature vector
    """
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):
    """
    Function to compute the color histogram features:    
     1. Compute the histogram of the RGB channels separately.
     2. Concatenate the histograms into a single feature vector.
     3. Return the individual histograms, bin_centers and feature vector.
    """
    channel1_hist = np.histogram(img[:,:,0], bins=nbins)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins)
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    return hist_features


def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True,
                        vis=False):    
    """
    Function to extract features from a single image window:
      1. Define an empty list to receive features
      2. Apply color conversion if other than 'RGB'
      3. Compute spatial features if flag is set
      4. Compute histogram features if flag is set    
      5. Compute HOG features if flag is set
      6. Return concatenated array of features    
    """
    img_features = []
    
    if color_space != 'RGB':
        feature_image = convert_color(img, color_space)
    else: 
        feature_image = np.copy(img)      

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        img_features.append(spatial_features)

    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        img_features.append(hist_features)

    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            if vis == True:
                hog_features, hog_image = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=True, feature_vec=True)
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        img_features.append(hog_features)

    if vis == True:
        return np.concatenate(img_features), hog_image
    else:
        return np.concatenate(img_features)
    
def test_single_img_features():
    """
    Test feature extraction from a single image:
      1. Choose random car & not-car indices.
      2. Read in car & notcar images.
      3. Define feature parameters.
      4. Extract the features for car & not-car.
      5. Visualize them.
    """
    #%matplotlib inline

    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(notcars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(notcars[notcar_ind])

    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0,1,2, or "ALL"
    spatial_size = (32, 32)  # spatial binning dimensions
    hist_bins = 16  # number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    car_features, car_hog_image = single_img_features(car_image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
    notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat, vis=True)

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['car image', 'car HOG image', 'notcar image', 'notcar HOG image']
    fig = plt.figure(figsize=(12, 3)) 
    visualize(fig, 1, 4, images, titles)


def show_features(image):
    
    color_space = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8  # HOG pixels per cell
    cell_per_block = 2  # HOG cells per block
    hog_channel = 0  # Can be 0,1,2, or "ALL"
    spatial_size = (32, 32)  # spatial binning dimensions
    hist_bins = 16  # number of histogram bins
    spatial_feat = True  # Spatial features on or off
    hist_feat = True  # Histogram features on or off
    hog_feat = True  # HOG features on or off

    features, hog_image = single_img_features(image, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat, vis=True)
    return hog_image

def extract_features(imgs, color_space='RGB', spatial_size=(32, 32), hist_bins=32, 
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    """
    Extract the features of a list of images.
    """
    features = []
    with tqdm(total=len(imgs), file=sys.stdout) as pbar:
      count=0
      for file in imgs:
          image = mpimg.imread(file)
          img_features = single_img_features(image, color_space, spatial_size,
                        hist_bins, orient, 
                        pix_per_cell, cell_per_block, hog_channel,
                        spatial_feat, hist_feat, hog_feat, vis=False)
            
          features.append(img_features)

          count += 1
          pbar.set_description('processed: %d' % (count))
          pbar.update(1)
    return features

def train_model(cars, notcars,
                color_space='YCrCb', spatial_size=(32, 32), hist_bins=32,
                orient=9, pix_per_cell=8, cell_per_block=2, hog_channel=0,
                spatial_feat=True, hist_feat=True, hog_feat=True,
                n_samples=0):
    """
    Function to train a SVC classifier from extracted features of images.
    
        1. Collects training data and extracts features using `extract_features`.
        2. Normalizes data using `sklearn.preprocessing.StandardScaler`.
        3. Splits data into train/test set using sklearn.model_selection.train_test_split, and
        4. Creates a LinearSVC model.
        5. train and computes accuracy on test set.
        6. Returns the model and the scaler.
    """
    t = time.time()
    if n_samples > 0:
        test_cars = np.array(cars)[np.random.randint(0, len(cars), n_samples)]
        test_notcars = np.array(notcars)[np.random.randint(0, len(notcars), n_samples)]
    else:
        test_cars = cars
        test_notcars = notcars
    print("Extracting car photo features...")
    car_features = extract_features(test_cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    print("Done")
    print("Extracting non-car photo features...")
    notcar_features = extract_features(test_notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    print("Done")
    t_features = time.time() - t
    
    print("Normalizing the features ... ", end='')
    # Normalize features
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)
    print("Done")

    print("Training the model ...", end='')
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2)

    # Use a linear SVC and train
    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)
    t_training = round(time.time() - t, 2)
    print("Done")
    
    # computes the accuracy
    accuracy = round(svc.score(X_test, y_test), 4)
    
    print('color:{}'.format(color_space), 'spatial_size:', spatial_size, 
          'orient:{}, hog:{}, hist:{}, feat:{}, time: {}s, acc: {}'.format(
            orient, hog_channel, hist_bins, 
            len(X_train[0]), 
            t_training, accuracy))

    return svc, X_scaler

def run_train_model(cars, notcars):
    print("Test training model...")
    for color_space in ['RGB', 'YCrCb']:
        for spatial_size in [(16, 16), (32, 32)]:
            for hist_bins in [16, 32]:
                for orient in [9, 12]:
                    for hog_channel in [0,'ALL']:
                        train_model(cars, notcars, color_space=color_space,
                                    spatial_size=spatial_size, 
                                    hist_bins=hist_bins, orient=orient, 
                                    pix_per_cell=8, cell_per_block=2,
                                    hog_channel=hog_channel,
                                    spatial_feat=True, hist_feat=True,
                                    hog_feat=True, n_samples=1000)

def save_model(model, scaler):
    """
    Pickles the model and save it to file system.
    """
    with open('./classifier.pkl', 'wb') as fp:
        data = {
            'model': model,
            'scaler': scaler
        }
        pickle.dump(data, fp)

def load_model():
    """
    Loads an existing trained model.
    """
    with open('./classifier.pkl', 'rb') as fp:
        data = pickle.load(fp)
    return data['model'], data['scaler']

def slide_window(img, x_start_stop=[None, None], y_start_stop=[400, 656], 
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """
    This function takes an image, start and stop positions in both x and y, 
    window size (x and y dimensions), and overlap fraction (for both x and y)
    """
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    
    # Compute the span of the region to be searched    
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]

    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    
    # Initialize a list to append window positions to
    window_list = []
    
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))

    return window_list

def search_windows(img, windows, clf, scaler, color_space='RGB', 
                    spatial_size=(32, 32), hist_bins=32, 
                    hist_range=(0, 256), orient=9, 
                    pix_per_cell=8, cell_per_block=2, 
                    hog_channel=0, spatial_feat=True, 
                    hist_feat=True, hog_feat=True):
    """
    Given an image and the list of windows to be 
    searched (output of slide_windows()),
    returns windows for positive detections:
      
    Foreach window in list:
        1. Extract the test window from original image
        2. Extract features for that window using single_img_features()
        3. Scale extracted features to be fed to classifier.
        4. Predict using your classifier.
        5. If positive (prediction == 1) then save the window.
    """
    on_windows = []
    off_windows = []
    for window in windows:        
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
        features = single_img_features(test_img, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        prediction = clf.predict(test_features)
        if prediction == 1:
            on_windows.append(window)
        else:
            off_windows.append(window)
            
    return on_windows, off_windows


def test_slide_window(example_images, model, scaler, xy_window=(128, 128), xy_overlap=(0.5, 0.5),
                      color_space='YCrCb', spatial_size=(16, 16), hist_bins=32, 
                      orient=12, pix_per_cell=8, cell_per_block=2, hog_channel='ALL', 
                      spatial_feat=True, hist_feat=True, hog_feat=True):

    print("Testing with window size:", xy_window)
    images = []
    titles = []

    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)
        draw_img = np.copy(img)

        img = img.astype(np.float32) / 255

        windows = slide_window(img, xy_window=xy_window, xy_overlap=xy_overlap)

        hot_windows, cold_windows = search_windows(img, windows, model, scaler,
                                                   color_space=color_space, 
                                                   spatial_size=spatial_size,
                                                   hist_bins=hist_bins, 
                                                   orient=orient,
                                                   pix_per_cell=pix_per_cell, 
                                                   cell_per_block=cell_per_block, 
                                                   hog_channel=hog_channel,
                                                   spatial_feat=spatial_feat, 
                                                   hist_feat=hist_feat,
                                                   hog_feat=hog_feat)                       


        window_img = draw_boxes(draw_img, cold_windows, color=(255,0, 0),
                                thick=6)
        window_img = draw_boxes(window_img, hot_windows, color=(0, 255, 0),
                                thick=6)
        #plt.imshow(window_img)
        #plt.show()
        images.append(window_img)
        titles.append('')
        # print(time.time() - t1, 'seconds to process one image searching', len(windows), 'windows')

    fig = plt.figure(figsize=(18, 18), dpi=300)
    visualize(fig, 5, 2, images, titles);
                                    
# test model on one image
def test_model(image, model, scaler):
    features = single_img_features(image, color_space='YCrCb',
                                   spatial_size=(16,16),
                                   hist_bins=32, orient=12, pix_per_cell=8,
                                   cell_per_block=2, hog_channel='ALL',
                                   spatial_feat=True, hist_feat=True,
                                   hog_feat=True, )

    test_features = scaler.transform(np.array(features).reshape(1, -1))

    prediction = model.predict(test_features)

    return prediction

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(image, labels):
    """
    Given an image and a list of detected cars, the
    function iterates through all detected cars:
      1. Find pixels with each car_number label value.
      2. Identify x and y values of those pixels.
      3. Define a bounding box based on min/max x and y.
      4. Draw the box on the image.
    Returns a copy of the image with bounding boxes.
    """
    img = np.copy(image)
    for car_number in range(1, labels[1]+1):
        nonzero = (labels[0] == car_number).nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        cv2.rectangle(img, bbox[0], bbox[1], (0,255,0), 6)
    return img

def find_cars(img, model, scaler, ystart, ystop, scale, spatial_size=(16, 16),
              hist_bins=32, orient=12, pix_per_cell=8, cell_per_block=2):
    """
    Function to detect cars in images. It uses approaches from 
    the previous functions and methodologies, and some improvements
    are introduced:
      1. Instead of using different window sizes we're resizing the whole image.
      2. Compute individual channel HOG features for the entire image, 
         once rather than computing for small portions of the image.
      3. It creates a heatmap image based on the prediction boxes.
    """
    draw_img = np.copy(img)
    heatmap = np.zeros_like(img[:,:,0])
    img = img.astype(np.float32) / 255
    img_boxes = []
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
    
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]
    
    nxblocks = (ch1.shape[1] // pix_per_cell) - 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - 1
    nfeat_per_block = orient * cell_per_block ** 2
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # Sliding window implementation
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step

            # extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
            
            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell
            
            #extract the image patch
            subimg = cv2.resize(
                ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], 
                (64, 64)
            )
            
            # get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)
            
            # scale features and make a prediction
            test_features = scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1,-1))
            test_prediction = model.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                cv2.rectangle(draw_img, 
                              (xbox_left, ytop_draw + ystart), 
                              (xbox_left + win_draw, ytop_draw + ystart + win_draw),
                              (0,255,0), 6)
                img_boxes.append((
                              (xbox_left, ytop_draw + ystart),
                              (xbox_left + win_draw, ytop_draw + ystart + win_draw)))
                heatmap[ytop_draw+ystart:ytop_draw+win_draw+ystart,
                       xbox_left:xbox_left+win_draw] += 1

    return draw_img, heatmap, img_boxes

class VideoProcessor():
    """
    Quick and dirty class toprocess each video frame.
    It uses other functions from the previous blocks of code
    in this notebook.
    """
    def __init__(self, model, scaler):
        self.frame_number = 0
        self._heatmaps = deque(maxlen=5)
        self.model = model
        self.scaler = scaler
        
    def process_image(self, img):
        """
        Find vehicles into it, given an undistorted image.
        """
        out_img, heatmap, _ = find_cars(img, self.model, self.scaler, 400,
                                        656, 1.5)
        self._heatmaps.append(heatmap)
        
        if len(self._heatmaps) < 5:
            return img

        # Calculate the average of the last heatmaps
        heatmaps = np.sum(np.array(self._heatmaps), axis=0).astype(np.float32)
        heatmaps = apply_threshold(heatmaps, 4)
        labels = label(heatmaps)
        draw_img = draw_labeled_bboxes(img, labels)
        
        return draw_img

def process_image(img, video_processor):
    """
    Function to process a frame of the video.
    It returns the image with overlayed boxes on detected cars.
    """
    draw_img = video_processor.process_image(img)
    return draw_img

def playvideo(filename):
    video = io.open(filename, 'r+b').read()
    encoded = base64.b64encode(video)
    return HTML(data='''<video alt="test" controls>
                    <source src="data:video/mp4;base64,{0}" type="video/mp4"/>
                 </video>'''.format(encoded.decode('ascii')))

def download_data():
  # download vehicle photos
  dw_cmd1 = ["wget", "https://tinyurl.com/y39psq9c", 
  "-O", "vehicles.zip"]
  output = subprocess.Popen(dw_cmd1, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)
  stdout,stderr=output.communicate()
  #print(stdout.decode())
  if not (stderr is None):
    print(stderr.decode())

  # unzip vehicle photos
  unzip_cmd1=["unzip", "vehicles.zip"]
  output = subprocess.Popen(unzip_cmd1, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)
  stdout,stderr=output.communicate()
  print("Vehicle files downloaded and unzipped")
  if not (stderr is None):
    print(stderr.decode())

  # download non-vehicle photos
  dw_cmd1 = ["wget", "https://tinyurl.com/y2ary7q7", 
  "-O", "non-vehicles.zip"]
  output = subprocess.Popen(dw_cmd1, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)
  stdout,stderr=output.communicate()
  #print(stdout.decode())
  if not (stderr is None):
    print(stderr.decode())

  # unzip non-vehicle photos
  unzip_cmd1=["unzip", "non-vehicles.zip"]
  output = subprocess.Popen(unzip_cmd1, stdout=subprocess.PIPE,
                   stderr=subprocess.STDOUT)
  stdout,stderr=output.communicate()
  print("Non-vehicle file downloaded and unzipped")
  if not (stderr is None):
    print(stderr.decode())

def load_training_images():
  # load vehicle images
  cars = load_images('./vehicles/')
  print('Number of vehicle images found: {}'.format(len(cars)))

  # load non vehicle images
  notcars = load_images('./non-vehicles/')
  print('Number of non-vehicle images found: {}'.format(len(notcars)))

  # load images for testing
  searchpath = 'ML_CV_Teaching/vehicle_detection/test_images/*'
  example_images = glob.glob(searchpath)
  print('Number of example images: {}'.format(len(example_images)))

  return cars, notcars, example_images

def show_random_car_image(cars):
  car_idx = random.randint(0,len(cars))
  car_img = mpimg.imread(cars[car_idx])
  car_feature_img = show_features(car_img)
  print("Showing one random car image:")
  plt.imshow(car_img)
  plt.show()
  plt.imshow(car_feature_img)

def show_random_notcar_image(notcars):
  # show the features of the above non-vehicle picture
  not_car_idx = random.randint(0,len(notcars))
  not_car_img = mpimg.imread(notcars[not_car_idx])
  not_car_feature_img = show_features(not_car_img)
  print("Showing one random not car image:")
  plt.imshow(not_car_img)
  plt.show()
  plt.imshow(not_car_feature_img)

def train_car_model(cars, notcars):
  model, scaler = train_model(cars, notcars,
                              color_space='YCrCb',
                              spatial_size=(16, 16), hist_bins=32,
                              orient=12, 
                              pix_per_cell=8, cell_per_block=2,
                              hog_channel='ALL', spatial_feat=True, 
                              hist_feat=True, 
                              hog_feat=True, 
                              n_samples=0)
  return model, scaler

def test_model_on_random_image(cars, notcars, model, scaler):
    if random.choice([True, False]):
        imgs = cars
    else:
        imgs = notcars
    
    car_idx = random.randint(0,len(imgs))
    car_img = mpimg.imread(imgs[car_idx])
    plt.imshow(car_img)
    is_car = test_model(car_img, model, scaler)
    if is_car:
        print("AI says this is a vehicle")
    else:
        print("AI says this is not a vehicle.")


def test_on_static_frames(model, scaler, example_images, xy_window=(128,128)):
    # with  128x128 sliding windows
    for i in range(len(example_images)):
        test_slide_window(example_images[i:i+1], model, scaler,
                          xy_window, xy_overlap=(0.5, 0.5))

def load_video_clip1():
    # short video
    clip1_path = "ML_CV_Teaching/vehicle_detection/test_video.mp4"
    clip = VideoFileClip(clip1_path)
    #playvideo("ML_CV_Teaching/vehicle_detection/test_video.mp4")

    return clip, clip1_path

def process_video_clip1(clip, model, scaler):
    video_processor = VideoProcessor(model, scaler)
    output_video = "./test_video_out.mp4"
    output_clip = clip.fl_image(lambda image: process_image(image,video_processor))
    output_clip.write_videofile(output_video, audio=False)

    #playvideo(output_video)

    return output_video

def load_video_clip2():
    # long video
    clip2_path = "./ML_CV_Teaching/vehicle_detection/project_video.mp4"
    clip = VideoFileClip("./ML_CV_Teaching/vehicle_detection/project_video.mp4")
    #playvideo("./ML_CV_Teaching/vehicle_detection/project_video.mp4")

    return clip, clip2_path
    
def process_video_clip2(clip, model, scaler):
    video_processor = VideoProcessor(model, scaler)
    output_video = "./project_video_out.mp4"
    output_clip = clip.fl_image(lambda image: process_image(image,video_processor))
    output_clip.write_videofile(output_video, audio=False)

    #playvideo(output_video)

    return (output_video)


def load_video_clip3():
    # long video
    clip2_path = "./ML_CV_Teaching/vehicle_detection/Shorter Video.mp4"
    clip = VideoFileClip(clip2_path)
    #playvideo("./ML_CV_Teaching/vehicle_detection/project_video.mp4")

    return clip, clip2_path
    
def process_video_clip3(clip, model, scaler):
    video_processor = VideoProcessor(model, scaler)
    output_video = "./shorter_video_out.mp4"
    output_clip = clip.fl_image(lambda image: process_image(image,video_processor))
    output_clip.write_videofile(output_video, audio=False)

    #playvideo(output_video)

    return (output_video)
    
