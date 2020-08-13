#!/usr/bin/python3

import matplotlib.pyplot as plt
import car_detect_pkg as cp
import matplotlib.image as mpimg
import os
import glob
from moviepy.editor import VideoFileClip

#%matplotlib inline

# load images
cars = cp.load_images('./vehicles/')
print('Number of vehicle images found: {}'.format(len(cars)))

notcars = cp.load_images('./non-vehicles/')
print('Number of non-vehicle images found: {}'.format(len(notcars)))

searchpath = 'CarND-Vehicle-Detection/test_images/*'
example_images = glob.glob(searchpath)
print('Number of example images: {}'.format(len(example_images)))

# show one car
car_img = mpimg.imread(cars[1010])
#plt.imshow(car_img)
#plt.show()

not_car_img = mpimg.imread(notcars[101])
# plt.imshow(not_car_img)
# plt.show()

# show features
car_feature_img = cp.show_features(car_img)
plt.imshow(car_feature_img)
#plt.show()

not_car_feature_img = cp.show_features(not_car_img)
plt.imshow(not_car_feature_img)
#plt.show()


# train model
#cp.run_train_model(cars, notcars);

if not os.path.exists('./classifier.pkl'):
    model, scaler = cp.train_model(cars[0], notcars,
                                   color_space='YCrCb',
                                   spatial_size=(16, 16), hist_bins=32,
                                   orient=12, 
                                   pix_per_cell=8, cell_per_block=2,
                                   hog_channel='ALL', spatial_feat=True, 
                                   hist_feat=True, hog_feat=True, n_samples=0)
    cp.save_model(model, scaler)
    #print("no file")
else:
    model, scaler = cp.load_model()

# test model
is_car = cp.test_model(car_img, model, scaler)
print(is_car)

is_car = cp.test_model(not_car_img, model, scaler)
print(is_car)

# show boxes and detected cars
# cp.test_slide_window(example_images, model, scaler, xy_window=(128, 128), xy_overlap=(0.5, 0.5))

# cp.test_slide_window(example_images, model, scaler, xy_window=(64, 64), xy_overlap=(0.5, 0.5))

# video test
# Create an instance of the videoProcessor, to be used 
# in the process_image function
video_processor = cp.VideoProcessor(model, scaler)

clip = VideoFileClip("./CarND-Vehicle-Detection/project_video.mp4")
output_video = "./project_video_out.mp4"
output_clip = clip.fl_image(lambda image: cp.process_image(image,video_processor))
output_clip.write_videofile(output_video, audio=False)
