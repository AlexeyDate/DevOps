from keras.models import load_model
from keras.utils import load_img
from keras.utils import image_utils
import numpy as np

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

img_width, img_height = 32, 32

classes = ["speed_limit_20", "speed_limit_30", "speed_limit_50"]        

model = load_model('GTSRB_CNN.h5')

img_path = 'speed_limit_50.ppm'

img = load_img(img_path, target_size=(img_width, img_height))

x = image_utils.img_to_array(img)
x = np.array([x])
x /= 255

prediction = model.predict(x)

prediction = np.argmax(prediction)
print("Name of sign >", classes[prediction])
