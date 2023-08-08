# Importing necessary functions for augmentation
import random
import keras.utils as image
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import *
#ImageDataGenerator,array_to_img,img_to_array, load_img
# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
rot = [45, 90, 135, 180, 225, 270,315]
index=0
datagen = ImageDataGenerator(
    rotation_range = rot[(index+1)%7], #random.randint(0,7)
    shear_range = 0.2,
    zoom_range = 0.1,
    horizontal_flip = True,
    brightness_range = (0.5, 1.5))

# Loading a sample image
for k in range(1,93):
    img = image.load_img(f'D:\\DRImp\\NewDataset\\N\image({k}).png')
    # Converting the input sample image to an array
    x = img_to_array(img)
    # Reshaping the input image
    x = x.reshape((1, ) + x.shape)


    # Generating and saving 7 augmented samples
    # using the above defined parameters.
    i = 0
    for batch in datagen.flow(x, batch_size = 1,
        save_to_dir ='agumented',
        save_prefix ='image', save_format ='jpg'):
        i += 1
        if i > 5:
            break
