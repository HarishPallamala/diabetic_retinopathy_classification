# Importing necessary functions for augmentation
import keras.utils as image
from tensorflow.keras.utils import img_to_array
from keras.preprocessing.image import * #ImageDataGenerator,array_to_img,img_to_array, load_img
# Initialising the ImageDataGenerator class.
# We will pass in the augmentation parameters in the constructor.
datagen = ImageDataGenerator(
    rotation_range = 60,
    shear_range = 0.2,
    zoom_range = 0,
    horizontal_flip = True,
    brightness_range = (0.5, 1.5))

# Loading a sample image
img = image.load_img('D:\\Diabetic Retinopathy\\diaretdb1_v_1_1\\resources\\images\\ddb1_fundusimages\\image001.png')
# Converting the input sample image to an array
x = img_to_array(img)
# Reshaping the input image
x = x.reshape((1, ) + x.shape)


# Generating and saving 6 augmented samples
# using the above defined parameters.
i = 0
for batch in datagen.flow(x, batch_size = 1,
    save_to_dir ='content/gdrive/MyDrive/aug2',
    save_prefix ='image', save_format ='png'):
    i += 1
    if i > 5:
        break
