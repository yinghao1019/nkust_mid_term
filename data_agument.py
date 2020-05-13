from keras.preprocessing.image import ImageDataGenerator
data_auge=ImageDataGenerator(brightness_range=[0.1,0.5],rotation_range=20)
data=data_auge.flow_from_directory('./pic/train/category',target_size=(256,256),color_mode='grayscale',classes=['t'],save_prefix='1',save_to_dir='./pic/train/category',save_format='jpg')
for i in range(10):
    next(data)
