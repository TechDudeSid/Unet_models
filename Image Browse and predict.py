import glob
import tkinter as tk
from tkinter import filedialog
from tkinter.filedialog import askopenfile
import os
from multi_unet_model import multi_unet_model #Uses softmax 
from keras.utils import normalize
import cv2
from skimage.io import imread,imshow
import numpy as np
from matplotlib import pyplot as plt




#Capture training image info as a list
train_images = []

#Resizing images, if needed
SIZE_X = 256
SIZE_Y = 256
n_classes=4 #Number of classes for segmentation

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=SIZE_X, IMG_WIDTH=SIZE_Y, IMG_CHANNELS=1)
model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.load_weights('MixedField_V2.hdf5')


f_types = [('Png Files', '*.png',),('Jpg Files', '*.jpg',)]
filenam = filedialog.askopenfilename(filetypes=f_types)
print(filenam)
test_img = cv2.imread(filenam,0) # X_test[test_img_number] cv2.imread(img_pat,0)
test_img =  cv2.resize(test_img, (SIZE_Y, SIZE_X))
tst_img = cv2.imread(filenam)
tst_img =  cv2.resize(test_img, (SIZE_Y, SIZE_X))

#test_img = np.array(test_img)
test_img = np.expand_dims(test_img, axis=2)
test_img = normalize(test_img, axis=1)
#ground_truth=y_test[0]
test_img_norm=test_img[:,:,0][:,:,None]
test_img_input=np.expand_dims(test_img_norm, 0)
prediction = (model.predict(test_img_input))
predicted_img=np.argmax(prediction, axis=3)[0,:,:]


plt.figure(figsize=(12, 8))
plt.subplot(121)
plt.title('Testing Image')
plt.imshow(tst_img)

plt.subplot(122)
plt.title('Prediction on test image')
plt.imshow(predicted_img, cmap='jet')
plt.show()

centroid = np.mean(np.argwhere(predicted_img),axis=0)
centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

plt.title(centroid_x)
plt.plot(centroid_x, 250, marker='x', color="black") 
plt.imshow(predicted_img) 

if (centroid_x - 128)>0:
    plt.xlabel("turn right")
else:
    plt.xlabel("turn left")  
plt.show() 