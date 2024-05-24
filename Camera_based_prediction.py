""" The objective of this code is to perform real-time image segmentation using a pre-trained U-Net model
    on frames captured from a webcam. It processes each frame by resizing and normalizing it, then uses the
    model to predict a segmentation map, identifying different classes within the frame. The centroid of the 
    segmented object is calculated to determine its position relative to the frame's center, guiding 
    directional decisions (left or right). This setup is intended for applications that require real-time 
    visual analysis and decision-making based on segmentation results, such as navigation or object tracking systems."""


from multi_unet_model import multi_unet_model
from keras.utils import normalize
import cv2
import numpy as np


SIZE_X = 256
SIZE_Y = 256
n_classes=4 #Number of classes for segmentation

IMG_HEIGHT= 256
IMG_WIDTH  = 256
IMG_CHANNELS = 1

def get_model():
    return multi_unet_model(n_classes=n_classes, IMG_HEIGHT=IMG_HEIGHT, IMG_WIDTH=IMG_WIDTH, IMG_CHANNELS=IMG_CHANNELS)
model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
         
model.load_weights('MixedField_V2.hdf5')

capture = cv2.VideoCapture(0)
while True:
    
    isTrue, frame = capture.read()
    if isTrue:
        cv2.imwrite('frame.jpg', frame)
        filenam = 'frame.jpg'

        test_img =  cv2.imread(filenam,0)
        test_img =  cv2.resize(test_img, (SIZE_Y, SIZE_X))
        tst_img = cv2.imread(filenam)
        tst_img =  cv2.resize(test_img, (SIZE_Y, SIZE_X))

        test_img = np.expand_dims(test_img, axis=2)
        test_img = normalize(test_img, axis=1)

        test_img_norm=test_img[:,:,0][:,:,None]
        test_img_input=np.expand_dims(test_img_norm, 0)
        prediction = (model.predict(test_img_input))
        predicted_img=np.argmax(prediction, axis=3)[0,:,:]

        centroid = np.mean(np.argwhere(predicted_img),axis=0)
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

        if (centroid_x - 128)>0:
            print("turn Right")
        else:
            print("turn Left")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
