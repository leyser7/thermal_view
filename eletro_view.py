print('load_dataset...')
from utils.elpv_reader import load_dataset
images, proba, types = load_dataset()

import matplotlib.pyplot as plt
import cv2
import numpy as np
# load model
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
print('load_model...')
model = load_model('last4_layers.h5')

# Train the model. Here we will be using the imageDataGenerator for data augmentation.

datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

# Data Generator for Training data
datagen.fit(images) 

def predict_one(model):  
	image_batch, classes_batch = next(datagen.flow(images, proba, batch_size=15, subset='training'))
	predicted_batch = model.predict(image_batch)
	for k in range(0,image_batch.shape[0]):
		image = image_batch[k]
		pred = predicted_batch[k]
		class_val = classes_batch[k]
		image = cv2.putText(image, 'predicted:'+str(round(pred[0],2)) , (10, 60) , cv2.FONT_HERSHEY_SIMPLEX ,  
			0.5, (255, 0, 0) , 2, cv2.LINE_AA)
		image = cv2.putText(image, 'real_value:'+str(round(class_val,2)) , (10, 30) , cv2.FONT_HERSHEY_SIMPLEX ,  
			0.5, (0, 0, 255) , 2, cv2.LINE_AA)
		#print(k)
		if k >= 0 and k < 5:
			if k == 0 :
				#print('here0')
				final_temp0 = image
			else :
				final_temp0 = cv2.hconcat([final_temp0, image])
		elif k >= 5  and k < 10:
			if k == 5 :
				final_temp1 = image
			else :
				final_temp1 = cv2.hconcat([final_temp1, image])
		elif k >= 10 and k < 15:
			if k == 10 :
				#print('here10')
				final_temp2 = image
			else :
				final_temp2 = cv2.hconcat([final_temp2, image])
		elif k >= 15 and k < 20:
			if k == 15 :
				#print('here15')
				final_temp3 = image
			else :
				final_temp3 = cv2.hconcat([final_temp3, image])
	final_temp0=cv2.vconcat([final_temp0, final_temp1])
	final_temp0=cv2.vconcat([final_temp0, final_temp2])
	final_temp0[np.where((final_temp0<[min(final_temp0[0][0]*0.8), 
		min(final_temp0[1][0]*0.8), min(final_temp0[2][0]*0.8)]).all(axis=2))] = [0,120,255]
	#final_temp0=cv2.vconcat([final_temp0, final_temp3])
	cv2.imshow("image", final_temp0)
	cv2.waitKey(1)
	
while 1 :
	predict_one(model)  
	