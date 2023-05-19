import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

# data_dir = 'D:\Ricky\MSc\Puebla\_MCC Puebla Universidad Benemerita\Propuesta de Proyecto - Sistemas Inteligentes\Proyecto_ISI\dataset'

################# DATA LOADING ######################

# Data is a GENERATOR that will contain the images by batches as the corresponding ITERATOR is called
# By calling the next method of the iterator the batches will be retrieved
# batch[0] is the actual batch of images size = 32 by default
# batch[1] is an array containing all the labels in this case 10 digits + 26 letters = 36 labels in the array

### DATA PIPELINE (ALL DATA IS LOADED IN THIS GENERATOR (Check concept of a generator)
# data = tf.keras.utils.image_dataset_from_directory(data_dir)
# data = tf.keras.utils.image_dataset_from_directory(data_dir,batch_size = 32,image_size=(28,28),color_mode='grayscale')

# SCALING DATA (AS IT IS RETRIEVED) #
# scaled_data = data.map(lambda x,y: (x/255,y))
# scaled_data.as_numpy_iterator().next()

# #### SPLIT DATA ###
# train_size = int(len(data) * 0.7)  # 70 percent will be used for training, this number is the number of batches that will be used for training
# val_size = int(len(data) * 0.2)
# test_size = int(len(data) * 0.1)

# NOTE: In tensor flow as the model is being trained it is also being validated that's why a val_size is set
# that's why we use val_size

# # Number of images for each purpose (train, val and test)
# train = scaled_data.take(train_size)
# val = scaled_data.skip(train_size).take(val_size)
# test = scaled_data.skip(train_size+val_size).take(test_size)
# print(len(test))

#################### BUILDING THE MODEL ######################## THIS WORKED

# model = Sequential()
#
# model.add(Conv2D(32,(5,5),input_shape=(28,28,1),activation = 'relu'))
#
# model.add(Conv2D(32,(5,5),activation = 'relu'))
# model.add(MaxPooling2D(2,2))
#
# model.add(Flatten())
#
# model.add(Dense(256,activation = 'relu'))
# model.add(Dense(26,activation = 'softmax'))
#
# model.compile(loss = 'sparse_categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
# model.summary()
#
# logdir='logs'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
# hist = model.fit(train, epochs=5,  validation_data=val)
# model.save('cnn_char_recognition.model')

# fig = plt.figure()
# plt.plot(hist.history['loss'], color='teal', label='loss')
# plt.plot(hist.history['val_loss'], color='orange', label='val_loss')
# fig.suptitle('Loss', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()
#
# fig = plt.figure()
# plt.plot(hist.history['accuracy'], color='teal', label='accuracy')
# plt.plot(hist.history['val_accuracy'], color='orange', label='val_accuracy')
# fig.suptitle('Accuracy', fontsize=20)
# plt.legend(loc="upper left")
# plt.show()

# model.save('cnn_char_recognition.model')

# from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
#
# pre = Precision()
# re = Recall()
# acc = BinaryAccuracy()
#
# for batch in test.as_numpy_iterator():
#     X, y = batch
#     yhat = model.predict(X)
#     pre.update_state(y, yhat)
#     re.update_state(y, yhat)
#     acc.update_state(y, yhat)
#
# print(pre.result(), re.result(), acc.result())

################ MAKING PREDICTIONS ##################

letter_mapping = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
model = tf.keras.models.load_model('cnn_char_recognition.model')
testing = 'testing' # dir
img_lst = os.listdir(testing)
i = 0
while i < len(img_lst):
    try:
        img = cv2.imread(os.path.join(testing,img_lst[i]))[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f'Prediction: {letter_mapping[np.argmax(prediction)]} ---', f'Actual: {img_lst[i][0]}')
        plt.imshow(img[0],cmap=plt.cm.binary)
        plt.show()
    except:
        print('Error')
    finally:
        i+=1


