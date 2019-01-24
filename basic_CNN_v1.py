#######################
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer
#label_dict={}
#
#label_dict[2]=0
#
#label_dict[3]=1
#
#label_dict[6]=2
#
#label_dict[11]=3
#
#label_dict[19]=4


basepath='E:/MACHINE_LEARNING_PROJECTS/ML_PROJECT_4/train/'

def load_image(basepath, image_id):
    images = np.zeros(shape=(512,512,4))
    images[:,:,0] = plt.imread(basepath + image_id + "_green" + ".png")
    images[:,:,1] = plt.imread(basepath + image_id + "_red" + ".png")
    images[:,:,2] = plt.imread(basepath + image_id + "_blue" + ".png")
    images[:,:,3] = plt.imread(basepath + image_id + "_yellow" + ".png")
    return images


chunk_size=1000
chunks=[]
for chunk in pd.read_csv('train.csv', chunksize=chunk_size):
    chunks.append(chunk)
    
df=pd.concat(chunks,axis=0)
training_all=df.values

train_dict={}

image_list=[]
image_id_list=[]
label_list=[]

count=0
bount=0
#label_of_interest={11,2,19,3,6}

for i in range(0,len(training_all)):
    
    image_id=training_all[i][0]
    count +=1    
    label_str=training_all[i][1]
    cur_label_list=label_str.split()

    cur_label_list=tuple([int(ff) for ff in cur_label_list])
    label_list.append(cur_label_list)
    image_id_list.append(image_id)
        


mlb = MultiLabelBinarizer()
mlb.fit(label_list)
Y_label=mlb.transform(label_list)


num_classes=28
batch_size=200

model = Sequential()
model.add(Conv2D(32, kernel_size=(7, 7),strides=(2, 2),
                 activation='relu',
                 input_shape=(512,512,4)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='sigmoid'))

model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])



batch_size=100

for i in range(0,int(len(label_list)/batch_size)):
    
    print(i)
    print('\n')
    end_ind=min((i*batch_size+batch_size),len(label_list))
#    index=np.arange(i*batch_size,end_ind)
#    
    y_batch= Y_label[i*batch_size:end_ind,:]

    #y_batch= y_train.toarray()
    
    b_image_list=image_id_list[i*batch_size:end_ind]
    
    bb_image=[]
    for i_name in b_image_list:
        cur_image=load_image(basepath,i_name)    
        bb_image.append(cur_image)
        
    x_batch=np.array(bb_image)
    
    
    X_train, X_test, y_train, y_test = train_test_split\
    (x_batch, y_batch, test_size=0.25, random_state=42)
    
    model.train_on_batch(X_train, y_train)
    
    score = model.evaluate(X_test, y_test)
    
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


    
rr=model.predict(cur_image.reshape(-1,512,512,4))
