#######################
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras.layers import Dense, Flatten, Dropout
from keras.models import Model
from keras.layers import Input
from sklearn.preprocessing import MultiLabelBinarizer
num_classes=28

from keras import backend as K

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))





model=keras.applications.resnet50.ResNet50(include_top=False,input_shape=(128,128,3), \
                                     weights='None',pooling='max')


#Create your own input format (here 3x200x200)
input = Input(shape=(128,128,3),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(4096, activation='relu', name='fc1')(x)
x = Dropout(0.3, name='d1')(x)
x = Dense(2048, activation='relu', name='fc2')(x)
x = Dropout(0.3, name='d2')(x)
x = Dense(num_classes, activation='sigmoid', name='predictions')(x)

#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
my_model.summary()

my_model.compile(loss=keras.losses.binary_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=[f1])



basepath='E:/MACHINE_LEARNING_PROJECTS/ML_PROJECT_4/train/'



def load_image(basepath, image_id):
    images = np.zeros(shape=(512,512,3))
    images[:,:,0] = plt.imread(basepath + image_id + "_green" + ".png")
    images[:,:,1] = plt.imread(basepath + image_id + "_red" + ".png")
    images[:,:,2] = plt.imread(basepath + image_id + "_blue" + ".png")
    
    newimg = cv2.resize(images,(128,128))
    return newimg


chunk_size=1000
chunks=[]
for chunk in pd.read_csv('train.csv', chunksize=chunk_size):
    chunks.append(chunk)
    
df=pd.concat(chunks,axis=0)
training_all=df.values
basepath='E:/MACHINE_LEARNING_PROJECTS/ML_PROJECT_4/train/'
image_list=[]
image_id_list=[]
label_list=[]

count=0
bount=0

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



X_train_name, X_test_name, y_train_label, y_test_label = train_test_split\
    (image_id_list, label_list, test_size=0.015)
    

train_label=mlb.transform(y_train_label)

y_test=mlb.transform(y_test_label)


bb_image=[]
for i_name in X_test_name:
    cur_image=load_image(basepath,i_name)    
    bb_image.append(cur_image)
    
x_test=np.array(bb_image)
    
basepath='E:/MACHINE_LEARNING_PROJECTS/ML_PROJECT_4/train/'
batch_size=16

iold=-1
for i in range(0,int(len(y_train_label)/batch_size)):
    
    print(i)

    end_ind=min((i*batch_size+batch_size),len(label_list))

    y_batch= train_label[i*batch_size:end_ind,:]

    b_image_list=X_train_name[i*batch_size:end_ind]
    
    bb_image=[]
    
    
    for i_name in b_image_list:
        
        print (i_name)
        cur_image=load_image(basepath,i_name)    
        bb_image.append(cur_image)
        
    x_batch=np.array(bb_image)
    
    
    
    my_model.train_on_batch(x_batch, y_batch)
    
    score = my_model.evaluate(x_test, y_test)
    
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])