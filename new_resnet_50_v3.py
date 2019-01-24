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
import tensorflow as tf
num_classes=28

from keras import backend as K



## read image data

def load_image(basepath, image_id):
    images = np.zeros(shape=(512,512,3))
    images[:,:,0] = plt.imread(basepath + image_id + "_green" + ".png").astype(np.float32)/255
    images[:,:,1] = plt.imread(basepath + image_id + "_red" + ".png").astype(np.float32)/255
    images[:,:,2] = plt.imread(basepath + image_id + "_blue" + ".png").astype(np.float32)/255
    
    newimg = cv2.resize(images,(128,128))
    return newimg


## read csv file
chunk_size=1000
chunks=[]
for chunk in pd.read_csv('train.csv', chunksize=chunk_size):
    chunks.append(chunk)
    
df=pd.concat(chunks,axis=0)


training_all=df.values
basepath='E:/MACHINE_LEARNING_PROJECTS/ML_PROJECT_4/train/'

## create 

image_id_list=[]
label_list=[]
label_dict={}


for i in range(0,len(training_all)):
    
    image_id=training_all[i][0]
   
    
    label_str=training_all[i][1]
    cur_label_list=label_str.split()
    cur_label_list=tuple([int(ff) for ff in cur_label_list])
    
    label_list.append(cur_label_list)
    image_id_list.append(image_id)
    label_dict[image_id]=cur_label_list
        

mlb = MultiLabelBinarizer()
mlb.fit(label_list)

X_train_name, X_test_name= train_test_split\
    (image_id_list,test_size=0.1)
    


### data_generator

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=4, dim=(128,128), n_channels=3,
                 n_classes=28, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y=[]

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = load_image(basepath,ID)
            
            print
            # Store class
            y.append(self.labels[ID])
            
        
        y_categorial=mlb.transform(y)
        return X, y_categorial


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1))-K.sum((1-alpha) * K.pow( pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

#def focal_loss(labels, logits, gamma=2.0, alpha=4.0):
#    """
#    focal loss for multi-classification
#    FL(p_t)=-alpha(1-p_t)^{gamma}ln(p_t)
#    Notice: logits is probability after softmax
#    gradient is d(Fl)/d(p_t) not d(Fl)/d(x) as described in paper
#    d(Fl)/d(p_t) * [p_t(1-p_t)] = d(Fl)/d(x)
#    Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017).
#    Focal Loss for Dense Object Detection, 130(4), 485–491.
#    https://doi.org/10.1016/j.ajodo.2005.02.022
#    :param labels: ground truth labels, shape of [batch_size]
#    :param logits: model's output, shape of [batch_size, num_cls]
#    :param gamma:
#    :param alpha:
#    :return: shape of [batch_size]
#    """
#    epsilon = 1.e-9
#    labels = tf.to_int64(labels)
#    labels = tf.convert_to_tensor(labels, tf.int64)
#    logits = tf.convert_to_tensor(logits, tf.float32)
#    num_cls = logits.shape[1]
#
#    model_out = tf.add(logits, epsilon)
#    onehot_labels = tf.one_hot(labels, num_cls)
#    ce = tf.multiply(onehot_labels, -tf.log(model_out))
#    weight = tf.multiply(onehot_labels, tf.pow(tf.subtract(1., model_out), gamma))
#    fl = tf.multiply(alpha, tf.multiply(weight, ce))
#    reduced_fl = tf.reduce_max(fl, axis=1)
#    # reduced_fl = tf.reduce_sum(fl, axis=1)  # same as reduce_max
#    return reduced_fl

def acc(preds,targs,th=0.0):
    preds = np.array((preds > th))
    targs = np.array(targs)
    oo1=(preds==targs)
    oo=oo1.astype(int)
    return oo.mean()
    
#def f1(y_true, y_pred):
#    def recall(y_true, y_pred):
#        """Recall metric.
#
#        Only computes a batch-wise average of recall.
#
#        Computes the recall, a metric for multi-label classification of
#        how many relevant items are selected.
#        """
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#        recall = true_positives / (possible_positives + K.epsilon())
#        return recall
#
#    def precision(y_true, y_pred):
#        """Precision metric.
#
#        Only computes a batch-wise average of precision.
#
#        Computes the precision, a metric for multi-label classification of
#        how many selected items are relevant.
#        """
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#        precision = true_positives / (predicted_positives + K.epsilon())
#        return precision
#    precision = precision(y_true, y_pred)
#    recall = recall(y_true, y_pred)
#    return 2*((precision*recall)/(precision+recall+K.epsilon()))

model=keras.applications.resnet50.ResNet50(include_top=False, \
                                     weights='imagenet')


#Create your own input format (here 3x200x200)
input = Input(shape=(128,128,3),name = 'image_input')

#Use the generated model 
output_vgg16_conv = model(input)

#Add the fully-connected layers 
x = Flatten(name='flatten')(output_vgg16_conv)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dropout(0.3, name='d1')(x)
x = Dense(2048, activation='relu', name='fc2')(x)
x = Dropout(0.3, name='d2')(x)
x = Dense(num_classes, activation='sigmoid', name='predictions')(x)

#Create your own model 
my_model = Model(input=input, output=x)

#In the summary, weights and layers from VGG part will be hidden, but they will be fit during the training
#my_model.summary()

my_model.compile(loss=focal_loss(alpha=.25, gamma=2),
              optimizer=keras.optimizers.Adam(lr=0.001),
              metrics=[acc])

# Parameters
params = {'dim': (128,128),
          'batch_size': 4,
          'n_classes': 28,
          'n_channels': 3,
          'shuffle': True}


# Generators
training_generator = DataGenerator(X_train_name, label_dict, **params)
validation_generator = DataGenerator(X_test_name, label_dict, **params)


# Train model on dataset
my_model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    verbose=1)