import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D
from keras.datasets import mnist
from keras import backend as K
import tensorflow as tf
from GAR import Genome
import datetime
import pandas as pd

# ---------------------------------------------------------------------------
# - test_genomes.py                                                         -
# -        this script is designed to test the functionality of GAR         -
# ---------------------------------------------------------------------------

# First, let's import some MNIST data
# code borrowed from Keras.examples.mnist

# data import stuff
batch_size = 128
num_classes = 10
epochs = 3
# input image dimensions
img_rows, img_cols = 28, 28
# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# define how many different genomes to generate
NUM_MODELS = 10
for iteration_num in range(NUM_MODELS):
    print('# ---------------------------------------------------------------------------')
    print('#')
    print('# - Beginning model # ' + str(iteration_num+1))
    print('#')
    print('# ---------------------------------------------------------------------------')


    # define relevant GAR variables
    net_must_start_with = [{'conv2d':{'filters':32, 'kernel_size':(3,3), 'activation':'relu',\
        'input_shape':input_shape}}]
    net_must_end_with = [{'dense':{'units':128}}, {'dropout':{}}, {'dense':{'units':num_classes, \
        'activation':'softmax'}}]
    max_depth = 7
    min_depth = 4
    
    # Here's GAR!
    gene = Gene(net_must_start_with=net_must_start_with, net_must_end_with=net_must_end_with, max_depth=max_depth, min_depth=min_depth)
    model, architecture = gene.build()


    # compiling the model is manual at the moment
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    print('# ---------------------------------------------------------------------------')
    print('# - Model # ' + str(iteration_num+1) + '; model size = ' + str(len(model.layers)))
    print('# ' + str(architecture))
    print('# ---------------------------------------------------------------------------')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    results = pd.read_csv('results.csv')
    # dump to CSV
    to_csv_df = pd.DataFrame({'test_loss':score[0], 'test_accuracy':score[1],\
        'architecture':[architecture],'timestamp':datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}, index=[0])
    pd.concat([to_csv_df,results], axis=0).to_csv('results.csv', index=False)
    # delete local objects, clear graph
    K.clear_session()
    # similar issue: https://github.com/fchollet/keras/issues/2397
    print('# ---------------------------------------------------------------------------')
    print('# - Completed model # ' + str(iteration_num+1))
    print('# ---------------------------------------------------------------------------')
    print('\n')
