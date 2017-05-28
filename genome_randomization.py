import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LocallyConnected1D, LocallyConnected2D

from datetime import datetime, timezone


def main():
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])
    return model.layers

def layer_add(model, layer_name,
        node_range=[16,32,64,128,256], \
        dropout_range=[0.1,0.25,0.5], \
        pool_or_kernel_range_2D=[(2,2),(3,3),(4,4)], \
        pool_or_kernel_range_1D=[2,3,4,6], \
        activation_funcs=['relu']):

    '''
    Generate a layer from a given layer name.

    Supported layers:
        Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LocallyConnected2D

        TODO:
            MaxPooling1D, Conv1D, LocallyConnected1D,
    '''
    layer_name = layer_name.lower()
    node_size = random.choice(node_range)
    if layer_name == 'dense':
        model.add(Dense(node_size))
    elif layer_name == 'dropout':
        model.add(Dropout(random.choice(dropout_range)))
    elif layer_name == 'flatten':
        model.add(Flatten())
    # elif layer_name == 'conv1d':
    #     model.add(Conv1D(node_size, random.choice(pool_or_kernel_range_1D), activation=random.choice(activation_funcs)))
    elif layer_name == 'conv2d':
        model.add(Conv2D(node_size, random.choice(pool_or_kernel_range_2D), activation=random.choice(activation_funcs)))
    # elif layer_name == 'maxpooling1d':
    #     model.add(MaxPooling1D(random.choice(pool_or_kernel_range_1D)))
    elif layer_name == 'maxpooling2d':
        model.add(MaxPooling2D(random.choice(pool_or_kernel_range_2D)))
    elif layer_name == 'locallyconnected1d':
        model.add(LocallyConnected1D(node_size, random.choice(pool_or_kernel_range_1D)))
    elif layer_name == 'locallyconnected2d':
        model.add(LocallyConnected2D(node_size, random.choice(pool_or_kernel_range_2D)))
    else: # layer unrecognized and not added
        raise ValueError('Could not find "%s" in supported layers. \n\tError occurred at: %s' % \
            (layer_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')))

def add_from_list(model, layer_list, model_architecture_list):
    '''
    Recevies a list and adds the given layers to the model.
    '''
    for layer in layer_list:
        try:
            layer_add(model, layer)
            model_architecture_list.append(layer)
        except ValueError as e:
            pass

def generate_genome(dimensionality, min_depth=2, max_depth=7, net_must_start_with=[], net_must_end_with=[]):
    '''
    Generate basic genome from given dimension, parameters.

    TODO:
        add ability to specify layer-specific parameters on opening and closing, i.e. node_size
    '''
    # define basic model input
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(10,100,28)))
    # available functions with respect to input dimensionality
    if dimensionality == 2:
        available_funcs = ['dense','dropout','conv2d','maxpooling2d','locallyconnected2d']
    if dimensionality == 1:
        available_funcs = ['dense','dropout','conv1d','maxpooling1d','locallyconnected1d']
    # generate architecture
    model_architecture = []
    net_size = random.randint(min_depth, max_depth)
    # add must_start_with
    add_from_list(model, net_must_start_with, model_architecture)
    for _ in range(net_size):
        while True:
            layer = random.choice(available_funcs)
            try:
                layer_add(model, layer)
                model_architecture.append(layer)
                break
            except ValueError as e:
                pass
    # add must_end_with
    add_from_list(model, net_must_end_with, model_architecture)
    return(model, model_architecture)


# !

for i in range(1,10):
    g, m = generate_genome(2, net_must_end_with=['dense','flatten'])
    print(m)



# layer_add(model, 'dense')
# layer_add(model, 'locallyconnected2d')
#
# layer_add(model, 'conv2d')
# layer_add(model, 'maxpooling2d')
# layer_add(model, 'dense')
# layer_add(model,'flatten')
#
# layer_add(model, 'dense')
# print(model.layers)
# model.summary()
