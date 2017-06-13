import random
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LocallyConnected1D, LocallyConnected2D
from datetime import datetime

# --------------------------------------------------
# -                                                -
# -  GAR.py                                        -
# -        contains the core functionality of GAR  -
# -                                                -
# --------------------------------------------------

def layer_add(layer_name,
        node_range=[16,32,64,128,256], \
        dropout_range=[0.1,0.25,0.5], \
        pool_or_kernel_range_2D=[(2,2),(3,3),(4,4)], \
        pool_or_kernel_range_1D=[2,3,4,6], \
        activation_funcs=['relu']):
    '''
    # Generate a layer from a given layer name.

    # Supported layers:
        Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LocallyConnected2D

        Future layers:
            MaxPooling1D, Conv1D, LocallyConnected1D, BatchNormalization

    # Raises:
        ValueError, if 'layer_name' is not recognized

    # Returns: layer object.

    # TODO! accept 'parameters' dictionary with 'layer_name','num_nodes','activation_func'
    '''
    # random selections done upfront
    layer_name = layer_name.lower()
    node_size = random.choice(node_range)
    activation = random.choice(activation_funcs)
    kernel_or_pool_size_2d = random.choice(pool_or_kernel_range_2D)
    kernel_or_pool_size_1d = random.choice(pool_or_kernel_range_1D)

    if layer_name == 'dense':
        layer = Dense(units=node_size, activation=activation)
    elif layer_name == 'dropout':
        layer = Dropout(random.choice(dropout_range))
    elif layer_name == 'flatten':
        layer = Flatten()
    elif layer_name == 'conv1d':
        model.add(Conv1D(filters=node_size, kernel_size=kernel_or_pool_size_1d, activation=activation))
    elif layer_name == 'conv2d':
        layer = Conv2D(filters=node_size, kernel_size=kernel_or_pool_size_2d, activation=activation)
    elif layer_name == 'maxpooling1d':
        layer = MaxPooling1D(pool_size=pool_or_kernel_range_1D)
    elif layer_name == 'maxpooling2d':
        layer = MaxPooling2D(pool_size=pool_or_kernel_range_2D)
    elif layer_name == 'locallyconnected1d':
        layer = LocallyConnected1D(filters=node_size, kernel_size=pool_or_kernel_range_1D, activation=activation)
    elif layer_name == 'locallyconnected2d':
        layer = LocallyConnected2D(filters=node_size, kernel_size=pool_or_kernel_range_2D, activation=activation)
    else: # layer unrecognized and not added
        msg = 'Could not find "%s" in supported layers. \n\tError occurred at: %s' % \
            (layer_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
        raise ValueError(msg)
    return layer

def add_from_list(model, layer_list, model_architecture_list):
    '''
    # Recevies a list and adds the given layers to the model.

    # Returns: model architecture list.
    '''
    for layer_name in layer_list:
        try:
            layer_to_add = layer_add(layer_name)
            model_architecture_list.append(layer_name)
            model.add(layer_to_add)
        except ValueError as e:
            pass
    return model_architecture_list

def generate_genome(model, dimensionality, min_depth=2, max_depth=7, net_must_start_with=[], net_must_end_with=[]):
    '''
    # Generate basic genome from given dimension, parameters.

    # TODO:
        add ability to specify layer-specific parameters on opening and closing, i.e. node_size

    # Raises:
        ValueError, if 'min_depth' and 'max_depth' are incorrectly sized
        TypeError, if 'net_must_start_with' & 'net_must_end_with' are not lists
    '''
    # ERROR HANDLING !
    # check depth args
    if min_depth >= max_depth:
        msg = 'Minimum depth variable "%i" needs to be bigger than max_depth variable "%i".\n\tError occurred at: %s' % \
            (min_depth, max_depth, datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
        raise ValueError(msg)
    # check net_must_start_with & net_must_end_with data types
    if type(net_must_start_with) != list:
        msg = 'Argument "net_must_start_with" must be a list.\n\tError occurred at: %s' % \
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
        raise TypeError(msg)
    if type(net_must_end_with) != list:
        msg = 'Argument "net_must_end_with" must be a list.\n\tError occurred at: %s' % \
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
        raise TypeError(msg)

    # check dimensionality, define universe of available functions
    # TODO: infer from input dimensions
    if dimensionality == 2:
        available_funcs = ['conv2d','dense','dropout','maxpooling2d'] # ,'locallyconnected2d']
    elif dimensionality == 1:
        # available_funcs = ['dense','dropout','conv1d','maxpooling1d','locallyconnected1d']
        # only supporting 2 dimensional data at this point
        msg = 'Not supporting 1D...Only supporting 2 dimensional data at this point.'
        raise ValueError(msg)
    else:
        msg = 'Dimensionality must be "1" or "2".\n\tError occurred at: %s' % \
            (datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
        raise ValueError(msg)
    # generate architecture
    model_architecture = []
    net_size = random.randint(min_depth, max_depth)
    # add must_start_with
    model_architecture = add_from_list(model, net_must_start_with, model_architecture)
    for _ in range(net_size): # being done one layer at a time to only generate working nets
        while True:
            layer = np.random.choice(available_funcs, 1, p=[0.35,0.35,0.1,0.2])[0]
            try:
                layer_to_add = layer_add(layer)
                model_architecture.append(layer)
                model.add(layer_to_add)
                break
            except ValueError as e:
                pass
    # add must_end_with
    model_architecture = add_from_list(model, net_must_end_with, model_architecture)
    return model_architecture
