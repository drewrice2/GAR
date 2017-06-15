import random
import numpy as np
from datetime import datetime
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LocallyConnected1D, LocallyConnected2D

# --------------------------------------------------
# -                                                -
# -  GAR.py                                        -
# -        contains the core functionality of GAR  -
# -                                                -
# --------------------------------------------------

# TODO: support more stuff
# TODO: support `self.model.compile`, `self.model.fit`

class Genome:

    def __init__(self, net_must_start_with, net_must_end_with, min_depth=4,
            max_depth=7, dimensionality=2):
        '''
        # Accepts relevant variables, type checks.
        '''
        self.net_must_start_with = net_must_start_with
        self.net_must_end_with = net_must_end_with
        self.dimensionality = dimensionality
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.net_depth = random.randint(min_depth, max_depth)
        self.model = Sequential() # only supporting Sequential models initially
        self.architecture = []
        # possibile criteria for randomization
        self.node_range = [16,32,64,128,256]
        self.conv_filter_range = [16,32,64,128,256]
        self.dropout_range = [0.1,0.25,0.5]
        self.pool_or_kernel_range_2D = [(2,2),(3,3),(4,4)]
        self.pool_or_kernel_range_1D = [2,3,4,6]
        self.activation_funcs = ['relu']

        def typecheck_and_error_handle():
            # !
            if self.dimensionality != 2:
                msg = "Only supporting 2D convolutional nets at this time. Change parameter `dimensionality`."
                raise ValueError(msg)
            # initial type checking
            if type(self.dimensionality) != int:
                msg = "Parameter `dimensionality` must be of type <class 'int'>. Found %s" % (type(self.dimensionality))
                raise TypeError(msg)
            if type(self.min_depth) != int:
                msg = "Parameter `min_depth` must be of type <class 'int'>. Found %s" % (type(self.min_depth))
                raise TypeError(msg)
            if type(self.max_depth) != int:
                msg = "Parameter `max_depth` must be of type <class 'int'>. Found %s" % (type(self.max_depth))
                raise TypeError(msg)
            if type(self.net_must_start_with) != list:
                msg = "Parameter `net_must_start_with` must be of type <class 'list'>. Found %s" % (type(self.net_must_start_with))
                raise TypeError(msg)
            if type(self.net_must_end_with) != list:
                msg = "Parameter `net_must_end_with` must be of type <class 'list'>. Found %s" % (type(self.net_must_end_with))
                raise TypeError(msg)
            if self.max_depth < self.min_depth:
                msg = "Parameter `max_depth` must be greater than or equal to `min_depth`."
                raise ValueError(msg)
            if (len(self.net_must_start_with) >= self.max_depth) or (len(self.net_must_end_with) >= self.max_depth):
                msg = "Net size too bit for max_depth. Check parameters: `max_depth`, `net_must_start_with`, `net_must_end_with`."
                raise ValueError(msg)
        # run typechecking
        typecheck_and_error_handle()

    def build(self):
        '''
        # Runs all network randomization and building.
        '''
        self.add_from_list(self.net_must_start_with)
        self.randomize_layers()
        self.add_from_list(self.net_must_end_with)
        return self.model, self.architecture

    def randomize_layers(self):
        '''
        # Randomize layers until `self.max_depth` is reached.
        '''
        # only supporting 2D nets at the moment, but ¯\_(ツ)_/¯
        if self.dimensionality == 2:
            # randomize number of convolutional layers
            num_conv_layers = int(self.net_depth * np.random.uniform()) # np.random.normal(loc=0.5,scale=0.1))
            # add convolutional layers to model
            for _ in range(num_conv_layers):
                layer = self.interpret_layer_dict({'conv2d':{}})
                self.add_layer_dict_to_model(layer)
                # random max pooling
                chance_of_max_pooling = np.random.uniform()
                if chance_of_max_pooling < 0.2:
                    max_pooling_layer = self.interpret_layer_dict({'maxpooling2d':{}})
                    self.add_layer_dict_to_model(max_pooling_layer)

            # add flatten layer
            layer = self.interpret_layer_dict({'flatten':{}})
            self.add_layer_dict_to_model(layer)

            # add dense layers
            num_fully_connected = self.net_depth - num_conv_layers
            for _ in range(num_fully_connected):
                layer = self.interpret_layer_dict({'dense':{}})
                self.add_layer_dict_to_model(layer)
                # random dropout
                chance_of_dropout = np.random.uniform()
                if chance_of_dropout < 0.2:
                    dropout_layer = self.interpret_layer_dict({'dropout':{}})
                    self.add_layer_dict_to_model(dropout_layer)

            # checks
            print(self.net_depth, self.max_depth, self.min_depth)
            print(num_conv_layers, num_fully_connected)

    def add_from_list(self, list_of_layers):
        '''
        # Input: list_of_layers to run through `self.interpret_layer_dict`.
            Called on `self.net_must_start_with`, `self.net_must_end_with`.

        # Raises: TypeError
                    if layer parameters is not a dictionary.
        '''
        for input_layer in list_of_layers:
            if type(input_layer) != dict:
                msg = "Parameter `input_layer` must be of type <class 'dict'>. Found %s" % (type(input_layer))
                raise TypeError(msg)
            else:
                layer = self.interpret_layer_dict(input_layer)
                self.add_layer_dict_to_model(layer)

    def add_layer_dict_to_model(self, layer_dictionary):
        '''
        # Receives a one-layer dictionary. Interprets and adds to `self.model`.

        # Input: `layer_dictionary` is a one-layer dictionary {`layer_name`:`params`}.
            Appends `layer_dictionary` to `self.architecture` list.

        # Raises: ValueError
                    if layer is not supported.
        '''
        layer = ''
        parameters = {}
        for key, value in layer_dictionary.items():
            layer = key
            parameters = value
        if layer == 'dense':
            self.model.add(Dense(**parameters))
        # Dropout
        elif layer == 'dropout':
            self.model.add(Dropout(**parameters))
        # Conv2D
        elif layer == 'conv2d':
            self.model.add(Conv2D(**parameters))
        # MaxPooling2D
        elif layer == 'maxpooling2d':
            self.model.add(MaxPooling2D(**parameters))
        # Flatten
        elif layer == 'flatten':
            self.model.add(Flatten())
        # Layer invalid
        else:
            msg = 'Could not find `%s` in supported layers.' % (layer_dictionary['layer_name'])
            raise ValueError(msg)
        # add layer specifications to `self.architecture` list
        self.architecture.append(layer_dictionary)

    def interpret_layer_dict(self, layer_dictionary):
        '''
        # Interprets a single-layer dictionary.

        # Returns: dictionary of `layer_name`:`parameters`
            where the `parameters` dict is fed directly into a Keras layer object

        # TODO: support Conv1D, LocallyConnected1D, LocallyConnected2D
        '''
        # dictionary for parameter pass
        keras_layer_parameters = {}

        for k, v in layer_dictionary.items():
            layer_name = k
            parameters = v

        # check for parameter 'input_shape'
        if len(self.model.layers) == 0:
            if 'input_shape' in parameters.keys():
                keras_layer_parameters['input_shape'] = parameters['input_shape']
            else:
                msg = "First model layer requires parameter `input_shape`."
                raise ValueError(msg)

        # Dense layer
        if layer_name == 'dense':
            if 'units' in parameters.keys():
                keras_layer_parameters['units'] = parameters['units']
            else:
                keras_layer_parameters['units'] = random.choice(self.node_range)
            if 'activation' in parameters.keys():
                keras_layer_parameters['activation'] = parameters['activation']
            else:
                keras_layer_parameters['activation'] = random.choice(self.activation_funcs)
        # Dropout
        elif layer_name == 'dropout':
            if 'rate' in parameters.keys():
                keras_layer_parameters['rate'] = parameters['rate']
            else:
                keras_layer_parameters['rate'] = random.choice(self.dropout_range)
        # Conv2D
        elif layer_name == 'conv2d':
            if 'filters' in parameters.keys():
                keras_layer_parameters['filters'] = parameters['filters']
            else:
                keras_layer_parameters['filters'] = random.choice(self.conv_filter_range)
            if 'kernel_size' in parameters.keys():
                keras_layer_parameters['kernel_size'] = parameters['kernel_size']
            else:
                keras_layer_parameters['kernel_size'] = random.choice(self.pool_or_kernel_range_2D)
            if 'activation' in parameters.keys():
                keras_layer_parameters['activation'] = parameters['activation']
            else:
                keras_layer_parameters['activation'] = random.choice(self.activation_funcs)
        # MaxPooling2D
        elif layer_name == 'maxpooling2d':
            if 'pool_size' in parameters.keys():
                keras_layer_parameters['pool_size'] = parameters['pool_size']
            else:
                keras_layer_parameters['pool_size'] = random.choice(self.pool_or_kernel_range_2D)
        # Flatten
        elif layer_name == 'flatten':
            pass
        # Layer invalid
        else:
            msg = 'Could not find `%s` in supported layers.' % (layer_name)
            raise ValueError(msg)

        return {layer_name:keras_layer_parameters}

    def clear_memory():
        '''
        # Delete any large variables from memory. TBD on what they are.
        '''
        pass

if __name__ == '__main__':
    x = Genome('b','b',dimensionality=2, min_depth=4,max_depth=4)

# def layer_add(layer_name,
#         node_range=[16,32,64,128,256], \
#         dropout_range=[0.1,0.25,0.5], \
#         pool_or_kernel_range_2D=[(2,2),(3,3),(4,4)], \
#         pool_or_kernel_range_1D=[2,3,4,6], \
#         activation_funcs=['relu']):
#     '''
#     # Generate a layer from a given layer name.
#
#     # Supported layers:
#         Dense, Dropout, Flatten, Conv2D, MaxPooling2D, LocallyConnected2D
#
#         Future layers:
#             MaxPooling1D, Conv1D, LocallyConnected1D, BatchNormalization
#
#     # Raises:
#         ValueError, if 'layer_name' is not recognized
#
#     # Returns: layer object.
#
#     # TODO! accept 'parameters' dictionary with 'layer_name','num_nodes','activation_func'
#     '''
#     # random selections done upfront
#     layer_name = layer_name.lower()
#     node_size = random.choice(node_range)
#     activation = random.choice(activation_funcs)
#     kernel_or_pool_size_2d = random.choice(pool_or_kernel_range_2D)
#     kernel_or_pool_size_1d = random.choice(pool_or_kernel_range_1D)
#
#     if layer_name == 'dense':
#         layer = Dense(units=node_size, activation=activation)
#     elif layer_name == 'dropout':
#         layer = Dropout(random.choice(dropout_range))
#     elif layer_name == 'flatten':
#         layer = Flatten()
#     elif layer_name == 'conv1d':
#         model.add(Conv1D(filters=node_size, kernel_size=kernel_or_pool_size_1d, activation=activation))
#     elif layer_name == 'conv2d':
#         layer = Conv2D(filters=node_size, kernel_size=kernel_or_pool_size_2d, activation=activation)
#     elif layer_name == 'maxpooling1d':
#         layer = MaxPooling1D(pool_size=pool_or_kernel_range_1D)
#     elif layer_name == 'maxpooling2d':
#         layer = MaxPooling2D(pool_size=pool_or_kernel_range_2D)
#     elif layer_name == 'locallyconnected1d':
#         layer = LocallyConnected1D(filters=node_size, kernel_size=pool_or_kernel_range_1D, activation=activation)
#     elif layer_name == 'locallyconnected2d':
#         layer = LocallyConnected2D(filters=node_size, kernel_size=pool_or_kernel_range_2D, activation=activation)
#     else: # layer unrecognized and not added
#         msg = 'Could not find "%s" in supported layers. \n\tError occurred at: %s' % \
#             (layer_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
#         raise ValueError(msg)
#     return layer
#
# def add_from_list(model, layer_list, model_architecture_list):
#     '''
#     # Recevies a list and adds the given layers to the model.
#
#     # Returns: model architecture list.
#     '''
#     for layer_name in layer_list:
#         try:
#             layer_to_add = layer_add(layer_name)
#             model_architecture_list.append(layer_name)
#             model.add(layer_to_add)
#         except ValueError as e:
#             pass
#     return model_architecture_list
#
# def generate_genome(model, dimensionality, min_depth=2, max_depth=7, net_must_start_with=[], net_must_end_with=[]):
#     '''
#     # Generate basic genome from given dimension, parameters.
#
#     # TODO:
#         add ability to specify layer-specific parameters on opening and closing, i.e. node_size
#
#     # Raises:
#         ValueError, if 'min_depth' and 'max_depth' are incorrectly sized
#         TypeError, if 'net_must_start_with' & 'net_must_end_with' are not lists
#     '''
#     # ERROR HANDLING !
#     # check depth args
#     if min_depth >= max_depth:
#         msg = 'Minimum depth variable "%i" needs to be bigger than max_depth variable "%i".\n\tError occurred at: %s' % \
#             (min_depth, max_depth, datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
#         raise ValueError(msg)
#     # check net_must_start_with & net_must_end_with data types
#     if type(net_must_start_with) != list:
#         msg = 'Argument "net_must_start_with" must be a list.\n\tError occurred at: %s' % \
#             (datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
#         raise TypeError(msg)
#     if type(net_must_end_with) != list:
#         msg = 'Argument "net_must_end_with" must be a list.\n\tError occurred at: %s' % \
#             (datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
#         raise TypeError(msg)
#
#     # check dimensionality, define universe of available functions
#     # TODO: infer from input dimensions
#     if dimensionality == 2:
#         available_funcs = ['conv2d','dense','dropout','maxpooling2d'] # ,'locallyconnected2d']
#     elif dimensionality == 1:
#         # available_funcs = ['dense','dropout','conv1d','maxpooling1d','locallyconnected1d']
#         # only supporting 2 dimensional data at this point
#         msg = 'Not supporting 1D...Only supporting 2 dimensional data at this point.'
#         raise ValueError(msg)
#     else:
#         msg = 'Dimensionality must be "1" or "2".\n\tError occurred at: %s' % \
#             (datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC'))
#         raise ValueError(msg)
#     # generate architecture
#     model_architecture = []
#     net_size = random.randint(min_depth, max_depth)
#     # add must_start_with
#     model_architecture = add_from_list(model, net_must_start_with, model_architecture)
#     for _ in range(net_size): # being done one layer at a time to only generate working nets
#         while True:
#             layer = np.random.choice(available_funcs, 1, p=[0.35,0.35,0.1,0.2])[0]
#             try:
#                 layer_to_add = layer_add(layer)
#                 model_architecture.append(layer)
#                 model.add(layer_to_add)
#                 break
#             except ValueError as e:
#                 pass
#     # add must_end_with
#     model_architecture = add_from_list(model, net_must_end_with, model_architecture)
#     return model_architecture
