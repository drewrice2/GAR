import random
import numpy as np
from datetime import datetime
from keras import backend as K
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

class Gene:

    def __init__(self, net_must_start_with, net_must_end_with, min_depth=4,
            max_depth=7, dimensionality=2):
        '''Accepts relevant variables, type checks.

        # Arguments
            net_must_start_with: list of dictionaries with shape: [{'layer_name':{'param_name':parameter}}]
            net_must_end_with: list of dictionaries. [{'layer_name':{'param_name':parameter}}]
            min_depth: minimum number of *core* layers for the net
            max_depth: maximum number of *core* layers for the net
            dimensionality: true input data dimensionality

        # Raises
            ValueError: if `dimensionality` != 2, or if `max_depth` < `min_depth`
            TypeError: if input vars are of incorrect type.
        '''
        self.backend = K.backend()
        self.net_must_start_with = net_must_start_with
        self.net_must_end_with = net_must_end_with
        self.dimensionality = dimensionality
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.net_depth = random.randint(min_depth, max_depth)
        self.model = Sequential() # only supporting Sequential models initially
        self.architecture = []
        # randomization universe
        self.units_range = [16,32,64,128,256]
        self.conv_filter_range = [16,32,64,128,256]
        self.dropout_range = [0.05,0.1,0.25,0.35]
        self.pool_or_kernel_range_2D = [(2,2),(3,3),(5,5)]
        self.pool_or_kernel_range_1D = [2,3,5]
        self.activation_funcs = ['relu']

        def _typecheck_and_error_handle():
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
        _typecheck_and_error_handle()

    def build(self):
        '''Runs all network randomization and building.

        # Returns
            model: an uncompiled Keras model object with GAR randomized layers
            architecture: list of dictionaries for calling GAR on, or for logging purposes
        '''
        self.add_from_list(self.net_must_start_with)
        self.randomize_layers()
        self.add_from_list(self.net_must_end_with)
        return self.model, self.architecture

    def randomize_layers(self):
        '''Randomize layers until `self.net_depth` is reached. Current workflow: randomized `num_conv_layers`,
            with randomized pooling, and randomized `num_fully_connected`, with randomized dropout.

            Only supporting 2D convolutional nets at the moment.
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

    def add_from_list(self, list_of_layers):
        '''Adds a given list of layers to the model.

        # Arguments
            list_of_layers: list, passing one at a time to `self.interpret_layer_dict`.

        # Raises
            TypeError: if layer parameters is not a dictionary.
        '''
        for input_layer in list_of_layers:
            if type(input_layer) != dict:
                msg = "Parameter `input_layer` must be of type <class 'dict'>. Found %s" % (type(input_layer))
                raise TypeError(msg)
            else:
                layer = self.interpret_layer_dict(input_layer)
                self.add_layer_dict_to_model(layer)

    def add_layer_dict_to_model(self, layer_dictionary):
        '''Receives a one-layer dictionary. Interprets and adds to `self.model`.

        # Arguments
            layer_dictionary: one-layer dictionary {`layer_name`:{`param_name`:`parameter`}}. Appends
                `layer_dictionary` to `self.architecture` list.

        # Raises
            ValueError: if layer is not supported.
        '''
        # interpret layer and parameter dictionary
        for k, v in layer_dictionary.items():
            layer = k
            parameters = v

        # Dense
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
        '''Interprets a single-layer dictionary. If correct parameters exist, pass them to
            `keras_layer_parameters`, else randomly generate from randomization universe.

        # Returns
            layer_to_keras: dictionary of shape {`layer_name`:{`param_name`: `parameter`}}
                where the `parameters` dict is fed directly into a Keras layer object

        # TODO: support Conv1D, LocallyConnected1D, LocallyConnected2D
        '''
        # dictionary for parameter pass
        keras_layer_parameters = {}

        # interpret layer and parameter dictionary
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

        # Dense
        if layer_name == 'dense':
            if 'units' in parameters.keys():
                keras_layer_parameters['units'] = parameters['units']
            else:
                keras_layer_parameters['units'] = random.choice(self.units_range)
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
            if self.backend == 'tensorflow':
                keras_layer_parameters['dim_ordering'] = 'tf'
            elif self.backend == 'theano':
                keras_layer_parameters['dim_ordering'] = 'th'
        # MaxPooling2D
        elif layer_name == 'maxpooling2d':
            if 'pool_size' in parameters.keys():
                keras_layer_parameters['pool_size'] = parameters['pool_size']
            else:
                keras_layer_parameters['pool_size'] = random.choice(self.pool_or_kernel_range_2D)
            if self.backend == 'tensorflow':
                keras_layer_parameters['dim_ordering'] = 'tf'
            elif self.backend == 'theano':
                keras_layer_parameters['dim_ordering'] = 'th'
        # Flatten
        elif layer_name == 'flatten':
            pass
        # Layer invalid
        else:
            msg = 'Could not find `%s` in supported layers.' % (layer_name)
            raise ValueError(msg)

        layer_to_keras = {layer_name: keras_layer_parameters}
        return layer_to_keras

    def clear_memory():
        '''Delete any large variables from memory. TBD on what they are yet.
        '''
        pass
