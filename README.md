# GAR
### Hey! You just found Genomic Architecture Randomization (GAR).

---
### The big idea.

Getting the best deep neural net architecture for any problem is not easy. But, with GPUs greatly accelerating the training of nets, random net architecture generation can assist researchers in quickly figuring out some basics about the problem at hand.

By abstracting layers to the following format: ```python {'layer_name': {'parameter_name': parameter}}```, GAR allows for a genome seed to be as customized or as randomized as a user specifies. The built-in "logging" functionality stores a GAR Genome, or generated architecture, along with the performance on the train and test data in a CSV, `results.csv`. If a performance would like to be recreated, the architecture list from a particular row can be dropped directly back into `GAR.add_from_list()` for recreation, or as a parameter to GAR if the user wants further randomization.

Each Keras layer object has a number of parameters to set. GAR randomizes the parameters that you don't specify. For each layer, there exists a universe of randomization possibilities, and these are available as GAR object attributes. For example, `units_range = [16,32,64,128,256]`, is all of the possible `units` sizes for a `keras.layers.Dense` layer. Adjusting the universe of possibilities is as simple as `self.units_range = [256, 512, 1024]`.

---
### GAR is built on top of [Keras](https://github.com/fchollet/keras).

This project is in its very early stages at the moment and will be improved upon whenever I have time or receive PRs. Any contribution is greatly appreciated.

At this point, I have only tested GAR with:
  - Keras 2.0.4
  - Tensorflow-gpu 1.1.0, Windows 10, CUDA 8.0, cuDNN 5.1, Python 3.5.2
  - Tensorflow 1.1.0, MacOS, Python 2.7.X

---
### Existing workflow.

Look to the `test_genomes.py` script to check out GAR in action. A GAR Genome is randomized *N* times for the MNIST scipt found in `keras.examples.mnist_cnn`. The stored output of each genome is the trained model's layers and model performance.

~ 1. Define GAR Genome input parameters:

```python
net_must_start_with = [{'conv2d': {'filters':32, 'kernel_size': (3,3),
            'activation': 'relu', 'input_shape': input_shape}}]

net_must_end_with = [{'dense':{'units':128}}, {'dropout':{}},
            {'dense':{'units': num_classes, 'activation': 'softmax'}}]

max_depth = 7

min_depth = 4
```

~ 2. Instantiate a GAR Genome with the desired parameters:

```python
genome = Genome(net_must_start_with = net_must_start_with,
            net_must_end_with = net_must_end_with,
            max_depth = max_depth, min_depth = min_depth)
```
~ 3. Call `genome.build()`. Manually compile model and store the architecture. (In the future, GAR's scope will cover these operations.)
```python
model, architecture = genome.build()
# compiling the model is manual at the moment
model.compile(loss=keras.losses.categorical_crossentropy,
            optimizer=keras.optimizers.Adadelta(),
            metrics=['accuracy'])
model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(x_test, y_test))
```
~ 4. Aggregate accuracy metrics and dump to `results.csv`.

NOTE: some nets will be bigger than `max_input`, depending on input parameters and randomization.
