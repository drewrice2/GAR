# GAR
### Hey! You just found Genomic Architecture Randomization (GAR).

---
### The big idea.

Getting the best deep neural net architecture for any problem is not easy. But, with GPUs greatly accelerating the training of nets, random net architecture generation can assist researchers in quickly figuring out some basics about the problem at hand.

By abstracting layers to the following format: `{'layer_name': {'parameter_name': parameter}}`, GAR allows for a genome seed to be as customized or as randomized as a user specifies. The built-in "logging" functionality stores a GAR generated architecture, along with the performance on the train and test data. The architecture list can be dropped directly back into GAR for further randomization if the user desires.

Each Keras layer object has a number of parameters to set. GAR randomizes the parameters that you don't specify. For each layer, there exists a universe of randomization possibilities, and these are available as GAR object attributes.

---
### GAR is built on top of [Keras](https://github.com/fchollet/keras).

This project is in its very early stages at the moment and will be improved upon whenever I have time or receive PRs. Any contribution is greatly appreciated.

At this point, I have only tested GAR with:
  - Keras 2.0.4
  - Tensorflow-gpu 1.1.0, Windows 10, CUDA 8.0, cuDNN 5.1, Python 3.5.2
  - Tensorflow 1.1.0, MacOS, Python 2.7.X

---
### Existing workflow.

1. Create input layer(s).
2. Pass the `model` object to GAR, to generate _N_ models from the base model.
3. Aggregate accuracy metrics and dump to `results.csv`.

NOTE: some nets will be bigger than `max_input`, depending on input parameters and randomization.
