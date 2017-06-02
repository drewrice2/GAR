# GAR
### Hey! You just found Genomic Architecture Randomization (GAR).

---
### The big idea.

Getting the best deep neural net architecture for any problem is not easy. But, with GPUs greatly accelerating the training of nets, random genome generation can assist researchers in figuring out some basics about the problem at hand.

---
#### GAR is built on top of [Keras](https://github.com/fchollet/keras).

This project is in its very early stages at the moment and will be improved upon whenever I have time or receive PRs. Any contribution is greatly appreciated.

At this point, I have only tested GAR with:
  - Keras 2.0.4
  - Tensorflow-gpu 1.1.0, Windows 10, CUDA 8.0, cuDNN 5.1, Python 3.5.2
  - Tensorflow 1.1.0, MacOS, Python 2.7.X

---
#### Existing workflow.

1. Create input layer(s).
2. Pass the `model` object to GAR, to generate _N_ models from the base model.
3. Aggregate accuracy metrics and dump to `results.csv`.
