# GAR
### Hey! You just found Genomic Architecture Randomization (GAR).

---

#### GAR is built on top of [Keras](https://github.com/fchollet/keras).

This project is in its very early stages at the moment and will be improved upon whenever I have time or receive PRs. Any contribution is greatly appreciated.

At this point, I have only tested GAR with:
  - Tensorflow-gpu 1.1.0
  - Keras 2.0.4

---
### The big idea.

Getting the best deep neural net architecture for any problem is not easy. But, with GPUs greatly accelerating the training of nets, random genome generation can assist researchers in figuring out some basics about the problem at hand.   

---

#### General workflow.

1. Initialize model with `model = Sequential()` and initial layer containing an `input_shape`.
2. For each model, generate layers up to a given depth.
3. Aggregate accuracy metrics and dump to `results.csv`.
