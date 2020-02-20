# imgclass
A collection of methods and classes to do image classification.

### Training
To train a model you will need to have a hyperparameters json and a hyperranges json. The hyperparameters json details the values of each of the training parameters that will be used for the training. See the [training_scripts readme](training_scripts/readme.md) for parameter details. The hyperranges json contains a subset of the hyperparameter keys each coupled to a list of values that will be cycled through for training. Every combination of the hyperranges key value pairs will be scheduled for training. This allows for easy hyperparameter searches. For example, if `lr` is the only key in the hyperranges json, then trainings for each listed value of the learning rate will be queued and processed in order. If `lr` and `l2` each are in the hyperranges json, then every combination of the `lr` and `l2` values will be queued for training.

To run a training session, navigate to the `training_scripts` folder:

```
$ cd training_scripts
```

And then select the cuda device index you will want to use (in this case 0) and type the following command:

```
$ CUDA_VISIBLE_DEVICES=0 python3 main.py path_to_hyperparameters.json path_to_hyperranges.json
```
### Current Models
- Tiny10: a model architecture used in [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414).
- Plain\(_8n-2_\): a model architecture used in [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414).

## Setup
Open terminal on your computer. Clone the repo and switch directories by typing the following:

```
$ git clone https://github.com/baccuslab/imgclass
$ cd imgclass
```

You will need to install the package using pip. If you are using anaconda, the following will likely work. If you are not using anaconda, try using `pip3` instead of `pip`.

```
$ pip install --user -e .
```

This should install imgclass and will allow you to do trainings!







