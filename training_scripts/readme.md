To train a model already defined in `imgclass/models.py`, you can simply run the following line at the terminal from the `training_scripts` folder:

    $ python3 main.py hyperparams.json hyperranges.json

The hyperparams.json should have a list of all the desired user setting for the training. The hyperranges.json should have the values wished to be searched over for each desired user setting key.

# Possible Hyperparameters
* `exp_name`: str
    * The name of the main experiment. Model folders will be saved within a folder of this name.
* `save_every_epoch`: bool
    * A boolean determining if the model `state_dict` should be saved for every epoch, or only the most recent epoch. Defaults to False, only the most recent epoch is saved.
* `n_repeats`: int
    * The number of times to repeat any given hyperparameter set
* `seed`: int or null
    * A manually set random seed value for both torch and numpy. If null, the random seed is set as the current value of `time.time()` and recorded in the hyperaparameters to ensure reproducibility.

* `model_type`: str
    * The string name of the main model class to be used for training. Options are each of the classes defined in `models.py`
* `width`: int or null
    * if argued, all layers in the model take on this channel count. If null, values default to the values listed in [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414)
* `plain_n`: int
    * the n value for the Plain(8n+2) architecture. Only applies for that architecture.
* `n_shakes`: int
    * the number of shake-shake modules to use for each convolutional filter. If 1, has no effect. See [Shake-Shake Regularization](https://arxiv.org/abs/1705.07485) for an explanation.
* `locrespnorm`: bool
    * if true, some model architectures use local response normalization where possible.
* `bnorm`: bool
    * if true, model uses batch normalization where possible
* `stackconvs`: bool
    * if true, convolutions are trained using linear convolution stacking
* `img_shape`: list of ints
    * the shape of the incoming stimulus to the model (do not include batchsize but do include depth of images) (C, H, W)
* `chans`: list of ints
    * the number of channels to be used in the intermediary layers
* `ksizes`: list of ints
    * the kernel sizes of the convolutions corresponding to each layer
* `drop_ps`: list of floats or null
    * the dropout probabilities for layers that use dropout. all default to 0 if null
* `paddings`: list of ints or null
    * the paddings of the convolutions corresponding to each layer. Defaults all to 0 if null.
* `strides`: list of ints or null
    * the strides of the convolutions corresponding to each layer. defaults all to 1 if null.
* `mid_dims`: int or null
    * the intermediary fully convolutional dimension sizes for select architectures.

* `batch_size`: int
    * the number of samples to used in a single step of SGD
* `n_epochs`: int
    * the number of complete training loops through the data
* `lr`: float
    * the learning rate
* `l2`: float
    * the l2 weight penalty
* `l1`: float
    * the l1 activation penalty applied on the final outputs of the model.
* `bn_moment`: float
    * the momentum of the batchnorm layers
* `scheduler`: str
    * the type of learning rate scheduler to be used during training.

* `dataset`: str
    * the name of the dataset to be used for training. code assumes the datasets are located in `~/experiments/data/`. The dataset should be a folder that contains h5 files.
* `lossfxn`: str
    * The name of the loss function that should be used for training the model. Currently options are "PoissonNLLLoss" and "MSELoss"
* `shuffle`: bool
    * boolean determining if the order of samples with in a batch should be shuffled. This does not shuffle the sequence itself.
* `val_p`: float
    * the portion of the data that should be used for validation
* `val_loc`: str
    * the location in the data that the validation set should be taken from. if shuffle is true, this argument has no statisitcal effect.
* `n_workers`: int
    * the number of workers to be used with pytorch's dataloader
