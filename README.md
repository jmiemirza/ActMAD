# Installation

1) `git clone` this repository.
2) `pip install -r requirements.txt` to install required packages


# Preparing Datasets

## For KITTI dataset
* Download Clear (Original) [KITTI dataset](http://www.cvlibs.net/datasets/kitti/).
* Download [KITTI-Fog/Rain](https://team.inria.fr/rits/computer-vision/weather-augment/) datasets.
* Super-impose snow on KITTI dataset through this [repository](https://github.com/hendrycks/robustness).
* Generate labels YOLO can use (see [Dataset directory structures](#dataset-directory-structures) subsection).


## For ImageNet and CIFAR datasets
* Download the original train and test set for [ImageNet](https://image-net.org/download.php) & [ImageNet-C](https://zenodo.org/record/2235448#.Yn5OTrozZhE) datasets.
* Download the original train and test set for [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) & [CIFAR-10C](https://zenodo.org/record/2535967#.Yn5QwbozZhE) datasets.
* Generate _corrupted_ version of train sets through this [repository](https://github.com/hendrycks/robustness).

## Dataset directory structures
### For KITTI labels:
To generate labels YOLO can use from the original KITTI labels run

`python main.py --kitti_to_yolo_labels /path/to/original/kitti`

This is expecting the path to the original KITTI directory structure
```
path_to_specify
└── raw
    └── training
        ├── image_2
        └── label_2
```
Which will create a `yolo_style_labels` directory in the `raw` directory, containing
the KITTI labels in a format YOLO can use.

### For all datasets:
Structure the choosen dataset(s) as described [here](directory_scructures.md).

# Running Experiments

We recommend first setting up user specific paths in the `PATHS` dictionary in `config.py`
By following the existing entry as an example. This will lead to less commandline
arguments. Alternatively all paths can be passed explicitly as commandline
arguments instead.

Assuming paths have been added to the `PATHS` dictionary for a user `sl`, ActMad KITTI experiments
for the highest _weather severity_ can be ran like this:
```
python main.py --usr sl
```
By default only ActMad will be executed, in a manner that only evaluates and saves
the results and checkpoint after all batches have been ran, using the default learning
rate of `0.0001`. To evaluate after every batch and save the model state add `--actmad_save each_batch`
and to use a different learning rate add `--actmad_lr 0.00009` for example.

Which method(s) to run can be specified like this:
```
python main.py --usr sl --methods actmad dua
```
ActMAD KITTI experiments with the highest _weather severity_ with specifying paths in commandline arguments
can be done like this:
```
python main.py --dataroot /path/to/dataroot --ckpt_path /path/to/checkpoint.pt
```
To first train the model on the initial task, omit the `ckpt_path`:
```
python main.py --dataroot /path/to/dataroot
```
This will train the model on the initial task and save it at `.checkpoints/kitti/yolov3/initial/weights/best.pt`

By default all KITTI tasks (fog, rain, snow) are being executed. Which tasks to run
and their order can be changed using `--tasks`. To only run ActMAD for the 'fog' task:
```
python main.py --usr sl --tasks fog
```

Many more options, such as batch size, learning rate, workers, etc. are accesible through commandline arguments.
They can be seen in `main.py` or by running `python main.py -h`