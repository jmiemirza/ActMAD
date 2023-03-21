# ActMAD: Activation Matching to Align Distributions for Test-Time Training (CVPR 2023)

This is the official repository for our CVPR 2023 paper: [ActMAD](https://arxiv.org/pdf/2211.12870.pdf)

ActMAD proposes a task- and architecture-agnostic Test-Time Training methodology which relies on aligning distributions
of the intermediate activation responses from the training data, and the out-of-distribution test data at test-time.
Our main contribution lies in proposing a location aware feature alignment approach where we model a distribution
over each pixel in an intermediate feature map.

Our experimental evaluation spans over 2 downstream tasks (image classification and object detection). In this repository
we provide code to reproduce CIFAR-10/100C and KITTI experiments from our main paper.
# Installation
1) `git clone` this repository.
2) `pip install -r requirements.txt` to install required packages

# Running Experiments
Prepare the datasets through the instructions listed
[here](utils/preparing_datasets.md) and also set the directory structure as described [here](utils/directory_scructures.md).


[comment]: <> (## For KITTI dataset)

[comment]: <> (* Download Clear &#40;Original&#41; [KITTI dataset]&#40;http://www.cvlibs.net/datasets/kitti/&#41;.)

[comment]: <> (* Download [KITTI-Fog/Rain]&#40;https://team.inria.fr/rits/computer-vision/weather-augment/&#41; datasets.)

[comment]: <> (* Super-impose snow on KITTI dataset through this [repository]&#40;https://github.com/hendrycks/robustness&#41;.)

[comment]: <> (* Generate labels YOLO can use &#40;see [Dataset directory structures]&#40;#dataset-directory-structures&#41; subsection&#41;.)

[comment]: <> (## For ImageNet and CIFAR datasets)

[comment]: <> (* Download the original train and test set for [ImageNet]&#40;https://image-net.org/download.php&#41; & [ImageNet-C]&#40;https://zenodo.org/record/2235448#.Yn5OTrozZhE&#41; datasets.)

[comment]: <> (* Download the original train and test set for [CIFAR-10]&#40;https://www.cs.toronto.edu/~kriz/cifar.html&#41; & [CIFAR-10C]&#40;https://zenodo.org/record/2535967#.Yn5QwbozZhE&#41; datasets.)

[comment]: <> (* Generate _corrupted_ version of train sets through this [repository]&#40;https://github.com/hendrycks/robustness&#41;.)

[comment]: <> (## Dataset directory structures)

[comment]: <> (### For KITTI labels:)

[comment]: <> (To generate labels YOLO can use from the original KITTI labels run)

[comment]: <> (`python main.py --kitti_to_yolo_labels /path/to/original/kitti`)

[comment]: <> (This is expecting the path to the original KITTI directory structure)

[comment]: <> (```)

[comment]: <> (path_to_specify)

[comment]: <> (└── raw)

[comment]: <> (    └── training)

[comment]: <> (        ├── image_2)

[comment]: <> (        └── label_2)

[comment]: <> (```)

[comment]: <> (Which will create a `yolo_style_labels` directory in the `raw` directory, containing)

[comment]: <> (the KITTI labels in a format YOLO can use.)

[comment]: <> (### For all datasets:)

[comment]: <> (Structure the choosen dataset&#40;s&#41; as described [here]&#40;directory_scructures.md&#41;.)

[comment]: <> (# Running Experiments)
Then setup up user specific paths in the `PATHS` dictionary in `config.py`.

### CIFAR-10/100C
For these experiments we use the pre-trained weights from [AugMix](https://arxiv.org/abs/1912.02781),
hosted [here](https://drive.google.com/drive/folders/1Yr3_IBB53b_DI2A6-KwLnypvgEUKSIHq?usp=sharing).

Also, add the `PATH` to these weights.
After setting up `PATHS` dictionary for a user, ActMad CIFAR-10/100 experiments
for the highest severity of corruptions can be run with the following command:
```
python main.py --usr <NAME_OF_USER> --dataset <NAME_OF_DATASET>
```
Replace `<NAME_OF_DATASET>` with `cifar10` or `cifar100`.

#### Wide-ResNet-40-2 CIFAR-10/100C Results
| | Mean Error | Gauss Noise | Shot Noise | Impulse Noise | Defocus Blur | Glass Blur | Motion Blur | Zoom Blur | Snow | Frost |  Fog | Brightness | Contrast | Elastic Transform | Pixelate | Jpeg |
|-----|---| ---------- | ---------| ------------| ----------- | ---------| ---------- | --------| ---| ---- | --- | --------- | ------- | ------------ | -------| --- |
|Source C10| 18.3|28.8|22.9|26.2|9.5|20.6|10.6|9.3|14.2|15.3|17.5|7.6|20.9|14.7|41.3|14.7|
|ActMAD|10.4|13.0|11.2|15.1|7.4|15.9|8.3|7.1|9.5|9.3|10.6|5.9|8.4|12.3|9.3|13.6|
||
|Source C100| 46.7|65.7|60.1|59.1|32.0|51.0|33.6|32.4|41.4|45.2|51.4|31.6|55.5|40.3|59.7|42.4|
|ActMAD|34.6|39.6|38.4|39.5|29.1|41.5|30.0|29.1|34.0|33.2|40.2|26.4|31.5|36.4|31.4|38.9|

[comment]: <> (First the network would be trained on the original _Kitti dataset_, then this trained network will be adapted to each )

[comment]: <> (of the different weather conditions &#40;fog, rain, snow&#41; in an online manner. )


### KITTI
ActMad KITTI experiments for the highest _weather severity_ can be run with the following command:
```
python main.py --usr <NAME_OF_USER> --dataset KITTI
```
First the network will be trained on the original _Kitti dataset_ (KITTI-Clear), then this
trained network will be adapted to each
of the different weather conditions KITTI- (fog, rain, snow) in an online manner.

#### YOLOv3 KITTI Results
| |  mAP@50 (%) | Car | Van | Truck | Pedestrian | Person Sitting | Cyclist | Tram | Misc |
|-----|---| ---------- |---------- |---------- |---------- |---------- |---------- |---------- |---------- |
|Source Fog| 19.6|31.3|15.0|6.0|34.8|33.6|20.2|6.7|9.1|
|ActMAD|47.7|67.0|41.2|25.5|62.2|68.7|50.9|30.5|35.7|
||
|Source Rain| 66.5|86.4|69.6|58.6|68.6|63.7|60.2|64.5|60.4|
|ActMAD|81.4|94.2|89.2|87.3|74.1|65.6|77.9|82.5|80.1|

#### To cite us:
```bibtex
@InProceedings{mirza2023actmad,
    author    = {Mirza, M. Jehanzeb and Soneira, Pol Jane and Lin, Wei and Kozinski, Mateusz and Possegger, Horst and Bischof, Horst},
    title     = {ActMAD: Activation Matching to Align Distributions for Test-Time Training},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    year      = {2023}
}
```