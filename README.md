# SMSnet:  Semantic  Motion  Segmentationusing  Deep  Convolutional  Neural  Networks
SMSnet is a deep learning model for motion image segmentation, where the goal is to assign motion labels (moving or static) to every pixel in the input image. SMSnet is easily trainable on a single GPU with 12 GB of memory and has a fast inference time. SMSnet is benchmarked on Cityscapes and KITTI datasets.

This repository contains our TensorFlow implementation of SMSnet which allows you to train your own model on any dataset and evaluate the results in terms of the mean IoU metric.

If you find the code useful for your research, please consider citing our paper:
```
@inproceedings{vertens2017smsnet,
  title={Smsnet: Semantic motion segmentation using deep convolutional neural networks},
  author={Vertens, Johan and Valada, Abhinav and Burgard, Wolfram},
  booktitle={2017 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={582--589},
  year={2017},
  organization={IEEE}
}
```

## Live Demo
http://deepmotion.cs.uni-freiburg.de/

## Example Segmentation Results

| Dataset       | RGB_Image     | Optical_Flow_Image| Motion_Segmented_Image|
| ------------- | ------------- | -------------  | -------------  |
| Cityscapes    |<img src="images/city_rgb.png" width=300> |  <img src="images/city_flow.png" width=300>| <img src="images/city_prediction.png" width=300>|
| Forest  | <img src="images/kitti_rgb.png" width=300>  |<img src="images/kitti_flow.png" width=300> |<img src="images/kitti_prediction.png" width=300> |

## Contacts
* [Abhinav Valada](http://www2.informatik.uni-freiburg.de/~valada/)
* [Rohit Mohan](https://github.com/mohan1914)

## System Requirements

#### Programming Language
```
Python 2.7
```

#### Python Packages
```
tensorflow-gpu 1.4.0
```
