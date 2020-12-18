Hybrid DenseNet-based CNN for Video Stream Depth Estimation (ROS)
============================================

This repository is ROS encapsultion from from karoly's repository [depth_estimation_with_densenet-unet_hybrid](https://github.com/karoly-hars/depth_estimation_with_densenet-unet_hybrid) to estimate depth from RGB. 

I barely touched the code and I haven't retrained or anything; I just wrapped it with ROS things. 

**Also, don't use this.** There are probably more modern things out there, which you should use insted. If you are thinking you can use the estimated cloud for something like [ICP](https://en.wikipedia.org/wiki/Iterative_closest_point), you probably can't. Not without serious improvements

### Requirements
The code was tested with:
- python 3.5 and 3.6
- pytorch (and torchvision) 1.3.0
- opencv-python 3.4.3
[- matplotlib 2.2.3] it was a hassle to get X working from the docker, so I just removed this
- numpy 1.15.4

### or (which is what I did) you can use this docker:
https://github.com/frederico-klein/Tch_depth.git

### How does it run?

I can do 20fps just fine (on a 1080Ti), but the results aren't great. I've added publishers of cameraInfo messages and also republishers for rgb so you can easily depth_image_rgbxyz nodelet that will let you see the output cloud in Rviz. you will just need to remap things properly, but I should include the launch file for this

Perhaps you need to can retrain for your environment with a kinect or something to get it slightly better.

_____________________________________________________________
****Copypaste from Karoly's repo, for reference:***

### Guide
- Predicting the depth of an arbitrary image:
```
python3 predict_img.py -i <path_to_image>
```

### Evalutation
- Quantitative results on the NYU depth v2 test set:
 
| REL  |  RMSE  | Log10 |  δ1 |  δ2 |  δ3 |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| 0.129 | 0.588 | 0.056 |0.833 |0.962 |0.990 |

### TODOs:

- include the launch file for the nodelet


