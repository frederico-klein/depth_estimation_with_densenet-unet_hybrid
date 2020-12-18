#!/usr/bin/env python

import rospy
import cv2
import torch
import numpy as np
from network import DensenetUnetHybrid
import image_utils

from std_msgs.msg import Header
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class Predictor:
    def __init__(self, republish_color=False):
        self.image_callback_called = False
        self.republish_color = republish_color ### only so that image and depth are synchronized. it saves using an approximate filter subscriber on other nodes
        self.pI = PredictInterface()
        rospy.init_node('predictor', anonymous=True)
        rospy.Subscriber("image_in", Image, self.image_callback) ## to image
        if self.republish_color:
            self.image_pub = rospy.Publisher('image/image_raw', Image, queue_size=1)
            self.camera_info_image_pub = rospy.Publisher('image/camera_info', CameraInfo, queue_size=1)

        self.depth_pub = rospy.Publisher('depth/image_raw', Image, queue_size=1)
        self.camera_info_depth_pub = rospy.Publisher('depth/camera_info', CameraInfo, queue_size=1)
        self.bridge = CvBridge()
        rospy.loginfo('Predictor instantiated.')
        rospy.spin()

    def image_callback(self, msg):
        try:
            the_now_time = rospy.Time.now()
            if not self.image_callback_called:
                rospy.loginfo('Received at least one image.')
                self.image_callback_called = True
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            outimage = self.pI.predict(cv_image)
            #i need to stamp or they will not synchronize!
            outDepthMsg = self.bridge.cv2_to_imgmsg(outimage, "passthrough")
            outDepthMsg.header.stamp = the_now_time
            outDepthMsg.header.frame_id = "camera_rgb_optical_frame" ## i kinda need to read this from the original camerrainfo

            self.depth_pub.publish(outDepthMsg)
            stdDepthCameraInfoMsg = CameraInfo()
            stdDepthCameraInfoMsg.width = 304
            stdDepthCameraInfoMsg.height = 228
            stdDepthCameraInfoMsg.distortion_model = "plumb_bob"
            stdDepthCameraInfoMsg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
            stdDepthCameraInfoMsg.K = [575.8157348632812, 0.0, 314.5, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 1.0]
            stdDepthCameraInfoMsg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            stdDepthCameraInfoMsg.P = [575.8157348632812, 0.0, 314.5, 0.0, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 0.0, 1.0, 0.0]
            stdDepthCameraInfoMsg.header = Header()
            stdDepthCameraInfoMsg.header.stamp = the_now_time
            stdDepthCameraInfoMsg.header.frame_id = "camera_rgb_optical_frame" ## i kinda need to read this from the original camerrainfo
            self.camera_info_depth_pub.publish(stdDepthCameraInfoMsg)
            if self.republish_color:
                outImageMsg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
                outImageMsg.header.stamp = the_now_time
                outImageMsg.header.frame_id = "camera_rgb_optical_frame"
                self.image_pub.publish(outImageMsg)
                stdImageCameraInfoMsg = CameraInfo()
                stdImageCameraInfoMsg.width = 640
                stdImageCameraInfoMsg.height = 480
                stdImageCameraInfoMsg.distortion_model = "plumb_bob"
                stdImageCameraInfoMsg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
                stdImageCameraInfoMsg.K = [575.8157348632812, 0.0, 314.5, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 1.0]
                stdImageCameraInfoMsg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
                stdImageCameraInfoMsg.P = [575.8157348632812, 0.0, 314.5, 0.0, 0.0, 575.8157348632812, 235.5, 0.0, 0.0, 0.0, 1.0, 0.0]
                stdImageCameraInfoMsg.header = Header()
                stdImageCameraInfoMsg.header.stamp = the_now_time
                stdImageCameraInfoMsg.header.frame_id = "camera_rgb_optical_frame" ## i kinda need to read this from the original camerrainfo
                self.camera_info_image_pub.publish(stdImageCameraInfoMsg)


        except CvBridgeError as e:
          print(e)

class PredictInterface:
    def __init__(self):
        # switch to CUDA device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        rospy.loginfo('Use GPU: {}'.format(str(self.device) != 'cpu'))

        # load model
        rospy.loginfo('Loading model...')
        self.model = DensenetUnetHybrid.load_pretrained(device=self.device)
        self.model.eval()

        self.predict_called = False

    def predict(self, img):
        img = image_utils.scale_image(img)
        img = image_utils.center_crop(img)
        inp = image_utils.img_transform(img)
        inp = inp[None, :, :, :].to(self.device)

        # inference
        if not self.predict_called:
            rospy.loginfo('Running the image through the network...')
            self.predict_called = True
        output = self.model(inp)

        # transform and output the results
        output = output.cpu()[0].data.numpy()
        pred = np.transpose(output, (1, 2, 0))
        return pred[:, :, 0]


def predict_img(img_path):
    """Inference a single image."""

    # load image
    img = cv2.imread(img_path)[..., ::-1]

    pI = PredictInterface()

    output = pI.predict(img)
    image_utils.show_img_and_pred(img, output)

if __name__ == '__main__':
    try:
        p = Predictor(republish_color=True)

    except rospy.ROSInterruptException:
        pass
