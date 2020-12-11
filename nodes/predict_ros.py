#!/usr/bin/env python

import rospy
import cv2
import torch
import numpy as np
from network import DensenetUnetHybrid
import image_utils

from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError

class Predictor:
    def __init__(self):
        self.image_callback_called = False
        self.pI = PredictInterface()
        rospy.init_node('predictor', anonymous=True)
        rospy.Subscriber("image_in", Image, self.image_callback) ## to image
        self.image_pub = rospy.Publisher('image_out', Image, queue_size=1)
        self.bridge = CvBridge()
        rospy.loginfo('Predictor instantiated.')
        rospy.spin()

    def image_callback(self, msg):
        try:
            if not self.image_callback_called:
                rospy.loginfo('Received at least one image.')
                self.image_callback_called = True
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            outimage = self.pI.predict(cv_image)
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(outimage, "passthrough"))

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
        p = Predictor()

    except rospy.ROSInterruptException:
        pass
    predict_img(args.img_path)
