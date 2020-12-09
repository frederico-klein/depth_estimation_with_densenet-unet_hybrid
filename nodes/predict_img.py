#!/usr/bin/env python
# MIT License
#
# Copyright (c) 2018 Károly Harsányi
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import cv2
import torch
import argparse
from network import DensenetUnetHybrid
import image_utils

def predict_img(img_path):
    """Inference a single image."""
    # switch to CUDA device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Use GPU: {}'.format(str(device) != 'cpu'))

    # load model
    print('Loading model...')
    model = DensenetUnetHybrid.load_pretrained(device=device)
    model.eval()

    # load image
    img = cv2.imread(img_path)[..., ::-1]
    img = image_utils.scale_image(img)
    img = image_utils.center_crop(img)
    inp = image_utils.img_transform(img)
    inp = inp[None, :, :, :].to(device)

    # inference
    print('Running the image through the network...')
    output = model(inp)

    # transform and plot the results
    output = output.cpu()[0].data.numpy()
    image_utils.show_img_and_pred(img, output)


def get_arguments():
    """Get command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img_path', required=True, type=str, help='Path to the input image.')
    return parser.parse_args()


if __name__ == '__main__':
    args = get_arguments()
    predict_img(args.img_path)
