#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019
# Author: Seongha Park
# Date:   16 November 2021

from __future__ import absolute_import, division, print_function

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from libs.models import *
from libs.utils import DenseCRF


import time
import argparse

import waggle.plugin as plugin
from waggle.data.vision import Camera

TOPIC_WATERDETECTOR = "env.detector.water"

plugin.init()


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap


def run(args):
    config_path = args.config_path
    model_path = args.model_path
    cuda = args.cuda
    crf = args.crf
    """
    Inference from a single image
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)








    sampling_countdown = -1
    if args.sampling_interval >= 0:
        print(f"Sampling enabled -- occurs every {args.sampling_interval}th inferencing")
        sampling_countdown = args.sampling_interval

    # print("Cloud cover estimation starts...")
    #camera = Camera(args.stream)
    camera = Camera()
    while True:
        #sample = camera.snapshot()
        #image = sample.data
        #timestamp = sample.timestamp
        image = cv2.imread('nature-bird-people-grass.jpg')
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        timestamp = time.time()

        if args.debug:
            s = time.time()



        # Inference
        image, raw_image = preprocessing(image, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        labels = np.unique(labelmap)


        outputclasses = [classes[i] for i in labels]
        value = 'false'
        if 'water-other' in outputclasses or 'river' in outputclasses:
            value = 'true'


        if args.debug:
            e = time.time()
            print(f'Time elapsed for inferencing: {e-s} seconds')

        plugin.publish(TOPIC_WATERDETECTOR, value, timestamp=timestamp)
        print(f"Cloud coverage: {value} at time: {timestamp}")

        if sampling_countdown > 0:
            sampling_countdown -= 1
        elif sampling_countdown == 0:
            sample.save('sample.jpg')
            plugin.upload_file('sample.jpg')
            print("A sample is published")
            # Reset the count
            sampling_countdown = args.sampling_interval

        if args.interval > 0:
            time.sleep(args.interval)


        exit(0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-debug', dest='debug',
        action='store_true', default=False,
        help='Debug flag')
    parser.add_argument(
        '-stream', dest='stream',
        action='store', default="camera",
        help='ID or name of a stream, e.g. sample')
    parser.add_argument(
        '-interval', dest='interval',
        action='store', default=0, type=int,
        help='Inference interval in seconds')
    parser.add_argument(
        '-sampling-interval', dest='sampling_interval',
        action='store', default=-1, type=int,
        help='Sampling interval between inferencing')
    parser.add_argument(
        '-threshold', dest='threshold',
        action='store', default=0.9, type=float,
        help='Cloud pixel determination threshold')
    parser.add_argument(
        '-config-path', dest='config_path',
        action='store', required=True, type=str,
        help='Dataset configuration file in YAML')
    parser.add_argument(
        '-model-path', dest='model_path',
        action='store', required=True, type=str,
        help='PyTorch model to be loaded')
    parser.add_argument(
        '-cuda', dest='cuda',
        action='store_false', default=True,
        help='Disable CUDA')
    parser.add_argument(
        '-crf', dest='crf',
        action='store_false', default=True,
        help='CRF post-processing')
    run(parser.parse_args())
