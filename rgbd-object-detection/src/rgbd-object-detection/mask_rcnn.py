#!/usr/bin/env python3

''' This node using Mask RCNN to do instance segmentation on 2D images

SUBSCRIBERS:
    camera/color/image_raw (sensor_msgs/Image): subscribe to the images from rgbd camera
PUBLISHERS:
    maskrcnn (sensor_msgs/Image): visualization results from maskrcnn
'''

from __future__ import print_function
import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import torch, torchvision
import time
print(torch.__version__, "Use CUDA? ", torch.cuda.is_available())

# detectron2 setup
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
# import some common libraries
import numpy as np
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


class MaskRCNN:

    def __init__(self):
        self.result_pub_ = rospy.Publisher("maskrcnn", Image, queue_size=10)
        self.bridge_ = CvBridge()
        self.color_sub_ = rospy.Subscriber("camera/color/image_raw", Image, self.color_callback)

        # detectron2 setup
        self.cfg_ = get_cfg()
        self.cfg_.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg_.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg_.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor_ = DefaultPredictor(self.cfg_)


    def color_callback(self, data):

        # Read the image
        try:
            cv_image = self.bridge_.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        # (rows, cols, channels) = cv_image.shape
        # print("Received image: ", rows, " ", cols, " ", channels)

        # inference on image
        # start = time.time()
        outputs = self.predictor_(cv_image)
        # end = time.time()
        # print("Inference time: ", end - start)

        # visualize results
        # print("Classes list: ", outputs["instances"].pred_classes)
        # print("Bounding boxes", outputs["instances"].pred_boxes)
        visual = Visualizer(cv_image[:, :, ::-1], MetadataCatalog.get(self.cfg_.DATASETS.TRAIN[0]), scale=1)
        out = visual.draw_instance_predictions(outputs["instances"].to("cpu"))
        try:
            self.result_pub_.publish(self.bridge_.cv2_to_imgmsg(out.get_image()[:, :, ::-1], "bgr8"))
        except CvBridgeError as e:
            print(e)
            return

def main(args):
    segmentation_server = MaskRCNN()
    rospy.init_node('mask_rcnn', anonymous=True)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)