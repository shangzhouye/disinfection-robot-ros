#!/usr/bin/env python3

''' This node using Mask RCNN to do instance segmentation on 2D images

SUBSCRIBERS:
    camera/color/image_raw (sensor_msgs/Image): subscribe to the images from rgbd camera
PUBLISHERS:
    maskrcnn (sensor_msgs/Image): visualization results from maskrcnn
    maskrcnn/bbox (rgbd_object_detection/MaskrcnnResult): publish maskrcnn results
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
from rgbd_object_detection.msg import MaskrcnnResult
from sensor_msgs.msg import Image, RegionOfInterest

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
        self.result_bbox_pub_ = rospy.Publisher('maskrcnn/bbox', MaskrcnnResult, queue_size=10)
        self.bridge_ = CvBridge()
        self.color_sub_ = rospy.Subscriber("camera/color/image_raw", Image, self.color_callback, queue_size=1)

        # detectron2 setup
        self.cfg_ = get_cfg()
        self.cfg_.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg_.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        self.cfg_.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor_ = DefaultPredictor(self.cfg_)
        self.class_names_ = MetadataCatalog.get(self.cfg_.DATASETS.TRAIN[0]).get("thing_classes", None)

        '''
        Interested categories and their ids in pretrained model

        bench       	13
        chair	        56
        couch	        57
        dining table	60
        '''
        self.interested_ids_ = [13, 56, 57, 60]


    def publish_result(self, img_header, predictions, color_image):
        ''' Publish the result from maskrcnn onto the topic

        Args:
            img_header: header from the image messgage
            predictions: prediction output from detectron2 model
            color_image: the original color image
        '''

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
        else:
            return

        result_msg = MaskrcnnResult()
        result_msg.header = img_header
        result_msg.color_image = self.bridge_.cv2_to_imgmsg(color_image, "bgr8")

        for i in range(predictions.pred_classes.size()[0]):

            # ignore this box if it is not an interested catagory
            if not (predictions.pred_classes[i] in self.interested_ids_):
                continue

            result_msg.class_ids.append(int(predictions.pred_classes[i]))
            result_msg.class_names.append(str(np.array(self.class_names_)[predictions.pred_classes[i]]))
            result_msg.scores.append(predictions.scores[i])
            x1, y1, x2, y2 = boxes[i].tensor[0]
            mask = np.zeros(masks[i].shape, dtype="uint8")
            mask[masks[i, :, :]]=255
            mask = self.bridge_.cv2_to_imgmsg(mask)
            result_msg.masks.append(mask)

            box = RegionOfInterest()
            box.x_offset = int(x1)
            box.y_offset = int(y1)
            box.height = int(y2 - y1)
            box.width = int(x2 - x1)
            result_msg.boxes.append(box)

        return result_msg



    def color_callback(self, data):

        if (data.header.seq % 5) != 0:
            return

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
        outputs_cpu = outputs["instances"].to("cpu")
        # end = time.time()
        # print("Inference time: ", end - start)

        # print("Classes list: ", outputs["instances"].pred_classes)
        # print("Bounding boxes", outputs["instances"].pred_boxes)

        publish_msg = self.publish_result(data.header, outputs_cpu, cv_image)
        self.result_bbox_pub_.publish(publish_msg)

        # visualize results
        visual = Visualizer(cv_image[:, :, ::-1], MetadataCatalog.get(self.cfg_.DATASETS.TRAIN[0]), scale=1)
        out = visual.draw_instance_predictions(outputs_cpu)
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