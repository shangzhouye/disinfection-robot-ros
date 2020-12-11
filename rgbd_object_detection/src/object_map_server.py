#!/usr/bin/env python

''' This node maintains an object-level map

Note that the initialization of objects are important

PARAMETERS:
    ground_plane_height_
    iou_thresh_
    dist_thresh_
    need_clean_thresh_
    duplicate_iou_thresh_
'''

from __future__ import print_function
import rospy
import cv2
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
import numpy as np
import sys
import copy
# use shapely for IoU calculation of polygons
import shapely.geometry

from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

import tf
from tf import TransformListener
from geometry_msgs.msg import PoseStamped

class ObjectMarker:
    def __init__(self, points):
        self.points_ = points # stores all the points of the convex hull
        self.associated_times_ = 1 # record how many times an object has been associated
        self.occupied_ = 0 # record how many frames an object has been occupied (a person is there)
        self.need_clean_ = False # record whether an object needs disinfection

    def update_points(self, new_points):
        self.points_ = new_points

    def add_associated_times(self):
        self.associated_times_ += 1
    
    def add_occupied_times(self):
        self.occupied_ += 1
    
    def reset_occupied_times(self):
        self.occupied_ = 0

    def set_need_clean(self):
        self.need_clean_ = True

    def reset_need_clean(self):
        self.need_clean_ = False


class ObjectMap:

    def __init__(self):
        self.result_pub_ = rospy.Publisher("object_map", MarkerArray, queue_size=10)

        # subscribe object detections from current frame
        self.convex_hull_sub_ = rospy.Subscriber("object_convex_hull", MarkerArray, self.convex_hull_callback, queue_size=10)
        
        self.people_sub_ = rospy.Subscriber("/hdl_people_tracking_nodelet/markers", MarkerArray, self.people_callback, queue_size=10)
        '''
        On the /hdl_people_tracking_nodelet/markers topic
            - the first marker is a CUBE_LIST
            - people poses are in marker_array.markers[0].points
            - following markers in marker_array are text
        '''


        self.ground_plane_height_ = -0.35

        self.object_map_ = []

        self.iou_thresh_ = 0.05 # iou threshold for data association
        self.dist_thresh_ = 0.6 # the distance from a person to an object for the object to be considered as occupied
        self.need_clean_thresh_ = 80 # how many frame an object is occupied make it needs to be cleaned
        self.duplicate_iou_thresh_ = 0.8 # threshold to view two detections as duplicate in the same frame

        self.tf_listener_ = TransformListener()

    def publish_convex_hull_marker(self):
        ''' Publish the object map
        '''
        marker_array = MarkerArray()

        for i in range(len(self.object_map_)):
            line_strip = Marker()
            line_strip.header.frame_id = "map"
            line_strip.header.stamp = rospy.Time.now()
            line_strip.ns = "object_map"
            line_strip.action = Marker.ADD
            line_strip.pose.orientation.w = 1.0

            line_strip.id = i

            line_strip.type = Marker.LINE_STRIP

            line_strip.scale.x = 0.05

            line_strip.color.g = 1.0
            if self.object_map_[i].occupied_ > 0:
                line_strip.color.r = self.object_map_[i].occupied_ / float(self.need_clean_thresh_)
            elif self.object_map_[i].need_clean_ == True:
                line_strip.color.r = 1.0
            else:
                line_strip.color.b = 1.0
            
            line_strip.color.a = 1.0

            for j in range(self.object_map_[i].points_.shape[0]):
                point = Point()
                point.x = self.object_map_[i].points_[j, 0]
                point.y = self.object_map_[i].points_[j, 1]
                point.z = 0
                line_strip.points.append(point)

            point = Point()
            point.x = self.object_map_[i].points_[0, 0]
            point.y = self.object_map_[i].points_[0, 1]
            point.z = 0
            line_strip.points.append(point)

            line_strip.lifetime = rospy.Duration(0)

            marker_array.markers.append(line_strip)

            # Add 'Need Disinfection' tag
            if self.object_map_[i].need_clean_ == True:
                text_marker = Marker()
                text_marker.header = copy.deepcopy(line_strip.header)
                text_marker.action = Marker.ADD
                text_marker.ns = "disinfection_tag"
                text_marker.id = copy.deepcopy(line_strip.id)
                text_marker.lifetime = rospy.Duration(0)
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.scale.z = 0.2
                text_marker.pose.position = copy.deepcopy(line_strip.points[0])
                text_marker.pose.position.z += 0.3
                text_marker.color.r = 1.0
                text_marker.color.g = 1.0
                text_marker.color.a = 1.0
                text_marker.text = "Need Disinfection"
                marker_array.markers.append(text_marker)
        
        self.result_pub_.publish(marker_array)


    def calculate_iou(self, polygon_1, polygon_2):
        ''' Calculate IoU of two polygons

        Args:
            polygon_1/polygon_2: polygons as 2D numpy array
        '''
        poly_1 = shapely.geometry.Polygon(polygon_1)
        poly_2 = shapely.geometry.Polygon(polygon_2)
        iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
        return iou

    def data_association(self, current_frame_detection):
        ''' Implement the data association pipeline
        '''
        # create a matrix of IoU scores
        # rows are detection list, columns are map list
        '''
        Example:
        Detection\Map     Object 0     Object 1     Object 2
        Detection A       IoU = 0         0            0 
        Detection B          0.56         0            0
        Detection C          0            0.77         0

        Object 2 in the map is not matches
        Detection A is a new object
        '''

        iou_matrix = np.zeros((len(current_frame_detection), len(self.object_map_)))

        for i in range(iou_matrix.shape[0]):
            for j in range(iou_matrix.shape[1]):
                # calculate iou and fill the matrix
                iou = self.calculate_iou(current_frame_detection[i], self.object_map_[j].points_)
                # it can be a new detection is iou is below the threshold
                if iou <= self.iou_thresh_:
                    iou = 0
                iou_matrix[i, j] = -iou # fill with negative value as cost

        new_detection_idx = []
        for i in range(iou_matrix.shape[0]):
            # find new detections
            if np.sum(iou_matrix[i, :]) == 0:
                # this is a new detection
                self.object_map_.append(ObjectMarker(current_frame_detection[i]))
                new_detection_idx.append(i)

        not_matched_objects = []
        for j in range(iou_matrix.shape[1]):
            # find not matched objects
            if np.sum(iou_matrix[:, j]) == 0:
                not_matched_objects.append(j)

        # run the hungarian algorithm
        row_ind, col_ind = linear_sum_assignment(iou_matrix)

        for idx in range(row_ind.shape[0]):
            if (row_ind[idx] in new_detection_idx) or (col_ind[idx] in not_matched_objects):
                # either no match or new detection case
                continue
            
            matched_detection = row_ind[idx]
            matched_object_in_map = col_ind[idx]

            # if the match iou is too small, do nothing
            # caveat: add negative sign here
            if -iou_matrix[matched_detection, matched_object_in_map] < self.iou_thresh_:
                continue

            # otherwise, get a new convex hull
            self.object_map_[matched_object_in_map].add_associated_times()
            new_points = np.append(self.object_map_[matched_object_in_map].points_, \
                                    current_frame_detection[matched_detection],\
                                    axis = 0)
            
            new_convex_hull = ConvexHull(new_points)
            # update the object
            self.object_map_[matched_object_in_map].update_points(new_points[new_convex_hull.vertices, :])

    def check_duplicate(self, polygon_1, polygon_2):
        ''' Check if two detections are duplicate by 
                checking whether one convex hull is (almost) completely inside the other one

        Args:
            polygon_1/polygon_2: polygons as 2D numpy array
        '''

        poly_1 = shapely.geometry.Polygon(polygon_1)
        poly_2 = shapely.geometry.Polygon(polygon_2)
        intersection = poly_1.intersection(poly_2).area
        if (intersection / poly_1.area) > self.duplicate_iou_thresh_ \
             or (intersection / poly_2.area) > self.duplicate_iou_thresh_:
            return True
        else:
            return False


    def remove_duplication_detection(self, current_frame_detection):
        ''' Remove duplicate detection and combine them into one convex hull
        '''
        output = []
        for i in range(len(current_frame_detection)):
            no_duplicate = True

            for j in range(i + 1, len(current_frame_detection)):
                if self.check_duplicate(current_frame_detection[i], current_frame_detection[j]):
                    no_duplicate = False
                    new_points = np.append(current_frame_detection[i], \
                                            current_frame_detection[j],\
                                            axis = 0)
                    new_convex_hull = ConvexHull(new_points)
                    # update the new convex hull to the latter one
                    # it will then be added to the output when the loop traverse to there
                    current_frame_detection[j] = new_points[new_convex_hull.vertices, :]
            
            if no_duplicate:
                output.append(current_frame_detection[i])
        
        return output


    def convex_hull_callback(self, data):
        # subcribe to all the objects (convex hull) in current frame

        current_frame_detection = [] # stores all the objects in this frame, a list of numpy arrays

        for marker in data.markers:
            object_convex_hull =  np.empty((0,2), float)
            for point in marker.points:
                # add each point into the list
                object_convex_hull = np.append(object_convex_hull, [[point.x, point.y]], axis=0)
            
            current_frame_detection.append(object_convex_hull)

        if (len(current_frame_detection) == 0):
            return
        
        current_frame_detection = self.remove_duplication_detection(current_frame_detection)

        # do data association using the Hungarian algorithm
        if len(self.object_map_) == 0:
            # initailize the map
            for det in current_frame_detection:
                self.object_map_.append(ObjectMarker(det))
            return
        else:
            self.data_association(current_frame_detection)


    def people_callback(self, data):
        # traverse all the objects in the map
        # if a person is within a specific distance, add 1 to occupied counting

        # if hasn't initialized yet, return
        if len(self.object_map_) == 0:
            return

        if self.tf_listener_.frameExists("map") and self.tf_listener_.frameExists("velodyne"):
            t = self.tf_listener_.getLatestCommonTime("map", "velodyne")

        
        # store all the person
        person_list = []
        for person_marker in data.markers[0].points:

            # transform to map frame
            pos_in_velodyne = PoseStamped()
            pos_in_velodyne.pose.position.x = person_marker.x
            pos_in_velodyne.pose.position.y = person_marker.y
            pos_in_velodyne.header.frame_id = "velodyne"
            pos_in_velodyne.pose.orientation.w = 1.0
            pos_in_map = self.tf_listener_.transformPose("map", pos_in_velodyne)


            person = shapely.geometry.Point(pos_in_map.pose.position.x,\
                                            pos_in_map.pose.position.y)
            person_list.append(person)
        
        
        for obj_idx in range(len(self.object_map_)):
            polygon = shapely.geometry.Polygon(self.object_map_[obj_idx].points_)
            if_occupied = False
            for person in person_list:
                distance = person.distance(polygon)
                if distance < self.dist_thresh_:
                    if_occupied = True
            
            if if_occupied == True:
                self.object_map_[obj_idx].add_occupied_times()
            else:
                self.object_map_[obj_idx].reset_occupied_times()
        
        for idx in range(len(self.object_map_)):
            # if an object has been occupied for more than a certain number of frames
            # set need clean to be true
            if self.object_map_[idx].occupied_ > self.need_clean_thresh_:
                self.object_map_[idx].set_need_clean()


def main(args):
    rospy.init_node('object_map_server', anonymous=True)
    object_map_server = ObjectMap()

    # Publish object marker in the main loop
    rate = rospy.Rate(10) # 10hz
    while not rospy.is_shutdown():
        if not len(object_map_server.object_map_) == 0:
            object_map_server.publish_convex_hull_marker()
        rate.sleep()

    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)