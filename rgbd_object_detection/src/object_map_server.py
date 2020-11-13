#!/usr/bin/env python

''' This node maintains an object-level map

Note that the initialization of objects are important

Todos:
- Wrap object into a class
- only publish an object when its associated time > threshold
- only set to can clean when an object is not occupied for a few consecutive frames
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
# use shapely for IoU calculation of polygons
import shapely.geometry

from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull

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

        self.object_map_ = None
        self.associated_times_ = None # record how many times an object has been associated
        self.occupied_ = None # record how many frames an object has been occupied (a person is there)
        self.need_clean_ = None # record whether an object needs disinfection

        self.iou_thresh_ = 0.1 # iou threshold for data association

        self.dist_thresh_ = 0.6 # the distance from a person to an object for the object to be considered as occupied

        self.need_clean_thresh_ = 60 # how many frame an object is occupied make it needs to be cleaned

    def publish_convex_hull_marker(self):
        ''' Publish the object map
        '''
        marker_array = MarkerArray()

        for i in range(len(self.object_map_)):
            line_strip = Marker()
            line_strip.header.frame_id = "velodyne"
            line_strip.header.stamp = rospy.Time.now()
            line_strip.ns = "object_map"
            line_strip.action = Marker.ADD
            line_strip.pose.orientation.w = 1.0

            line_strip.id = i

            line_strip.type = Marker.LINE_STRIP

            line_strip.scale.x = 0.05

            line_strip.color.g = 1.0
            if self.need_clean_[i] == True:
                line_strip.color.r = 1.0
            else:
                line_strip.color.b = 1.0
            
            line_strip.color.a = 1.0

            for j in range(self.object_map_[i].shape[0]):
                point = Point()
                point.x = self.object_map_[i][j, 0]
                point.y = self.object_map_[i][j, 1]
                point.z = self.ground_plane_height_
                line_strip.points.append(point)

            point = Point()
            point.x = self.object_map_[i][0, 0]
            point.y = self.object_map_[i][0, 1]
            point.z = self.ground_plane_height_
            line_strip.points.append(point)

            line_strip.lifetime = rospy.Duration(0)

            marker_array.markers.append(line_strip)
        
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

    def convex_hull_callback(self, data):
        # subcribe to all the objects (convex hull) in current frame

        current_frame_detection = [] # stores all the objects in this frame, a list of numpy arrays

        for marker in data.markers:
            object_convex_hull =  np.empty((0,2), float)
            for point in marker.points:
                # add each point into the list
                object_convex_hull = np.append(object_convex_hull, [[point.x, point.y]], axis=0)
            
            current_frame_detection.append(object_convex_hull)


        # do data association using the Hungarian algorithm

        if self.object_map_ == None:
            # initailize the map
            self.object_map_ = current_frame_detection
            self.associated_times_ = [1] * len(current_frame_detection)
            self.occupied_ = [0] * len(current_frame_detection)
            self.need_clean_ = [False] * len(current_frame_detection)
            return

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
                iou = self.calculate_iou(current_frame_detection[i], self.object_map_[j])
                iou_matrix[i, j] = -iou # fill with negative value as cost

        new_detection_idx = []
        for i in range(iou_matrix.shape[0]):
            # find new detections
            if np.sum(iou_matrix[i, :]) == 0:
                # this is a new detection
                self.object_map_.append(current_frame_detection[i])
                self.associated_times_.append(1)
                self.occupied_.append(0)
                self.need_clean_.append(False)
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
            self.associated_times_[matched_object_in_map] += 1
            new_points = np.append(self.object_map_[matched_object_in_map], \
                                    current_frame_detection[matched_detection],\
                                    axis = 0)
            
            new_convex_hull = ConvexHull(new_points)
            # update the object
            self.object_map_[matched_object_in_map] = new_points[new_convex_hull.vertices, :]

        

        # publish object list
        self.publish_convex_hull_marker()


    def people_callback(self, data):
        # traverse all the objects in the map
        # if a person is within a specific distance, add 1 to occupied counting

        # if hasn't initialized yet, return
        if self.object_map_ == None:
            return
        
        # store all the person
        person_list = []
        for person_marker in data.markers[0].points:
            person = shapely.geometry.Point(person_marker.x,\
                                            person_marker.y)
            person_list.append(person)
        
        
        for obj_idx in range(len(self.object_map_)):
            polygon = shapely.geometry.Polygon(self.object_map_[obj_idx])
            if_occupied = False
            for person in person_list:
                distance = person.distance(polygon)
                if distance < self.dist_thresh_:
                    if_occupied = True
            
            if if_occupied == True:
                self.occupied_[obj_idx] += 1
        
        print(self.occupied_)
        for idx in range(len(self.occupied_)):
            # if an object has been occupied for more than a certain number of frames
            # set need clean to be true
            if self.occupied_[idx] > self.need_clean_thresh_:
                self.need_clean_[idx] = True

def main(args):
    object_map_server = ObjectMap()
    rospy.init_node('object_map_server', anonymous=True)
    rospy.spin()

if __name__ == '__main__':
    main(sys.argv)