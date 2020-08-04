#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
import numpy as np
import math

from scipy.spatial import KDTree

num = 0
out_path = '/home/nived/Training_Images/'
count = 0

STATE_COUNT_THRESHOLD = 3

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        self.traffic_lights_2d = None
        self.traffic_light_tree = None

        #rospy.spin()
        self.loop()

    def loop(self):
        rate = rospy.Rate(1)
        while not rospy.is_shutdown():
            rate.sleep()

    def distance(self, ax, ay, bx, by, az = 0, bz = 0):
        dist = math.sqrt((ax-bx)**2 + (ay-by)**2  + (az-bz)**2)
        return dist

    #Function to check closest traffic light
    def get_closest_traffic_light_idx(self):
        #X and Y coordinate from Current Position of vehicle
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        #Query the tree to fond the closest waypoint to the vehicle position
        closest_idx = self.traffic_light_tree.query([x,y],1)[1]

        #Check if the vehicle is ahead or behind

        closest_cord = self.traffic_lights_2d[closest_idx]
        previous_cord = self.traffic_lights_2d[closest_idx -1] # One paypoint behind -1

        #Convert to array to create a hyperplane
        closest_vector = np.array(closest_cord)
        previous_vector = np.array(previous_cord)
        position_vector = np.array([x,y])

        dot_product = np.dot(closest_vector-previous_vector, position_vector - closest_vector)

        if dot_product > 0:
            closest_idx = (closest_idx + 1) % len(self.traffic_lights_2d)
        return closest_idx


    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights
        if not self.traffic_lights_2d: # If not initialized
            #Convert to a list with 2D coordinates X and Y only
            self.traffic_lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y] for light in self.lights]
            self.traffic_light_tree = KDTree(self.traffic_lights_2d)


    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg

        #Collect Training Data
        traffic_light_id = self.get_closest_traffic_light_idx()
        x1 = self.pose.pose.position.x
        y1 = self.pose.pose.position.y
        x2 = self.lights[traffic_light_id].pose.pose.position.x
        y2 = self.lights[traffic_light_id].pose.pose.position.y

        dist = self.distance(x1, y1, x2, y2)
        light_dict = {0: 'red', 1: 'yellow', 2:'green', 4:'unknown'}
        light_color = light_dict[self.lights[traffic_light_id].state]

        global num

        if dist < 50:
            save_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            save_name = out_path + light_color + '%04d.png' % num
            cv2.imwrite(save_name, save_image)
            num = num +1
        global count

        #Training Data End

        light_wp, state = self.process_traffic_lights()
        count = 0

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''

        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            self.state_count += 1


    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        #TODO implement
        return 0

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
