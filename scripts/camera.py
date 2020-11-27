#!/usr/bin/env python

import os
import cv2
import rospy
import random
from sensor_msgs.msg import Image
from ros_ai_task.msg import Int32Header
from cv_bridge import CvBridge

if __name__ == '__main__':
    image_publisher = rospy.Publisher('raw_image', Image, queue_size=1)
    number_publisher = rospy.Publisher("image_number", Int32Header, queue_size=1)
    rospy.init_node('camera', anonymous=True)
    current_folder = os.path.dirname(__file__)
    cvBridge = CvBridge()
    while not rospy.is_shutdown():
        intValue = random.randint(0, 9)
        img = cv2.imread(current_folder + '/../img/' + str(intValue) + '.png')
        img_msg = cvBridge.cv2_to_imgmsg(img)
        int_msg = Int32Header(data=intValue)
        image_publisher.publish(img_msg)
        number_publisher.publish(int_msg)
        rospy.sleep(1)
