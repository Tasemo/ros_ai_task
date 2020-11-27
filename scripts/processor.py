#!/usr/bin/env python

import rospy
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def onImage(data):
    cvBridge = CvBridge()
    image = cvBridge.imgmsg_to_cv2(data)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (28, 28))
    image_publisher.publish(cvBridge.cv2_to_imgmsg(image))

if __name__ == '__main__':
    image_publisher = rospy.Publisher('processed_image', Image, queue_size=1)
    rospy.init_node('processor', anonymous=True)
    rospy.Subscriber('raw_image', Image, onImage)
    rospy.spin()
