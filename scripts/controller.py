#!/usr/bin/env python

import rospy
import message_filters
from sensor_msgs.msg import Image
from ros_ai_task.msg import Int32Header
from ros_ai_task.srv import ImagePrediction, ImagePredictionResponse

def onImageAndNumber(image, number):
    node_id = rospy.get_caller_id()
    rospy.loginfo("%s got image (width = %i, height = %i, encoding = %s)", node_id, image.width, image.height, image.encoding)
    rospy.loginfo("%s got int = %i", node_id, number.data)
    result = prediction(image)
    equal = result.number == number.data
    probability = result.probability * 100
    rospy.loginfo("AI node predicted int = %i with a probability = %i%%, matching: %s", result.number, probability, equal)

if __name__ == '__main__':
    rospy.wait_for_service('image_prediction')
    prediction = rospy.ServiceProxy('image_prediction', ImagePrediction)
    rospy.init_node('controller', anonymous=True)
    image_subscriber = message_filters.Subscriber('processed_image', Image)
    number_subscriber = message_filters.Subscriber('image_number', Int32Header)
    ts = message_filters.TimeSynchronizer([image_subscriber, number_subscriber], 10)
    ts.registerCallback(onImageAndNumber)
    rospy.spin()
