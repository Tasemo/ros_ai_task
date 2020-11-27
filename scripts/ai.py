#!/usr/bin/env python

import os
import rospy
import torch
from ros_ai_task.model import MNISTModel
from ros_ai_task.srv import ImagePrediction, ImagePredictionResponse

def predictImage(req):
    normalized = [x / 255 for x in req.image.data]
    result = model(torch.Tensor(normalized))
    probability = torch.exp(result.max()).item()
    number = result.argmax().item()
    return ImagePredictionResponse(number, probability)

if __name__ == "__main__":
    model = MNISTModel(28, 28)
    current_folder = os.path.dirname(__file__)
    state = torch.load(current_folder + "/../model/trainedModel.pt")
    model.load_state_dict(state)
    model.eval()
    rospy.init_node('ai')
    rospy.Service('image_prediction', ImagePrediction, predictImage)
    rospy.spin()
