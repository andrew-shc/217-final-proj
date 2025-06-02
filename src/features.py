import cv2
import numpy as np

def get_features(img):
    orb = cv2.ORB.create()

    # find keypts
    kp = orb.detect(img, None)

    # compute descriptors
    kp, des = orb.compute(img, kp)

    # convert des into numpy
    des = np.array(des)

    return kp, des