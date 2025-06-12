"""
tracking.py

The tracking is in charge of localizing the camera with every frame and 
determining when a new keyframe is inserted. 

Due to timely constraints, we will pretend:
* tracking is never lost (no heavy occlusions)
* no abrupts movements
* frame 0 is a Keyframe 

After initialization is finished, a local visible map is retrieved from
the covisibility graph of keyframes. Matches w/ local map pts are searched
through reprojection, and then camera movement is optimized with all the 
matches. 

Keyframe insertion is decided then. 

"""

# imports
from util import Keyframe

class Tracking:
    def __init__(
            self
        ) -> None:
        pass

    def orb_extraction(
            self
        ) -> None:
        pass
        
    def pose_estimation():
        pass

    def optimize():
        pass

    def add_keyframe():
        pass




