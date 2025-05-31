from util import Keyframe



class LocalMapping:
    def __init__(self):
        self.covisibility_graph = CovisibilityGraph()

    def map_initialization(self, kf1: Keyframe, kf2: Keyframe):
        pass

    def insert_keyframe(self, kf: Keyframe):
        pass

    def _map_points_culling(self):
        """
        of the provided points, it must satisfy:
        - more than 25% of the (predictied to be visible; covisible) frames
        - must be observed from at least three keyframes from map point creation
        """

        pass

    def _new_map_points(self):
        """
        new map points from triangulated ORB features by the connected keyframes via the covisibility graph
        """
        pass

    def _local_bundle_adjustment(self):
        """
        optimizes current keyframe + adjacent keyframe (via cov. graph) + all map points seen by these keyframes
        """
        pass

    def _local_keyframe_culling(self):
        """
        keyframes are discareded if 90% of its mapped points is visible/seen by other 3 keyframes in same/finer scale
        """
        pass


class CovisibilityGraph:
    def __init__(self):
        pass

