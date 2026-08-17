from rocobrick.utils import *

##################################
# Implement your own policy here #
##################################
class Policy():
    def __init__(self):
        pass

    def get_action(self, obs):
        """
        Generate the robot action (joint positions) based on the current observation.
        The observation will be provided as a dictionary.
        """
        # Available Observations. Do not use more than these for your policy.
        joint_positions = obs["joint_positions"]
        images = obs["images"]
        head_rgb = images["head_rgb"]
        head_depth = images["head_depth"]
        wrist_right_rgb = images["wrist_right_rgb"]
        wrist_right_depth = images["wrist_right_depth"]
        wrist_left_rgb = images["wrist_left_rgb"]
        wrist_left_depth = images["wrist_left_depth"]

        act = None
        return act

    def is_done(self):
        """
        Termination condition for the policy. The task will be evaluated when this returns True.
        """
        return False
