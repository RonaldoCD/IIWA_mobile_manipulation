from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import Frame


from pydrake.all import (
    InverseKinematics,
    Solve,
    SolverOptions,
    MinimumDistanceLowerBoundConstraint,
    CalcGridPointsOptions,
    RotationMatrix,
    PositionConstraint
)

import numpy as np
from pydrake.math import RigidTransform
import utils
from environment import MovableBody
from robots import IIWA
from settings import GRIPPER_BASE_LINK
from .Action import Action
from generate_models import Playground
from .Command import Command


class Grasp(Action):
    """
    Moves PR2 from pre-grasp pose into into grasping the object of interest.
    """

    def __init__(self, playground: Playground, movable_body: MovableBody):
        super().__init__(playground)
        self.movable_body = movable_body
        self.start_time = 0
        self.my_traj = None
        self.total_time = 0
        self.iiwa = None
        self.gripper_env = None
        self.gripper = None
        self.gripper_frame_name = GRIPPER_BASE_LINK

    def state_init(self):
        self.start_time = self.time
        self.iiwa = IIWA(self.playground.construct_welded_sim(self.continuous_state))

        plant_context = self.iiwa.env.get_fresh_plant_context()

        object_frame: Frame = self.movable_body.get_body(self.iiwa.plant).body_frame()
        X_WObject = object_frame.CalcPoseInWorld(plant_context)
        X_WGrasp = self.get_grasp_frame(X_WObject)
        # try:
        utils.min_distance_collision_checker(self.iiwa.plant, self.iiwa.get_fresh_plant_context(), 0.01)
        self.try_only_to(X_WGrasp)
        
    def run(self, prev_command: Command):
        t = self.time - self.start_time
        # print("Time Grasp", self.time)
        done = t > self.total_time
        return prev_command.new_command_ignore_grippers(self.my_traj.value(t)), done


    def try_only_to(self, X_WGrasp):
        distance_lower_bound = 0.0025
        # print("distance lower bound is ", distance_lower_bound)
        self.my_traj = self.iiwa.kinematic_trajectory_optimization_gripper(
            self.gripper_frame_name, X_WGrasp,
            distance_lower_bound=distance_lower_bound,
            num_control_points=5,
            max_duration=2,
            min_duration=0.1
        )
        self.total_time = self.my_traj.end_time() - self.my_traj.start_time()

    def get_grasp_frame(self, X_WObject):
        
        offset = np.array([-0.11, 0.0, 0.025])
        R0 = RotationMatrix(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T)
        # p0 = np.array([0.5, 0.0, 0.3])
        # R0 = RotationMatrix()
        
        X_WGoal = RigidTransform(R0, X_WObject.translation() + offset)
        print("    X_WGrasp: ", X_WGoal.translation())
        self.display_frame(X_WGoal, "X_WG_Grasp")
        return X_WGoal
    
