from pydrake.multibody.tree import Frame

import utils
from environment import MovableBody
from generate_models import Playground
from planner.Action import Action
from planner.Command import Command
from robots import IIWA
from settings import GRIPPER_BASE_LINK

import numpy as np

from pydrake.all import (
    Solve,
    SolverOptions,
    MinimumDistanceLowerBoundConstraint,
    RigidTransform, 
    Context, 
    RotationMatrix
)
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import SnoptSolver

from utils import *


class PostGrasp(Action):
    """
    Moves Iiwa into a post-grasp pose where the block is no longer in contact with the shelf floor,
    avoiding issues stemming from the resulting friction.
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
        # since gripper is too close to the object it cause problem for MinDistanceConstraint?
        # we weld the object to the gripper
        print("trying to construct welded env with object")
        self.iiwa = IIWA(self.playground.construct_welded_sim_with_object_welded(
            continuous_state=self.continuous_state,
            frame_name_to_weld=self.gripper_frame_name,
            mb=self.movable_body,
        ))
        # self.gripper_env = self.playground.construct_welded_sim_with_gripper(self.continuous_state)
        # self.gripper = PR2Gripper(plant=self.gripper_env.plant, base_link_name=GRIPPER_BASE_LINK)

        plant_context = self.iiwa.env.get_fresh_plant_context()

        mb = self.movable_body
        gripper_frame: Frame = self.iiwa.plant.GetFrameByName(self.gripper_frame_name)
        object_frame: Frame = mb.get_body(self.iiwa.plant).body_frame()
        X_init_WGripper = gripper_frame.CalcPoseInWorld(plant_context)
        print("X_init_gripper: ", X_init_WGripper.translation())
        X_init_WObject = object_frame.CalcPoseInWorld(plant_context)
        print("X_init_object: ", X_init_WObject.translation())

        X_WG =  mb.get_pose(self.iiwa.plant, plant_context)
        print("X_init moving body: ", X_WG.translation())
        # X_final_WObject = self.get_post_grasp_pose(X_WG, z=0.03, x=-0.3)
        # X_GO = X_init_WGripper.inverse() @ X_init_WObject
        # X_final_WGripper = X_final_WObject @ X_GO.inverse()
        X_final_WGripper = self.get_post_grasp_pose(X_WG, z=0.03, x=-0.3)
        print("X_final gripper position: ", X_WG.translation())
        
        # self.display_frame(X_final_WObject, "X_WGripper")
        print('starting traj opt')
        utils.min_distance_collision_checker(self.iiwa.plant, self.iiwa.get_fresh_plant_context(), 0.01)
        self.try_only_to(X_final_WGripper)
        print('traj opt done')

    def run(self, prev_command: Command):
        t = self.time - self.start_time
        print("Time Post Grasp", self.time)
        done = t > self.total_time
        return prev_command.new_command_ignore_grippers(self.my_traj.value(t)), done

    def try_only_to(self, X_final_WGripper):
        distance_lower_bound = 0.003
        self.my_traj = self.iiwa.kinematic_trajectory_optimization_ungrasping(
            self.gripper_frame_name, X_final_WGripper,
            distance_lower_bound=distance_lower_bound,
            num_control_points=10,
            max_duration=5,
            min_duration=1
        )
        self.total_time = self.my_traj.end_time() - self.my_traj.start_time()

    def get_post_grasp_pose(self, X_WG, x=-0.3, y=0., z = 0.03):
        # offset = np.array([x, y, z])
        # R0 = X_WG.rotation()
        # X_final_WObject = RigidTransform(R0, offset)
        
        p0 = np.array([0.45, 0.0, 0.64])
        R0 = RotationMatrix(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T)
        X_final_WObject = RigidTransform(R0, p0)
        
        self.display_frame(X_final_WObject, "X_WObject")
        return X_final_WObject
    
