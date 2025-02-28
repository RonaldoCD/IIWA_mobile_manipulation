import numpy as np
from pydrake.all import MultibodyPlant, RotationMatrix
from pydrake.geometry import SceneGraph
from pydrake.math import le
from pydrake.multibody.inverse_kinematics import PositionConstraint, MinimumDistanceLowerBoundConstraint, OrientationConstraint
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.solvers import Solve

from pydrake.multibody import inverse_kinematics
from pydrake.systems.framework import Context, Diagram

from generate_models.create_environment import Environment, GetIiwaNominalPosition, AddIiwaPlant
from settings import GRIPPER_BASE_LINK
from utils import *

class IIWA:
    """
    Class that holds functions and information about the IIWA model in the simulator for 
    easy access and use on the planner side.
    """

    def __init__(self, env: Environment):
        # do not save context. create it each time you need...
        # source of actual robot position in simulation
        self.env: Environment = env
        self.diagram: Diagram = env.diagram
        self.plant: MultibodyPlant = env.plant
        self.scene_graph: SceneGraph = env.scene_graph
        self.world = self.plant.world_frame()
        self.models_id = env.models_id
        self.l_finger_joint_name = "wsg_left_finger_sliding_joint_x" 
        self.r_finger_joint_name = "wsg_right_finger_sliding_joint_x"
        self.gripper_base_frame = GRIPPER_BASE_LINK
        # # plant frame
        # self.l_gripper_frame_name = 'l_gripper_tool_frame'
        # self.r_gripper_frame_name = 'r_gripper_tool_frame'

        # self.l_gripper_joint_name = 'pr2_l_gripper_l_finger_joint_q'
        # self.r_gripper_joint_name = 'pr2_r_gripper_l_finger_joint_q'

        # self.l_gripper_base_link_name = "l_gripper_palm_link"
        # self.r_gripper_base_link_name = "r_gripper_palm_link"

        # do not directly use _nominal_q since this value should stay constant
        self._nominal_q = GetIiwaNominalPosition()

        assert self.plant.num_positions() == self.plant.num_actuators(), "in this plant everything should be welded other than PR2 joints"
        print("Iiwa initialized")

    def get_fresh_plant_context(self):
        diagram_context = self.diagram.CreateDefaultContext()
        return self.plant.GetMyContextFromRoot(diagram_context)

    def get_fresh_scene_graph_context(self):
        diagram_context = self.diagram.CreateDefaultContext()
        return self.scene_graph.GetMyContextFromRoot(diagram_context)

    def get_robot_nominal_position(self):
        return self._nominal_q.copy()

    # todo probably deprecate this
    def get_fresh_robot_plant(self):
        robot_plant = MultibodyPlant(time_step=self.plant.time_step())
        AddIiwaPlant(robot_plant)
        robot_plant.Finalize()
        return robot_plant

    def num_actuated_joints(self):
        num_actuators = 0
        for model_id in self.models_id:
            num_actuators += len(self.plant.GetActuatedJointIndices(model_id))
        return num_actuators

    def num_joints(self):
        return len(self._nominal_q)

    def ik_escape_collision(self, max_tries=10, context=None, distance_lower_bound=None,
                            initial_guess=None, tol=2e-2):
        if initial_guess is None:
            initial_guess = self._nominal_q[:]

        ik = inverse_kinematics.InverseKinematics(self.plant, self.get_fresh_plant_context(), with_joint_limits=True)

        q_variables = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()  # Get MathematicalProgram

        if context is None:
            context = self.get_fresh_plant_context()
        cur_q = self.plant.GetPositions(context)
        prog.AddConstraint(
            le(q_variables - cur_q, tol)
        )
        prog.AddConstraint(
            le(cur_q - q_variables, tol)
        )
        nominal_error = (q_variables[3:] - self._nominal_q[3:])
        prog.AddCost(nominal_error @ nominal_error)
        if distance_lower_bound is not None:
            ik.AddMinimumDistanceLowerBoundConstraint(distance_lower_bound)

        # min_distance_collision_checker(self.plant, self.get_fresh_plant_context(), 0.01)
        best_solution = None
        min_cost = None
        for i in range(max_tries):
            ub = self.get_bounded_position_upper_limit()
            lb = self.get_bounded_position_lower_limit()
            if i == 0:
                rands = initial_guess
            else:
                rands = (ub - lb) * np.random.uniform() + lb
            prog.SetInitialGuess(q_variables, rands)
            result = Solve(prog)
            if result.is_success():
                # todo maybe put a threshold to return if cost is less than a certain amount (to reduce the computation time?)
                if (best_solution is None) or (result.get_optimal_cost() < min_cost):
                    best_solution = result.GetSolution(q_variables)
                    best_solution[2] = best_solution[2] % (2 * np.pi)  # todo this is theta. later handle this in State
                    min_cost = result.get_optimal_cost()
        if best_solution is None:
            raise Exception("IK failed")
        else:
            return best_solution

    def get_bounded_position_lower_limit(self):
        lim = self.plant.GetPositionLowerLimits()
        # print("Lim lower: ", lim)
        return np.where(lim == -np.inf, -10, lim)

    def get_bounded_position_upper_limit(self):
        lim = self.plant.GetPositionUpperLimits()
        # print("Lim upper: ", lim)
        return np.where(lim == np.inf, 10, lim)
    
    def get_bounded_pos_upper_lim_wrist_adj(self):
        lim = self.plant.GetPositionUpperLimits()
        return lim
    
    def get_bounded_velocity_lower_limit(self):
        lim = self.plant.GetVelocityLowerLimits()
        # limit on rotation and prismatic joints
        return np.where(lim == -np.inf, -0.1, lim)

    def get_bounded_velocity_upper_limit(self):
        lim = self.plant.GetVelocityUpperLimits()
        # limit on rotation and prismatic joints
        return np.where(lim == np.inf, 0.1, lim)

    # todo deprecate this
    def get_robot_position_from_plant_context(self, plant_context):
        positions = self.plant.GetPositions(plant_context)
        # print("positions: ", positions)
        return np.concatenate([
            self.plant.GetPositionsFromArray(q=positions, model_instance=model_id)
            for model_id in self.models_id
        ])

    # Deprecated: Use the updated method with multiple model IDs
    def get_robot_velocity_from_plant_context(self, plant_context):
        velocities = self.plant.GetVelocities(plant_context)
        return np.concatenate([
            self.plant.GetVelocitiesFromArray(v=velocities, model_instance=model_id)
            for model_id in self.models_id
        ])

    def kinematic_trajectory_optimization_gripper(self,
                                                  gripper_frame_name: str,
                                                  X_WFinal,
                                                  num_control_points=10,
                                                  distance_lower_bound=None,
                                                  min_duration=0.1,
                                                  max_duration=5
                                                  ):
        plant = self.plant
        plant_context = self.get_fresh_plant_context()
        num_q = self.plant.num_positions()
        # todo maybe replace this with the nominal position? this is where we want to be centered around
        q0 = plant.GetPositions(plant_context)

        ee_frame = plant.GetFrameByName(gripper_frame_name)

        X_WStart = ee_frame.CalcPoseInWorld(plant_context)

        trajopt = KinematicTrajectoryOptimization(plant.num_positions(), num_control_points=num_control_points)
        prog = trajopt.get_mutable_prog()
        trajopt.AddDurationCost(1.0)
        trajopt.AddPathLengthCost(1.0)
        trajopt.AddPositionBounds(
            self.get_bounded_position_lower_limit(), self.get_bounded_position_upper_limit()
        )
        trajopt.AddVelocityBounds(
            self.get_bounded_velocity_lower_limit(), self.get_bounded_velocity_upper_limit()
        )

        trajopt.AddDurationConstraint(min_duration, max_duration)
        
        gripper_offset = np.array([0.0, 0., 0.])
        
        # start constraint
        start_constraint = PositionConstraint(
            plant=plant,
            plant_context=plant_context,
            frameA=self.plant.world_frame(),
            frameB=ee_frame,
            p_BQ=gripper_offset,
            p_AQ_lower=X_WStart.translation() - [1e-2, 1e-3, 1e-3],
            p_AQ_upper=X_WStart.translation() + [1e-2, 1e-3, 1e-3],
        )

        # todo add start constraints to be equal to starting qs
        trajopt.AddPathPositionConstraint(start_constraint, 0)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), q0, trajopt.control_points()[:, 0]
        )

        # goal constraint
        goal_orientation_constraint = OrientationConstraint(
            plant=plant,
            plant_context=plant_context,
            frameAbar=self.plant.world_frame(),
            R_AbarA=X_WFinal.rotation(),
            frameBbar=ee_frame,
            R_BbarB=RotationMatrix(),
            theta_bound=0.25*np.pi/180,
        )

        goal_constraint = PositionConstraint(
            plant=plant,
            plant_context=plant_context,
            frameA=self.plant.world_frame(),
            frameB=ee_frame,
            p_BQ=gripper_offset,
            p_AQ_lower=X_WFinal.translation() - [2e-3, 5e-4, 1e-3],
            p_AQ_upper=X_WFinal.translation() + [2e-3, 5e-4, 1e-3],
        )
        trajopt.AddPathPositionConstraint(goal_constraint, 1)
        trajopt.AddPathPositionConstraint(goal_orientation_constraint, 1)

        prog.AddQuadraticErrorCost(
            np.eye(num_q), q0, trajopt.control_points()[:, -1]
        )

        # start and end with zero velocity
        trajopt.AddPathVelocityConstraint(
            np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
        )
        trajopt.AddPathVelocityConstraint(
            np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
        )

        # Solve once without the collisions and set that as the initial guess for
        # the version with collisions.
        result = Solve(prog)
        if not result.is_success():
            print("Trajectory optimization failed, even without collisions!")
            print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
            print("traj_opt Solver Name", result.get_solver_id().name())
        trajectory = trajopt.ReconstructTrajectory(result)
        if distance_lower_bound is None:
            return trajectory

        # now we solve with min distance constraint
        trajopt.SetInitialGuess(trajectory)

        # collision constraints
        collision_constraint = MinimumDistanceLowerBoundConstraint(
            plant, distance_lower_bound, plant_context, None, 0.1
        )

        # sample more dense at the beginning and end
        # todo be careful that this take too long
        # evaluate_at_s = np.concatenate([
        #     np.linspace(0, 0.1, 20),
        #     np.linspace(0.1, 0.8, 20),
        #     np.linspace(0.8, 1, 20)
        # ])
        evaluate_at_s = np.linspace(0, 1, 25)
        for s in evaluate_at_s:
            trajopt.AddPathPositionConstraint(collision_constraint, s)

        result = Solve(prog)
        if not result.is_success():
            print("Trajectory optimization failed")
            print(result.get_solver_id().name())
        trajectory = trajopt.ReconstructTrajectory(result)
        return trajectory

    def kinematic_trajectory_optimization_ungrasping(self,
                                                     ee_frame_name: str,
                                                     X_WFinal,
                                                     num_control_points=10,
                                                     distance_lower_bound=None,
                                                     min_duration=0.5,
                                                     max_duration=5
                                                     ):
        # plant = self.plant
        plant_context = self.get_fresh_plant_context()
        num_q = self.plant.num_positions()
        # todo maybe replace this with the nominal position? this is where we want to be centered around
        q0 = self.plant.GetPositions(plant_context)

        ee_frame = self.plant.GetFrameByName(ee_frame_name)

        X_WStart = ee_frame.CalcPoseInWorld(plant_context)

        trajopt = KinematicTrajectoryOptimization(num_q, num_control_points=num_control_points)
        prog = trajopt.get_mutable_prog()
        trajopt.AddDurationCost(1.0)
        trajopt.AddPathLengthCost(1.0)
        trajopt.AddPositionBounds(
            self.get_bounded_position_lower_limit(), self.get_bounded_position_upper_limit()
        )
        trajopt.AddVelocityBounds(
            self.get_bounded_velocity_lower_limit(), self.get_bounded_velocity_upper_limit()
        )

        trajopt.AddDurationConstraint(min_duration, max_duration)

        gripper_offset = np.array([0.0, 0., 0.])
        min_distance_target = 0.03
        eps = 1e-4

        # start constraint
        start_constraint = PositionConstraint(
            plant=self.plant,
            plant_context=plant_context,
            frameA=self.plant.world_frame(),
            frameB=ee_frame,
            p_BQ=gripper_offset,
            p_AQ_lower=X_WStart.translation() - min_distance_target,
            p_AQ_upper=X_WStart.translation() + min_distance_target,
        )


        # todo add start constraints to be equal to starting qs
        trajopt.AddPathPositionConstraint(start_constraint, 0)
        # trajopt.AddPathPositionConstraint(start_orientation_constraint, 0)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), q0, trajopt.control_points()[:, 0]
        )

        # goal constraint
        goal_constraint = PositionConstraint(
            plant=self.plant,
            plant_context=plant_context,
            frameA=self.plant.world_frame(),
            frameB=ee_frame,
            p_BQ=gripper_offset,
            p_AQ_lower=X_WFinal.translation() - [0.02, 0.02, 0.02],
            p_AQ_upper=X_WFinal.translation() + [0.02, 0.02, 0.02],
        )

        trajopt.AddPathPositionConstraint(goal_constraint, 1)

        prog.AddQuadraticErrorCost(
            np.eye(num_q), q0, trajopt.control_points()[:, -1]
        )

        # Solve once without the collisions and set that as the initial guess for
        # the version with collisions.
        result = Solve(prog)
        if not result.is_success():
            print("Trajectory optimization failed, even without collisions!")
            print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
            print("traj_opt Solver Name", result.get_solver_id().name())
        trajectory = trajopt.ReconstructTrajectory(result)
        if distance_lower_bound is None:
            return trajectory

        # now we solve with min distance constraint
        trajopt.SetInitialGuess(trajectory)

        # collision constraints
        collision_constraint = MinimumDistanceLowerBoundConstraint(
            self.plant, distance_lower_bound, plant_context, None, 0.01
        )
        evaluate_at_s = np.linspace(0, 1, 25)
        for s in evaluate_at_s:
            trajopt.AddPathPositionConstraint(collision_constraint, s)

        result = Solve(prog)
        if not result.is_success():
            print("Trajectory optimization failed")
            print(result.get_solver_id().name())
        trajectory = trajopt.ReconstructTrajectory(result)
        return trajectory


    def get_close_gripper_position(self, initial_positions, finger_position = 0.01):
        # todo this might be slow but ok because we don't use it too often...
        # maybe replace by a lookup later?
        i = 0
        for idx, name in enumerate(self.plant.GetPositionNames()):
            if name == self.l_finger_joint_name:
                i += 1    
                initial_positions[idx] = finger_position  # open angle
            elif name == self.r_finger_joint_name:
                initial_positions[idx] = -finger_position # open angle
                i += 1
            if i == 2:
                return
        raise Exception(f"Joint names haven't been found")

    def get_open_gripper_position(self, initial_positions):
        # todo this might be slow but ok because we don't use it too often...
        # maybe replace by a lookup later?
        i = 0
        for idx, name in enumerate(self.plant.GetPositionNames()):
            # print("Position Names: ", name)
            if name == self.l_finger_joint_name:
                i += 1    
                initial_positions[idx] = 0.05  # open angle
            elif name == self.r_finger_joint_name:
                initial_positions[idx] = -0.05  # open angle
                i += 1
            if i == 2:
                return
        raise Exception(f"Joint names haven't been found")

    def set_ignoring_gripper_position(self, positions_from, positions_to):
        # todo this might be slow but ok because we don't use it too often...
        # maybe replace by a lookup later?
        for idx, name in enumerate(self.plant.GetPositionNames()):
            if (name == self.l_finger_joint_name) or (self.r_finger_joint_name == name):
                continue
            positions_to[idx] = positions_from[idx]

   
    