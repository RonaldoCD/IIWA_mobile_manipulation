from environment import MovableBody
from generate_models import Playground
from planner.Action import Action
from planner.Command import Command

from random import random

import numpy as np
from pydrake.all import (
    InverseKinematics,
    Solve,
    SolverOptions,
    MinimumDistanceLowerBoundConstraint,
    CalcGridPointsOptions,
    RotationMatrix,
)

from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.multibody.optimization import Toppra


from robot import ConfigurationSpace, Range

from pydrake.solvers import SnoptSolver

from manipulation.exercises.trajectories.rrt_planner.geometry import AABB, Point
from pydrake.math import RigidTransform

from generate_models import Playground
from rrt_planning import Problem
from settings import GRIPPER_BASE_LINK
from utils import *

from robots import IIWA


class MoveToDeliver(Action):
    """
    Navigates from post-grasp pose to delivery
    """

    def __init__(self, playground: Playground, movable_body: MovableBody, max_iterations=1000, prob_sample_goal=0.05):
        super().__init__(playground)
        self.movable_body = movable_body
        self.max_iterations = max_iterations
        self.prob_sample_goal = prob_sample_goal
        self.rrt_ix = 0
        self.start_time = 0
        self.traj_opt = None
        self.toppra_path = None
        self.regenerate_path = False
        self.iiwa = None
        self.gripper_name = "wsg"
        # self.gripper_base_link_name = "l_gripper_palm_link" if "l_gripper" in gripper_name else "r_gripper_palm_link"
        self.gripper_frame_name = GRIPPER_BASE_LINK
        self.ik_solution_preseeded = True

    def state_init(self):
        self.start_time = self.time
        self.iiwa = IIWA(self.playground.construct_welded_sim_with_object_welded(
                continuous_state=self.continuous_state,
                frame_name_to_weld=self.gripper_frame_name,
                mb=self.movable_body,
            ))
        plant_context = self.iiwa.get_fresh_plant_context()

        X_WG = self.movable_body.X_WO_end
        self.X_WDeliver = self.get_deliver_pose(X_WG)
        
        query_object = self.iiwa.scene_graph.get_query_output_port().Eval(self.iiwa.get_fresh_scene_graph_context())
        self.q_init = self.iiwa.plant.GetPositions(plant_context)
        
        if self.ik_solution_preseeded:
            self.q_goal = np.array([-1.256648681160374, -0.42877396095962583, -1.1408169427411972, 
                                    -1.179779654371061, -1.3890241125896183, 0.7543578588148786, 
                                    -2.3128262167745812, -0.024999999938508502, 0.024999992236774907])
        else:    
            self.q_goal = self.solve_ik(self.X_WDeliver, self.gripper_name)
        iiwa_problem = Iiwa_Problem(self.iiwa, query_object, plant_context, self.q_init, self.q_goal)
        self.path = self.rrt_planning(iiwa_problem, self.max_iterations, self.prob_sample_goal)
        self.path = self.path[1:] if self.path[0] == self.path[1] else self.path
        print("Move to Deliver Path Len: ", len(self.path))
        self.rrt_ix = 0

    def run(self, prev_command: Command):
        t = self.time - self.start_time
        done = False
        
        if self.rrt_ix + 1 < len(self.path) and matching_q_holding_obj(self.continuous_state[:self.iiwa.num_joints()], 
                                                                       self.path[self.rrt_ix], 
                                                                       atol=8e-2):
            print("  rrt_ix: ", self.rrt_ix)
            # print("Deliver: ", self.time)
            self.rrt_ix += 1
            self.iiwa = IIWA(self.playground.construct_welded_sim_with_object_welded(
                continuous_state=self.continuous_state,
                frame_name_to_weld=self.gripper_frame_name,
                mb=self.movable_body,
            ))
            plant_context = self.iiwa.get_fresh_plant_context()
            q_start = self.iiwa.plant.GetPositions(plant_context)
            q_end = np.array(self.path[self.rrt_ix])
            self.traj_opt = self.init_traj_opt(q_start, q_end, # max_t=scaled_max_t,
                                               start=True if self.rrt_ix - 1 == 0 else False,
                                               end=True if self.rrt_ix + 1 == len(self.path) else False)
            
            gridpts = Toppra.CalcGridPoints(self.traj_opt, CalcGridPointsOptions()).reshape(-1, 1)
            self.toppra_path = Toppra(self.traj_opt, self.iiwa.plant, gridpts).SolvePathParameterization()
            self.record_ee_position(plant_context)
            self.display_path()

        if matching_q_holding_obj(self.continuous_state[:self.iiwa.num_joints()], self.path[-1], 
                                  atol=8e-2):
            done = True
        
        
        return prev_command.new_command_ignore_grippers(self.traj_opt.value(self.toppra_path.value(t))), done

    def init_traj_opt(self, q_start, q_end, min_t=0.5, max_t=4.0, n_ctrl_pts=10, eps=8e-3, avoid_collisions=True, start=False, end=False):
        plant_context = self.iiwa.get_fresh_plant_context()
        solver = SnoptSolver()
        num_q = self.iiwa.num_joints()
        current_robot_pos = self.iiwa.plant.GetPositions(plant_context)[:num_q]
        
        traj_opt = KinematicTrajectoryOptimization(num_q, num_control_points=n_ctrl_pts)
        prog = traj_opt.get_mutable_prog()
        traj_opt.AddDurationCost(1.0)
        traj_opt.AddPositionBounds(
            self.iiwa.plant.GetPositionLowerLimits(), 
            self.iiwa.plant.GetPositionUpperLimits()
        )

        traj_opt.AddVelocityBounds(
            self.iiwa.get_bounded_velocity_lower_limit()/2.,
            self.iiwa.get_bounded_velocity_upper_limit()/2.
        )
        
        traj_opt.AddDurationConstraint(min_t, max_t)

        traj_opt.AddPathPositionConstraint(q_start-eps, q_start+eps, 0)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), current_robot_pos, traj_opt.control_points()[:, 0]
        )

        traj_opt.AddPathPositionConstraint(q_end-eps, q_end+eps, 1)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), current_robot_pos, traj_opt.control_points()[:, -1]
        )

        # Solve once without the collisions and set that as the initial guess for
        # the version with collisions.
        opts = SolverOptions()
        opts.SetOption(solver.id(), "minor feasibility tolerance", 1e-6)
        result = solver.Solve(prog, solver_options=opts)
        if not result.is_success():
            print("traj opt failed: no collision checking!")
            print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
            print("traj_opt solver name", result.get_solver_id().name())
            print('q_start', q_start)
            print("q_end", q_end)
        traj_opt.SetInitialGuess(traj_opt.ReconstructTrajectory(result))

        if avoid_collisions:
            # collision constraints
            collision_constraint = MinimumDistanceLowerBoundConstraint(
                self.iiwa.plant, 0.01, plant_context, None, 0.1
            )
            evaluate_at_s = np.linspace(0, 1, 10)
            for s in evaluate_at_s:
                traj_opt.AddPathPositionConstraint(collision_constraint, s)

            result = Solve(prog, solver_options=opts)
            if not result.is_success():
                print("traj opt failed: with collision checking!")
                print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
                print("collisions", min_distance_collision_checker(self.iiwa.plant, plant_context, 0.01))
                print("traj opt solver name", result.get_solver_id().name())
       
        return traj_opt.ReconstructTrajectory(result)

    def solve_ik(self, X_WG, gripper_name, max_tries=10):
        ik_context = self.iiwa.get_fresh_plant_context()
        ik = InverseKinematics(self.iiwa.plant, ik_context, with_joint_limits=True)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()  # Get MathematicalProgram
        solver = SnoptSolver()
        
        prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.iiwa.get_robot_nominal_position(), q_variables)
        gripper_frame_name = "body"
        # goal frames
        p_X_WG = X_WG.translation()
        pos_offset = 0.01*np.ones_like(p_X_WG)
        R_WG = X_WG.rotation()
        gripper_offset = np.array([0.0, 0., 0.])
        ee_pos = ik.AddPositionConstraint(
                frameA=self.iiwa.plant.world_frame(),
                frameB=self.iiwa.plant.GetFrameByName(gripper_frame_name),
                p_BQ=gripper_offset,
                p_AQ_lower=p_X_WG - pos_offset,
                p_AQ_upper=p_X_WG + pos_offset,
            )
        ee_rot = ik.AddOrientationConstraint(
                frameAbar=self.iiwa.plant.world_frame(),
                R_AbarA=R_WG,
                frameBbar=self.iiwa.plant.GetFrameByName(gripper_frame_name),
                R_BbarB=RotationMatrix(),
                theta_bound=5*np.pi/180.0,
            )
        ee_pos.evaluator().set_description("EE Position Constraint")
        ee_rot.evaluator().set_description("EE Rotation Constraint")

        # min_bound = ik.AddMinimumDistanceLowerBoundConstraint(0.03)
        # min_bound.evaluator().set_description("Minimum Distance Lower Bound Constraint")
        
        for count in range(max_tries):
            # Compute a random initial guesses
            upper_lim = self.iiwa.get_bounded_position_lower_limit()
            lower_lim = self.iiwa.get_bounded_position_upper_limit()
            
            rands = (upper_lim - lower_lim)*np.random.uniform() + lower_lim
            current_robot_pos = self.iiwa.plant.GetPositions(ik_context)
            
            prog.SetInitialGuess(q_variables, current_robot_pos + rands)

            # solve the optimization and keep the first successful one
            opts = SolverOptions()
            opts.SetOption(solver.id(), "minor feasibility tolerance", 1e-6)
            result = solver.Solve(prog, solver_options=opts)
            if result.is_success():
                print("IK succeeded in %d tries!" % (count + 1))
                print("ik solution", result.GetSolution(q_variables))
                return result.GetSolution(q_variables)

        assert result.is_success(), "IK failed!"

    def rrt_planning(self, problem, max_iterations, prob_sample_q_goal):
        """
        Input:
            problem: instance of a utility class
            max_iterations: the maximum number of samples to be collected
            prob_sample_q_goal: the probability of sampling q_goal

        Output:
            path (list): [q_start, ...., q_goal].
                        Note q's are configurations, not RRT nodes
        """
        rrt_tools = RRT_tools(problem)
        q_goal = problem.goal
        q_start = problem.start

        for k in range(max_iterations):
            q_sample = rrt_tools.sample_node_in_configuration_space()
            random_num = random()
            if random_num < prob_sample_q_goal:
                q_sample = q_goal
                n_near = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)
                intermediate_q = rrt_tools.calc_intermediate_qs_wo_collision(n_near.value, q_sample)
                # print("intermediate_q: ", intermediate_q)
                last_node = n_near
                for n in range(len(intermediate_q)):
                    last_node = rrt_tools.grow_rrt_tree(last_node, intermediate_q[n])
                    if rrt_tools.node_reaches_goal(last_node):
                        path = rrt_tools.backup_path_from_node(last_node)

                        ##
                        plant_copy = self.iiwa.plant.Clone()
                        ##
                        return path

        return None
    
    
    def record_ee_position(self, plant_context):
        frame = self.iiwa.plant.GetFrameByName(self.gripper_frame_name)
        X_WF = frame.CalcPoseInWorld(plant_context)  # Get world pose of the frame
        position = X_WF.translation().reshape(3, 1)  # Ensure it's a column vector (3x1)
        # Concatenate along axis=1 to maintain 3xN shape
        self.ee_path = np.concatenate((self.ee_path, position), axis=1)
    
    def get_deliver_pose(self, X_WG: RigidTransform) -> RigidTransform:
        """
        Returns a new RigidTransform that is 30 cm (0.3 m) closer to the origin 
        in the X direction relative to X_WG.

        Args:
            X_WG (RigidTransform): The original pose of the gripper in world coordinates.

        Returns:
            RigidTransform: The modified pose moved 30 cm in the negative X direction.
        """
        # translation_offset = [-0.3, 0, 0]  # Move 30 cm (0.3 m) in -X direction
        # X_pre_grasp = RigidTransform(
        #     X_WG.rotation(),  # Keep the same orientation
        #     X_WG.translation() + translation_offset  # Adjust the position
        # )
        p0 = np.array([-0.5, 0.0, 0.90])
        # p0 = np.array([-0.50, 0.0, 0.67])
        R0 = RotationMatrix(np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).T)
        
        X_WGoal = RigidTransform(R0, p0)
        self.display_frame(X_WGoal, "MoveToDeliver")
        return X_WGoal

class Iiwa_Problem(Problem):
    def __init__(self, iiwa, query_object, plant_context, q_start: np.array, q_goal: np.array):
        self.iiwa = iiwa
        self.plant_context = plant_context
        num_q = iiwa.num_actuated_joints()

        lb = iiwa.get_bounded_position_lower_limit()
        ub = iiwa.get_bounded_position_upper_limit()

        q_start = np.clip(q_start.tolist(), lb, ub)
        q_goal = np.clip(q_goal.tolist(), lb, ub)

        range_list = []
        for i in range(lb.shape[0]):
            # if i == 2:
            #     range_list.append(Range(0, 2*np.pi))
            # else:
            range_list.append(Range(lb[i], ub[i]))
        max_steps = 1 * [np.pi / 180]  # three degrees
        cspace_pr2 = ConfigurationSpace(range_list, l2_distance, max_steps)

        # override 
        Problem.__init__(self,
                           x=5,
                           y=5,
                           robot=None,
                           obstacles=None,
                           start=tuple(q_start),
                           goal=tuple(q_goal),
                           region=AABB(Point(-3, -3), Point(3, 3)),
                           cspace=cspace_pr2,
                           display_tree=False)
        
    
    def collide(self, configuration):
        q = np.array(configuration)
        return self.exists_collision()
    
    def exists_collision(self):
        query_object = self.iiwa.plant.get_geometry_query_input_port().Eval(self.plant_context)
        inspector = query_object.inspector()
        collision_pairs = inspector.GetCollisionCandidates()
        for pair in collision_pairs:
            body_1_name = inspector.GetName(inspector.GetFrameId(pair[0]))
            body_2_name = inspector.GetName(inspector.GetFrameId(pair[1]))

            # print(f"Checking collision: {body_1_name} <--> {body_2_name}")

            if (body_1_name == "wsg::left_finger" or body_1_name == "wsg::right_finger") and body_2_name == "thing_1::brick_center":
                continue
            result = query_object.ComputeSignedDistancePairClosestPoints(pair[0], pair[1])
            distance = result.distance

            # print(f" - Distance: {distance:.5f}")
            # val = query_object.ComputeSignedDistancePairClosestPoints(pair[0], pair[1]).distance, inspector.GetName(inspector.GetFrameId(pair[0])), inspector.GetName(inspector.GetFrameId(pair[1]))
            if distance <= 0.: # slack to account for shelve and brick collision
                print(f"⚠️ Collision detected between {body_1_name} and {body_2_name}!")
                return True
        return False
    
    
