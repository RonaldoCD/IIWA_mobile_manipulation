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
    PositionConstraint
)
from pydrake.planning import KinematicTrajectoryOptimization
from pydrake.multibody.optimization import Toppra
from pydrake.math import RigidTransform

from robot import (
    ConfigurationSpace,
    Range,
)
import matplotlib.pyplot as plt
from pydrake.perception import PointCloud
from pydrake.geometry import Rgba, Sphere



from pydrake.solvers import SnoptSolver

from manipulation.exercises.trajectories.rrt_planner.geometry import AABB, Point

from generate_models import Playground
from rrt_planning import Problem
from settings import GRIPPER_BASE_LINK
from utils import *

from robots import IIWA
from generate_models.create_environment import filterIiwaCollisionGeometry

class MoveToGrasp(Action):
    """
    Navigates for IIWA initialization to the pre-grasp pose in front of the object of interest
    """

    def __init__(self, playground: Playground, movable_body: MovableBody, max_iterations=1000, prob_sample_goal=0.01):
        super().__init__(playground)
        self.movable_body = movable_body # target body
        self.max_iterations = max_iterations
        self.prob_sample_goal = prob_sample_goal
        self.rrt_ix = 0
        self.start_time = 0
        self.traj_opt = None
        self.toppra_path = None
        self.regenerate_path = False
        self.iiwa = None
        self.current_shelf = None
        self.gripper_name = "wsg"
        self.ik_solution_preseeded = False
        self.gripper_frame_name = GRIPPER_BASE_LINK
        
    def state_init(self):
        self.start_time = self.time
        self.iiwa = IIWA(self.playground.construct_welded_sim(self.continuous_state))
        plant_context = self.iiwa.get_fresh_plant_context()

        X_WG = self.movable_body.get_pose(self.iiwa.plant, plant_context)

        # Compute the new pre-grasp pose
        self.X_WPregrasp = self.get_pre_grasp_pose(X_WG)
        
        self.iiwa.plant.GetFrameByName(self.gripper_frame_name)
        # print("X_WPregrasp: ", self.X_WPregrasp.translation)
        query_object = self.iiwa.scene_graph.get_query_output_port().Eval(self.iiwa.get_fresh_scene_graph_context())
        self.q_init = self.iiwa.plant.GetPositions(plant_context)
        # print("q_init: ", self.q_init)
        if self.ik_solution_preseeded:
            self.q_goal = np.array([0.75778046, -0.15025176, -0.64249042, 
                                    -1.87726127, -2.51179728,  0.23095827,
                                    -0.70537215, -0.02499148, 0.02499081])
        else:
            self.q_goal = self.solve_ik(self.X_WPregrasp, self.gripper_name)
        # print("q_goal: ", self.q_goal)
        
        iiwa_problem = Iiwa_Problem(self.iiwa, query_object, plant_context, self.q_init, self.q_goal)
        self.path = self.rrt_planning(iiwa_problem, self.max_iterations, self.prob_sample_goal)
        # print("PATH: ", self.path)
        self.path = self.path[1:] if self.path[0] == self.path[1] else self.path # hunt down source of duplicate init later
        print("  Len Path: ", len(self.path))
        self.rrt_ix = 0

    def run(self, prev_command: Command):
        
        t = self.time - self.start_time
        done = False

        if self.rrt_ix + 1 < len(self.path) and matching_q(self.continuous_state[:self.iiwa.num_joints()], self.path[self.rrt_ix], atol=1e-1):
            # print("  rrt_ix: ", self.rrt_ix)
            # print("Time Move: ", self.time)
            self.rrt_ix += 1
            # print("  self.continuous state: ", self.continuous_state)
            self.iiwa = IIWA(self.playground.construct_welded_sim(self.continuous_state))
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

        if matching_q(self.continuous_state[:self.iiwa.num_joints()], self.path[-1], atol=8e-2):
            done = True
        
        # q_values =  self.traj_opt.value(self.toppra_path.value(t))
        # print(f"Ix: {self.rrt_ix} | Time: {self.time:.3f}s | Q_values: {q_values.reshape(-1)}")
        
        return Command(self.iiwa, self.traj_opt.value(self.toppra_path.value(t))), done

    def init_traj_opt(self, q_start, q_end, min_t=0.5, max_t=4.0, n_ctrl_pts=10, eps=8e-3, avoid_collisions=True, start=False, end=False):
        plant_context = self.iiwa.get_fresh_plant_context()
        solver = SnoptSolver()
        num_q = self.iiwa.num_joints()
        current_robot_pos = self.iiwa.plant.GetPositions(plant_context)[:num_q]
        
        traj_opt = KinematicTrajectoryOptimization(num_q, num_control_points=n_ctrl_pts)
        prog = traj_opt.get_mutable_prog()
        
        traj_opt.AddDurationCost(1.0)
        # traj_opt.AddPathLengthCost(1.0)
        # print("upper limits: ", self.iiwa.plant.GetPositionUpperLimits())
        # print("lower limits: ", self.iiwa.plant.GetPositionLowerLimits())
        
        traj_opt.AddPositionBounds(
            self.iiwa.plant.GetPositionLowerLimits(), 
            self.iiwa.plant.GetPositionUpperLimits()
        )
        traj_opt.AddVelocityBounds(
            self.iiwa.get_bounded_velocity_lower_limit(),
            self.iiwa.get_bounded_velocity_upper_limit()
        )
        
        traj_opt.AddDurationConstraint(min_t, max_t)

        traj_opt.AddPathPositionConstraint(q_start-eps, q_start+eps, 0)
        # traj_opt.AddPathPositionConstraint(q_start, q_start, 0)
        # print("Control points: ", traj_opt.control_points()[:, 0])
        prog.AddQuadraticErrorCost(
            np.eye(num_q), current_robot_pos, traj_opt.control_points()[:, 0]
        )

        traj_opt.AddPathPositionConstraint(q_end-eps, q_end+eps, 1)
        prog.AddQuadraticErrorCost(
            np.eye(num_q), current_robot_pos, traj_opt.control_points()[:, -1]
        )

        # start and end with zero velocity
        if start:
            traj_opt.AddPathVelocityConstraint(
                np.zeros((num_q, 1)), np.zeros((num_q, 1)), 0
            )
        if end:
            traj_opt.AddPathVelocityConstraint(
                np.zeros((num_q, 1)), np.zeros((num_q, 1)), 1
            )

        # Solve once without the collisions and set that as the initial guess for
        # the version with collisions.
        # model_instance = self.iiwa.env.models_id[0]  # Get the model instance index

        # # Get all geometry IDs associated with this model instance
        # inspector = self.iiwa.env.scene_graph.model_inspector()
        # geometry_ids = inspector.GetGeometries(model_instance, Role.kProximity)

        # collision_filter_manager = self.iiwa.env.scene_graph.collision_filter_manager()

        # collision_filter_manager.Apply(
        #     CollisionFilterDeclaration().ExcludeWithin(
        #         GeometrySet(geometry_ids)
        #     )
        # )
        # filterIiwaCollisionGeometry(self.iiwa.env.scene_graph)

        opts = SolverOptions()
        opts.SetOption(solver.id(), "minor feasibility tolerance", 1e-6)
        result = solver.Solve(prog, solver_options=opts)
        if not result.is_success():
            print("Trajectory optimization failed, even without collisions!")
            print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
            print("traj_opt Solver Name", result.get_solver_id().name())
        traj_opt.SetInitialGuess(traj_opt.ReconstructTrajectory(result))

        if avoid_collisions:
            # collision constraints

            collision_constraint = MinimumDistanceLowerBoundConstraint(
                self.iiwa.plant, 0.01, plant_context, None, 0.001
            )
            evaluate_at_s = np.linspace(0, 1, 10)
            for s in evaluate_at_s:
                traj_opt.AddPathPositionConstraint(collision_constraint, s)

            result = Solve(prog, solver_options=opts)
            if not result.is_success():
                print("Trajectory optimization failed WITH COLLISION")
                infeasible_constraints = result.GetInfeasibleConstraintNames(prog)
                # print("Infeasible Constraints:")
                # for constraint in infeasible_constraints:
                #     print(f"  - {constraint}")
                print("infeasible constraints", result.GetInfeasibleConstraintNames(prog))
                print("collisions", min_distance_collision_checker(self.iiwa.plant, plant_context, 0.01))
                print(result.get_solver_id().name())
       
        return traj_opt.ReconstructTrajectory(result)

    def solve_ik(self, X_WG, gripper_name, max_tries=10):
        ik_context = self.iiwa.get_fresh_plant_context()
        ik = InverseKinematics(self.iiwa.plant, ik_context, with_joint_limits=True)
        q_variables = ik.q()  # Get variables for MathematicalProgram
        prog = ik.prog()  # Get MathematicalProgram
        solver = SnoptSolver()
        
        prog.AddQuadraticErrorCost(np.identity(len(q_variables)), self.iiwa.get_robot_nominal_position(), q_variables)
                
        # goal frames
        p_X_WG = X_WG.translation()
        pos_offset = 0.01*np.ones_like(p_X_WG)
        R_WG = X_WG.rotation()
        gripper_offset = np.array([0.0, 0., 0.])
        ee_pos = ik.AddPositionConstraint(
                frameA=self.iiwa.plant.world_frame(),
                frameB=self.iiwa.plant.GetFrameByName(GRIPPER_BASE_LINK),
                p_BQ=gripper_offset,
                p_AQ_lower=p_X_WG - pos_offset,
                p_AQ_upper=p_X_WG + pos_offset,
            )
        ee_rot = ik.AddOrientationConstraint(
                frameAbar=self.iiwa.plant.world_frame(),
                R_AbarA=R_WG,
                frameBbar=self.iiwa.plant.GetFrameByName(GRIPPER_BASE_LINK),
                R_BbarB=RotationMatrix(),
                theta_bound=np.pi/180.0,
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
            # print("Rands: ", rands)
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

    def get_pre_grasp_pose(self, X_WG: RigidTransform) -> RigidTransform:
        """
        Returns a new RigidTransform that is 30 cm (0.3 m) closer to the origin 
        in the X direction relative to X_WG.

        Args:
            X_WG (RigidTransform): The original pose of the gripper in world coordinates.

        Returns:
            RigidTransform: The modified pose moved 30 cm in the negative X direction.
        """
        # p0 = np.array([0.55, 0.0, 0.65])
        # R0 = RotationMatrix(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T)   
        # X_WGoal = RigidTransform(R0, p0)

        
        p0 = np.array([7.34858533e-01 - 0.03, -2.35778741e-05, 6.24294858e-01])
        R0 = RotationMatrix(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T)   
        X_WGoal = RigidTransform(R0, p0)
        self.display_frame(X_WGoal, "MoveToGrasp")
        return X_WGoal

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
        q_samples = []

        for k in range(max_iterations):
            # print("k: ", k)
            q_sample = rrt_tools.sample_node_in_configuration_space()
            q_samples.append(q_sample)
            random_num = random()
            if random_num < prob_sample_q_goal:
                print("---------------")
                q_sample = q_goal
                n_near = rrt_tools.find_nearest_node_in_RRT_graph(q_sample)
                # print("n_near: ", n_near.value)
                intermediate_q = rrt_tools.calc_intermediate_qs_wo_collision(n_near.value, q_sample)
                # print("Intermadiate q: ", intermediate_q)
                last_node = n_near
                # self.display_q_samples(q_samples)
                for n in range(len(intermediate_q)):

                    last_node = rrt_tools.grow_rrt_tree(last_node, intermediate_q[n])
                    if rrt_tools.node_reaches_goal(last_node):
                        path = rrt_tools.backup_path_from_node(last_node)
                        self.plot_ee_trajectory(path, q_samples)
                        return path

        return None
    
    def display_q_samples(self, q_samples):
        # print("Q samples: ", len(q_samples))
        # Step 1: Copy the plant and create a new context
        plant = self.iiwa.plant.Clone()
        context = plant.CreateDefaultContext()

        # Step 2: Get the end-effector frame
        ee_frame = plant.GetFrameByName(GRIPPER_BASE_LINK)

        # Step 4: Compute EE positions for q_samples (smaller points)
        ee_sample_positions = []
        for joint_values in q_samples:
            plant.SetPositions(context, joint_values)
            X_WF = ee_frame.CalcPoseInWorld(context)
            ee_sample_positions.append(X_WF.translation())
        ee_sample_positions = np.array(ee_sample_positions)

        # Step 6: Display q_samples (as small individual points)
        if ee_sample_positions.size > 0:
            point_cloud = PointCloud(ee_sample_positions.shape[0])  # N points
            point_cloud.mutable_xyzs()[:] = ee_sample_positions.T  # Transpose to (3, N)
            self.playground.meshcat.SetObject(
                path="samples_ee",
                cloud=point_cloud,
                point_size=0.01,  # Smaller size to be subtle
                rgba=Rgba(0.5, 0.5, 0.5, 1.0)  # Gray color
            )
    
    def record_ee_position(self, plant_context):
        frame = self.iiwa.plant.GetFrameByName(self.gripper_frame_name)
        X_WF = frame.CalcPoseInWorld(plant_context)  # Get world pose of the frame
        position = X_WF.translation().reshape(3, 1)  # Ensure it's a column vector (3x1)

        # Concatenate along axis=1 to maintain 3xN shape
        self.ee_path = np.concatenate((self.ee_path, position), axis=1)
    
    def plot_ee_trajectory(self, joint_paths, q_samples):
        """
        Copies the plant and computes the end-effector trajectory in 3D space.

        Args:
            original_plant (MultibodyPlant): The original plant.
            ee_frame_name (str): Name of the end-effector frame.
            joint_paths (list of list of float): List of joint configurations.
        """
        print("Q samples: ", len(q_samples))
        # Step 1: Copy the plant and create a new context
        plant = self.iiwa.plant.Clone()
        context = plant.CreateDefaultContext()

        # Step 2: Get the end-effector frame
        ee_frame = plant.GetFrameByName(GRIPPER_BASE_LINK)

        # Step 3: Compute EE position for each joint configuration
        ee_positions = []  # Store EE positions

        for joint_values in joint_paths:
            # Set joint positions in the context
            plant.SetPositions(context, joint_values)

            # Get the EE pose in world frame
            X_WF = ee_frame.CalcPoseInWorld(context)  # RigidTransform
            ee_positions.append(X_WF.translation())

        # Step 4: Compute EE positions for q_samples (smaller points)
        ee_sample_positions = []
        for joint_values in q_samples:
            plant.SetPositions(context, joint_values)
            X_WF = ee_frame.CalcPoseInWorld(context)
            ee_sample_positions.append(X_WF.translation())
        ee_sample_positions = np.array(ee_sample_positions)

        ee_positions = np.array(ee_positions)  # Convert to NumPy array

        # Step 5: Display main trajectory (as a connected line)
        self.playground.meshcat.SetLine(
            path="path_ee_rrt", 
            vertices=ee_positions.T,  # Meshcat expects 3xN format
            rgba=Rgba(1.0, 0., 0., 1.0),  # Blue color for main trajectory
            line_width=3.0
        )

        # Step 6: Display q_samples (as small individual points)
        if ee_sample_positions.size > 0:
            point_cloud = PointCloud(ee_sample_positions.shape[0])  # N points
            point_cloud.mutable_xyzs()[:] = ee_sample_positions.T  # Transpose to (3, N)
            self.playground.meshcat.SetObject(
                path="samples_ee",
                cloud=point_cloud,
                point_size=0.01,  # Smaller size to be subtle
                rgba=Rgba(0.5, 0.5, 0.5, 1.0)  # Gray color
            )
        
class Iiwa_Problem(Problem):
    def __init__(self, iiwa: IIWA, query_object, plant_context, q_start: np.array, q_goal: np.array):
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
            #     range_list.append(Range(0, 2*np.pi, wrap_around=True))xcee
            # else:
            range_list.append(Range(lb[i], ub[i]))
        max_steps = 3 * [np.pi / 180]  # three degrees
        cspace_iiwa = ConfigurationSpace(range_list, l2_distance, max_steps)

        # override 
        Problem.__init__(self,
                           x=5,
                           y=5,
                           robot=None,
                           obstacles=None,
                           start=tuple(q_start),
                           goal=tuple(q_goal),
                           region=AABB(Point(-3, -3), Point(3, 3)),
                           cspace=cspace_iiwa,
                           display_tree=False)
        
    
    def collide(self, configuration):
        q = np.array(configuration)
        # print("    Collide function")
        return self.exists_collision(q)
    
    def exists_collision(self, q):
        # plant = self.iiwa.plant.Clone()
        # context = plant.CreateDefaultContext()
        self.iiwa.plant.SetPositions(self.plant_context, q)

        query_object = self.iiwa.plant.get_geometry_query_input_port().Eval(self.plant_context)
        inspector = query_object.inspector()
        collision_pairs = inspector.GetCollisionCandidates()
        for pair in collision_pairs:
            val = query_object.ComputeSignedDistancePairClosestPoints(pair[0], pair[1]).distance, inspector.GetName(inspector.GetFrameId(pair[0])), inspector.GetName(inspector.GetFrameId(pair[1]))
            if val[0] <= 0.:
                return True
        return False

    # def exists_collision(self):
    #     query_object = self.iiwa.plant.get_geometry_query_input_port().Eval(self.plant_context)
    #     inspector = query_object.inspector()
    #     collision_pairs = inspector.GetCollisionCandidates()
    #     for pair in collision_pairs:
    #         val = query_object.ComputeSignedDistancePairClosestPoints(pair[0], pair[1]).distance, inspector.GetName(inspector.GetFrameId(pair[0])), inspector.GetName(inspector.GetFrameId(pair[1]))
    #         if val[0] <= 0.:
    #             return True
    #     return False
