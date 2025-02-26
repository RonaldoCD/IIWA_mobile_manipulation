import multiprocessing as mp
import os.path
import time
from collections import OrderedDict
from typing import Dict

import numpy as np
import pydot
from IPython.display import SVG, display
from pydrake.common.value import AbstractValue
from pydrake.geometry import (
    Meshcat,
    MeshcatVisualizer,
    QueryObject,
    Rgba,
    Role,
    SceneGraph,
    Sphere,
    StartMeshcat,
)
from pydrake.geometry.optimization import (
    HPolyhedron,
    IrisInConfigurationSpace,
    IrisOptions,
    LoadIrisRegionsYamlFile,
    SaveIrisRegionsYamlFile,
)
from pydrake.math import RigidTransform, RollPitchYaw, RotationMatrix
from pydrake.multibody.inverse_kinematics import InverseKinematics
from pydrake.multibody.meshcat import JointSliders
from pydrake.multibody.parsing import PackageMap, Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, MultibodyPlant
from pydrake.multibody.tree import Body
from pydrake.solvers import MathematicalProgram, Solve
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.visualization import AddDefaultVisualization, ModelVisualizer

from manipulation import running_as_notebook
from manipulation.utils import FindDataResource
from robots.iiwa import IIWA
from generate_models import Playground


iris_filename = "my_iris.yaml"
# iris_filename = FindDataResource("iiwa_shelve_and_bins_science_robotics.yaml")
# iris_regions = dict()
# q = []

iris_options = IrisOptions()
iris_options.iteration_limit = 10
# increase num_collision_infeasible_samples to improve the (probabilistic)
# certificate of having no collisions.
iris_options.num_collision_infeasible_samples = 3
iris_options.require_sample_point_is_contained = True
iris_options.relative_termination_threshold = 0.01
iris_options.termination_threshold = -1

# Additional options for this notebook:

# If use_existing_regions_as_obstacles is True, then iris_regions will be
# shrunk by regions_as_obstacles_margin, and then passed to
# iris_options.configuration_obstacles.
use_existing_regions_as_obstacles = True
regions_as_obstacles_scale_factor = 0.95

# We can compute some regions in parallel.
num_parallel = mp.cpu_count()

def ScaleHPolyhedron(hpoly, scale_factor):
    # Shift to the center.
    xc = hpoly.ChebyshevCenter()
    A = hpoly.A()
    b = hpoly.b() - A @ xc
    # Scale
    b = scale_factor * b
    # Shift back
    b = b + A @ xc
    return HPolyhedron(A, b)


def _CheckNonEmpty(region):
    prog = MathematicalProgram()
    x = prog.NewContinuousVariables(region.ambient_dimension())
    region.AddPointInSetConstraints(prog, x)
    result = Solve(prog)
    assert result.is_success()


def _CalcRegion(name, seed):
    # builder = DiagramBuilder()
    # plant = AddMultibodyPlantSceneGraph(builder, 0.0)[0]
    # LoadRobot(plant)
    # plant.Finalize()
    playground = Playground(meshcat=meshcat, time_step=0.001)
    iiwa = IIWA(playground.construct_welded_sim(playground.default_continuous_state()))
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyContextFromRoot(diagram_context)
    plant.SetPositions(plant_context, seed)
    if use_existing_regions_as_obstacles:
        iris_options.configuration_obstacles = [
            ScaleHPolyhedron(r, regions_as_obstacles_scale_factor)
            for k, r in iris_regions.items()
            if k != name
        ]
        for h in iris_options.configuration_obstacles:
            _CheckNonEmpty(h)
    else:
        iris_options.configuration_obstacles = None
    display(f"Computing region for seed: {name}")
    start_time = time.time()
    hpoly = IrisInConfigurationSpace(plant, plant_context, iris_options)
    display(
        f"Finished seed {name}; Computation time: {(time.time() - start_time):.2f} seconds"
    )

    _CheckNonEmpty(hpoly)
    reduced = hpoly.ReduceInequalities()
    _CheckNonEmpty(reduced)

    return reduced


def GenerateRegion(name, seed):
    global iris_regions
    iris_regions[name] = _CalcRegion(name, seed)
    SaveIrisRegionsYamlFile(f"{iris_filename}.autosave", iris_regions)


def GenerateRegions(seed_dict, verbose=True):
    if use_existing_regions_as_obstacles:
        # Then run serially
        for k, v in seed_dict.items():
            GenerateRegion(k, v)
        return

    loop_time = time.time()
    with mp.Pool(processes=num_parallel) as pool:
        new_regions = pool.starmap(_CalcRegion, [[k, v] for k, v in seed_dict.items()])

    if verbose:
        print("Loop time:", time.time() - loop_time)

    global iris_regions
    iris_regions.update(dict(list(zip(seed_dict.keys(), new_regions))))


def DrawRobot(query_object: QueryObject, meshcat_prefix: str, draw_world: bool = True):
    rgba = Rgba(0.7, 0.7, 0.7, 0.3)
    role = Role.kProximity
    # This is a minimal replication of the work done in MeshcatVisualizer.
    inspector = query_object.inspector()
    for frame_id in inspector.GetAllFrameIds():
        if frame_id == inspector.world_frame_id():
            if not draw_world:
                continue
            frame_path = meshcat_prefix
        else:
            frame_path = f"{meshcat_prefix}/{inspector.GetName(frame_id)}"
        frame_path.replace("::", "/")
        frame_has_any_geometry = False
        for geom_id in inspector.GetGeometries(frame_id, role):
            path = f"{frame_path}/{geom_id.get_value()}"
            path.replace("::", "/")
            meshcat.SetObject(path, inspector.GetShape(geom_id), rgba)
            meshcat.SetTransform(path, inspector.GetPoseInFrame(geom_id))
            frame_has_any_geometry = True

        if frame_has_any_geometry:
            X_WF = query_object.GetPoseInWorld(frame_id)
            meshcat.SetTransform(frame_path, X_WF)


def VisualizeRegion(region_name, num_to_draw=30, draw_illustration_role_once=True):
    """
    A simple hit-and-run-style idea for visualizing the IRIS regions:
    1. Start at the center. Pick a random direction and run to the boundary.
    2. Pick a new random direction; project it onto the current boundary, and run along it. Repeat
    """

    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    LoadRobot(plant)
    plant.Finalize()
    if draw_illustration_role_once:
        MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)
    scene_graph_context = scene_graph.GetMyContextFromRoot(context)

    global iris_regions
    region = iris_regions[region_name]

    q = region.ChebyshevCenter()
    plant.SetPositions(plant_context, q)
    diagram.ForcedPublish(context)

    query = scene_graph.get_query_output_port().Eval(scene_graph_context)
    DrawRobot(query, f"{region_name}/0", True)

    rng = np.random.default_rng()
    nq = plant.num_positions()
    prog = MathematicalProgram()
    qvar = prog.NewContinuousVariables(nq, "q")
    prog.AddLinearConstraint(region.A(), 0 * region.b() - np.inf, region.b(), qvar)
    cost = prog.AddLinearCost(np.ones((nq, 1)), qvar)

    for i in range(1, num_to_draw):
        direction = rng.standard_normal(nq)
        cost.evaluator().UpdateCoefficients(direction)

        result = Solve(prog)
        assert result.is_success()

        q = result.GetSolution(qvar)
        plant.SetPositions(plant_context, q)
        query = scene_graph.get_query_output_port().Eval(scene_graph_context)
        DrawRobot(query, f"{region_name}/{i}", False)


def VisualizeRegions():
    for k in iris_regions.keys():
        meshcat.Delete()
        VisualizeRegion(k)
        button_name = f"Visualizing {k}; Press for next region"
        meshcat.AddButton(button_name, "Enter")
        print("Press Enter to visualize the next region")
        while meshcat.GetButtonClicks(button_name) < 1:
            time.sleep(1.0)
        meshcat.DeleteButton(button_name)


# TODO(russt): See https://github.com/RobotLocomotion/drake/pull/19520
class PoseSelector(LeafSystem):
    def __init__(
        self,
        body_index=None,
    ):
        LeafSystem.__init__(self)
        self._body_index = body_index
        self.DeclareAbstractInputPort(
            "body_poses", AbstractValue.Make([RigidTransform()])
        )
        self.DeclareAbstractOutputPort(
            "pose",
            lambda: AbstractValue.Make(RigidTransform()),
            self.CalcOutput,
        )

    def CalcOutput(self, context, output):
        body_poses = self.get_input_port().Eval(context)
        output.set_value(body_poses[self._body_index])


## DEFINE ROBOT ENVIRONMENT

def LoadRobot(plant: MultibodyPlant) -> Body:
    """Setup your plant, and return the body corresponding to your
    end-effector."""
    parser = Parser(plant)

    model_directives = """    
directives:

# Add iiwa
- add_model:
    name: iiwa
    file: package://drake_models/iiwa_description/urdf/iiwa14_primitive_collision.urdf
    default_joint_positions:
        iiwa_joint_1: [0]
        iiwa_joint_2: [0.3]
        iiwa_joint_3: [0]
        iiwa_joint_4: [-1.8]
        iiwa_joint_5: [0]
        iiwa_joint_6: [1]
        iiwa_joint_7: [1.57]

- add_weld:
    parent: world
    child: iiwa::base

# Add schunk
- add_model:
    name: wsg
    file: package://drake_models/wsg_50_description/sdf/schunk_wsg_50_welded_fingers.sdf

- add_weld:
    parent: iiwa::iiwa_link_7
    child: wsg::body
    X_PC:
      translation: [0, 0, 0.114]
      rotation: !Rpy { deg: [90.0, 0.0, 0.0 ]}

# Add shelves
- add_model:
    name: shelves
    file: package://gcs/models/shelves/shelves.sdf

- add_weld:
    parent: world
    child: shelves::shelves_body
    X_PC:
      translation: [0.85, 0, 0.4]

# Add Bins
- add_model:
    name: binR
    file: package://gcs/models/bin/bin.sdf

- add_weld:
    parent: world
    child: binR::bin_base
    X_PC:
      translation: [0, -0.6, 0]
      rotation: !Rpy { deg: [0.0, 0.0, 90.0 ]}

- add_model:
    name: binL
    file: package://gcs/models/bin/bin.sdf

- add_weld:
    parent: world
    child: binL::bin_base
    X_PC:
      translation: [0, 0.6, 0]
      rotation: !Rpy { deg: [0.0, 0.0, 90.0 ]}

# Add table
- add_model:
    name: table
    file: package://gcs/models/table/table_wide.sdf

- add_weld:
    parent: world
    child: table::table_body
    X_PC:
      translation: [0.4, 0.0, 0.0]
      rotation: !Rpy { deg: [0., 0., 00]}
"""

    parser.AddModelsFromString(model_directives, ".dmd.yaml")
    brick = parser.AddModelsFromUrl(
        "package://drake_models/manipulation_station/061_foam_brick.sdf"
    )[0]
    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("base_link", brick),
        RigidTransform([0.88, 0, 0.62]),
    )
    gripper = plant.GetModelInstanceByName("wsg")
    end_effector_body = plant.GetBodyByName("body", gripper)
    return end_effector_body


# Sometimes it's useful to use inverse kinematics to find the seeds. You might
# need to adapt this to your robot. This helper takes an end-effector frame, E,
# and a desired pose for that frame in the world coordinates, X_WE.
def MyInverseKinematics(X_WE, plant=None, context=None):
    if not plant:
        plant = MultibodyPlant(0.0)
        LoadRobot(plant)
        plant.Finalize()
    if not context:
        context = plant.CreateDefaultContext()
    # E = ee_body.body_frame()
    E = plant.GetBodyByName("body").body_frame()

    ik = InverseKinematics(plant, context)

    ik.AddPositionConstraint(
        E, [0, 0, 0], plant.world_frame(), X_WE.translation(), X_WE.translation()
    )

    ik.AddOrientationConstraint(
        E, RotationMatrix(), plant.world_frame(), X_WE.rotation(), 0.001
    )

    prog = ik.get_mutable_prog()
    q = ik.q()

    q0 = plant.GetPositions(context)
    prog.AddQuadraticErrorCost(np.identity(len(q)), q0, q)
    prog.SetInitialGuess(q, q0)
    result = Solve(ik.prog())
    if not result.is_success():
        print("IK failed")
        return None
    plant.SetPositions(context, result.GetSolution(q))
    return result.GetSolution(q)

def CollisionGeometryReport(iiwa_plant, iiwa_diagram):
    context = iiwa_diagram.CreateDefaultContext()
    plant_context = iiwa_plant.GetMyContextFromRoot(context)

    query_object = iiwa_plant.get_geometry_query_input_port().Eval(plant_context)
    inspector = query_object.inspector()
    pairs = inspector.GetCollisionCandidates()
    for geomA, geomB in pairs:
        frameA = inspector.GetFrameId(geomA)
        frameB = inspector.GetFrameId(geomB)
        print(
            f"{inspector.GetName(geomA)} (in {inspector.GetName(frameA)}) + {inspector.GetName(geomB)} (in {inspector.GetName(frameB)})"
        )
