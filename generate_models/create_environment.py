import warnings
from typing import List

import numpy as np

from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, GeometrySet, Role, CollisionFilterDeclaration
from pydrake.all import (
    AddMultibodyPlantSceneGraph, 
    Parser, 
    RigidTransform, 
    ConstantVectorSource, 
    SharedPointerSystem, 
    SchunkWsgPositionController, 
    MakeMultibodyStateToWsgStateSystem
)
from pydrake.multibody.tree import Body
from pydrake.systems.controllers import InverseDynamicsController
from pydrake.systems.framework import DiagramBuilder, Diagram, Context
from pydrake.systems.primitives import PassThrough, Demultiplexer, Adder, \
    StateInterpolatorWithDiscreteDerivative, LogVectorOutput, Multiplexer, ZeroOrderHold
from pydrake.visualization import ApplyVisualizationConfig, VisualizationConfig

from environment.fixed_body import FixedBody, Shelve
from settings import DEFAULT_ENV_URL, IIWA_MODEL_URL, IIWA_MAIN_LINK, GRIPPER_MODEL_URL, DEFAULT_SCENARIO_URL, PACKAGE_XML_PATH
from pydrake.all import MultibodyPlant

# from .create_environment_files import create_environment_files
from environment.movable_body import MovableBody
from package_utils import CustomConfigureParser, GetPathFromUrl
from environment import Environment
from .parameters import SHELVE_BODY_NAME, SHELF_FLOORS, SHELF_THICKNESS, SHELF_DEPTH, SHELF_WIDTH, SHELF_HEIGHT
from pydrake.manipulation import SimIiwaDriver
from pydrake.manipulation import IiwaControlMode, ParseIiwaControlMode
from manipulation.station import LoadScenario, MakeHardwareStation, MakeMultibodyPlant
from manipulation.utils import ConfigureParser


def fixIiwaGripperCollisionWithObjectInGripper(scene_graph, body_name, context = None):
    # todo later maybe generalize this instead of copying it from utils?
    if context is None:
        filter_manager = scene_graph.collision_filter_manager()
    else:
        filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()
    iiwa = {}
    exclude_bodies = []
    # print("body_name: ", body_name)
    for gid in inspector.GetGeometryIds(
            GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        # print("Name: ", gid_name)
        if "iiwa" in gid_name:
            link_name = gid_name.split("::")[1]
            # print("IIWA Link name: ", link_name)
            iiwa[link_name] = [gid]
        elif "wsg" in gid_name:
            link_name = gid_name.split("::")[1]
            # print("WSG Link name: ", link_name)
            iiwa[link_name] = [gid]
            if len(iiwa[link_name]) > 0:
                iiwa[link_name].append(gid)
        if body_name in gid_name:
            # print("Body name: ", body_name)
            exclude_bodies.append(gid)

    # print('exclude bodies:')
    # print(exclude_bodies)

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # todo this excludes both hands. what if we want to exclude only one gripper?
    add_exclusion(exclude_bodies,
                  iiwa["iiwa_link_7"] +  # Last joint before end-effector
                  iiwa["iiwa_link_6"] +
                  iiwa["iiwa_link_5"] +
                  iiwa["iiwa_link_4"] +
                  iiwa["body"] +
                  iiwa["left_finger"] +
                  iiwa["right_finger"]
    )

def filterIiwaCollisionGeometry(scene_graph, context=None):
    """
    Filters out self-collision issues for the IIWA robot in Drake.
    Specifically removes false-positive collisions between iiwa_link_5 and iiwa_link_7.
    """
    if context is None:
        filter_manager = scene_graph.collision_filter_manager()
    else:
        filter_manager = scene_graph.collision_filter_manager(context)
    inspector = scene_graph.model_inspector()

    iiwa = {}

    # Retrieve all IIWA geometries
    for gid in inspector.GetGeometryIds(
        GeometrySet(inspector.GetAllGeometryIds()), Role.kProximity
    ):
        gid_name = inspector.GetName(inspector.GetFrameId(gid))
        if "iiwa" in gid_name:
            link_name = gid_name.split("::")[1]  # Extract the link name
            # print("link_name: ", link_name)
            iiwa[link_name] = [gid]

    def add_exclusion(set1, set2=None):
        if set2 is None:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeWithin(GeometrySet(set1))
            )
        else:
            filter_manager.Apply(
                CollisionFilterDeclaration().ExcludeBetween(
                    GeometrySet(set1), GeometrySet(set2)
                )
            )

    # Exclude collisions between iiwa_link_5 and iiwa_link_7
    add_exclusion(iiwa["iiwa_link_5"], iiwa["iiwa_link_7"])

    # Exclude adjacent links to prevent unnecessary self-collisions
    add_exclusion(iiwa["iiwa_link_1"], iiwa["iiwa_link_2"])
    add_exclusion(iiwa["iiwa_link_2"], iiwa["iiwa_link_3"])
    add_exclusion(iiwa["iiwa_link_3"], iiwa["iiwa_link_4"])
    add_exclusion(iiwa["iiwa_link_4"], iiwa["iiwa_link_5"])
    add_exclusion(iiwa["iiwa_link_5"], iiwa["iiwa_link_6"])
    add_exclusion(iiwa["iiwa_link_6"], iiwa["iiwa_link_7"])

    # Exclude collisions between the gripper and the wrist (if applicable)
    if "wsg" in iiwa:
        add_exclusion(iiwa["iiwa_link_7"], iiwa["wsg"])

    print("IIWA self-collision filtering applied.")

def CustomParser(plant):
    parser = Parser(plant)
    return CustomConfigureParser(parser)

def GetIiwaNominalPosition():
    default_positions = [0.5, 0.6, 0, -1.75, 0, 1.0, 0, -0.05, 0.05]
    return np.array(default_positions).astype(np.float64)

def AddIiwaPlant(plant):
    iiwa = CustomParser(plant).AddModelsFromUrl(IIWA_MODEL_URL)[0]
    # plant.WeldFrames(plant.world_frame(), plant.GetBodyByName(IIWA_MAIN_LINK).body_frame())
    return iiwa

def AddIiwa(builder, plant: MultibodyPlant, scene_graph):
    iiwa = AddIiwaPlant(plant)
    # filterPR2CollsionGeometry(scene_graph)
    return iiwa
    
def SetDefaultIiwaNominalPosition(plant, robot_id):
    default_context = plant.CreateDefaultContext()
    default_positions = GetIiwaNominalPosition()
    ##iiwa
    plant.SetPositions(default_context, robot_id[0], default_positions[:7])
    plant.SetDefaultPositions(robot_id[0], default_positions[:7])
    ##wsg
    plant.SetPositions(default_context, robot_id[1], default_positions[7:])
    plant.SetDefaultPositions(robot_id[1], default_positions[7:])

# Function without point cloud
def GetAllMovableBodies(plant, yaml_file):
    default_context = plant.CreateDefaultContext()
    filename = "/home/ronaldocd/Desktop/IMARO/stage/Project/models/main_environment.dmd.yaml"
    directives = LoadScenario(filename=filename).directives
    # print("Directives: ", directives)
    model_name_to_url = {d.add_model.name: d.add_model.file for d in directives if d.add_model}

    movable_bodies = []

    for body_index in plant.GetFloatingBaseBodies():
        body = plant.get_body(body_index)
        model_name = plant.GetModelInstanceName(body.model_instance())
        # model_url = model_name_to_url[model_name]

        origin_frame = plant.GetFrameByName(model_name + "_origin")
        final_frame = plant.GetFrameByName(model_name + "_destination")

        X_WO_init = origin_frame.CalcPoseInWorld(default_context)
        pregrasp_space = np.array([0., -0.4, 0.])
        # R0 = RotationMatrix(np.array([[0, 1, 0], [0, 0, -1], [-1, 0, 0]]).T)
        ## it is possible to implement a function that gets the grasp frame
        X_WPre_Grasp = RigidTransform(X_WO_init.translation() + pregrasp_space)
        
        # print("Movable Body: ", body.name())
        movable_bodies.append(MovableBody(
            body_name=body.name(),
            model_name=model_name,
            X_WO_init=origin_frame.CalcPoseInWorld(default_context),
            X_WO_end=final_frame.CalcPoseInWorld(default_context),
            X_WPre_Grasp=X_WPre_Grasp
        ))

    return movable_bodies

def GetAllFixedBodies(plant: MultibodyPlant) -> List[FixedBody]:
    default_context = plant.CreateDefaultContext()

    # todo this is the hacky way. just take all the objects we want by name...
    fixed_bodies = []
    for body in plant.GetBodiesWeldedTo(plant.world_body()):
        # todo using SHELVE_DEPTH, ... is bad because then we have to regenerate the models all the time if we change those parameters...
        # technical debt
        # print("body name: ", body.name())
        if body.name() != SHELVE_BODY_NAME:
            continue
        body: Body = body
        # print("Fixed Body: ", body.name())
        fixed_bodies.append(Shelve(
            body_name=body.name(),
            model_name=plant.GetModelInstanceName(body.model_instance()),
            depth=SHELF_DEPTH,
            width=SHELF_WIDTH,
            height=SHELF_HEIGHT,
            floors=SHELF_FLOORS,
            thickness=SHELF_THICKNESS,
            normal_direction=body.body_frame().CalcPoseInWorld(default_context).rotation() @ [1, 0, 0],
            center_pos=body.body_frame().CalcPoseInWorld(default_context).translation()
        ))
    return fixed_bodies

def AddThingsDefaultPos(plant, movable_bodies: List[MovableBody]):
    for mb in movable_bodies:
        plant.SetDefaultFreeBodyPose(mb.get_body(plant), mb.X_WO_init)

def fix_order(builder, port, cur_order, final_order):
    n = len(cur_order)
    assert (n == len(final_order))
    assert (n == port.size())

    def core_name(name: str):
        name = name.removeprefix("control_")
        name = name.removeprefix("iiwa_")
        name = name.removesuffix("_q")
        name = name.removesuffix("_x")
        name = name.removesuffix("_y")
        name = name.removesuffix("_joint")
        return name

    final_order = [core_name(name) for name in final_order]
    cur_order = [core_name(name) for name in cur_order]

    final_order_map = {name: i for i, name in enumerate(final_order)}
    assert (len(final_order_map) == n)
    for name in cur_order:
        assert (name in final_order_map)

    demux = builder.AddSystem(Demultiplexer(n, 1))
    mux = builder.AddSystem(Multiplexer(n))
    builder.Connect(port, demux.get_input_port())
    for i, name in enumerate(cur_order):
        builder.Connect(
            demux.get_output_port(i),
            mux.get_input_port(final_order_map[name])
        )
    return mux.get_output_port()

def CreateIiwaControllerPlant(scenario_data):
    """creates plant that includes only the robot and gripper, used for controllers."""
    filename = "/home/ronaldocd/Desktop/IMARO/stage/Project/models/main_environment.dmd.yaml"
    scenario = LoadScenario(filename=filename)
    # print("Create iiwa: ", scenario.directives)
    plant_robot = MakeMultibodyPlant(
        scenario=scenario, model_instance_names=["iiwa"]
    )

    # print("Number of positions:", plant_robot.num_positions())
    
    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant_robot.GetFrameByName("iiwa_link_" + str(i)).index()
        )

    return plant_robot, link_frame_indices

def AddIiwaDriver(builder, plant: MultibodyPlant, models, scenario_data):
    model_instance_name_iiwa = models[0]
    model_instance = plant.GetModelInstanceByName(model_instance_name_iiwa)
    num_iiwa_positions = plant.num_positions(model_instance)

    # Make the plant for the iiwa controller to use.
    controller_plant, _ = CreateIiwaControllerPlant(scenario_data)
    # Keep the controller plant alive during the Diagram lifespan.
    builder.AddNamedSystem(
        f"{model_instance_name_iiwa}_controller_plant_pointer_system",
        SharedPointerSystem(controller_plant),
    )

    control_mode = ParseIiwaControlMode("position_only")
    sim_iiwa_driver = SimIiwaDriver.AddToBuilder(
        plant=plant,
        iiwa_instance=model_instance,
        controller_plant=controller_plant,
        builder=builder,
        ext_joint_filter_tau=0.01,
        desired_iiwa_kp_gains=np.full(num_iiwa_positions, 100),
        control_mode=control_mode,
    )
    # iiwa_position.get_output_port()
    # demux = builder.AddSystem(
    #     Multiplexer(input_sizes=[7, 7])
    # )

    for i in range(sim_iiwa_driver.num_input_ports()):
        port = sim_iiwa_driver.get_input_port(i)
        # print(f"Input Port {i}: {port.get_name()}") 
        if not builder.IsConnectedOrExported(port):
            builder.ExportInput(port, f"{model_instance_name_iiwa}.{port.get_name()}")
            # builder.ExportInput(port, "zeros")
    for i in range(sim_iiwa_driver.num_output_ports()):
        port = sim_iiwa_driver.get_output_port(i)
        # print(f"Output Port {i}: {port.get_name()}")
        builder.ExportOutput(port, f"{model_instance_name_iiwa}.{port.get_name()}")

    ## WSG
    model_instance_name_wsg = models[1]
    model_instance = plant.GetModelInstanceByName(model_instance_name_wsg)
    # Wsg controller.
    wsg_controller = builder.AddSystem(SchunkWsgPositionController())
    wsg_controller.set_name(model_instance_name_wsg + ".controller")
    builder.Connect(
        wsg_controller.get_generalized_force_output_port(),
        plant.get_actuation_input_port(model_instance),
    )
    builder.Connect(
        plant.get_state_output_port(model_instance),
        wsg_controller.get_state_input_port(),
    )
    builder.ExportInput(
        wsg_controller.get_desired_position_input_port(),
        model_instance_name_wsg + ".position",
    )
    builder.ExportInput(
        wsg_controller.get_force_limit_input_port(),
        model_instance_name_wsg + ".force_limit",
    )
    wsg_mbp_state_to_wsg_state = builder.AddSystem(
        MakeMultibodyStateToWsgStateSystem()
    )
    builder.Connect(
        plant.get_state_output_port(model_instance),
        wsg_mbp_state_to_wsg_state.get_input_port(),
    )
    builder.ExportOutput(
        wsg_mbp_state_to_wsg_state.get_output_port(),
        model_instance_name_wsg + ".state_measured",
    )
    builder.ExportOutput(
        wsg_controller.get_grip_force_output_port(),
        model_instance_name_wsg + ".force_measured",
    )
    ##


class Playground:
    def __init__(self,
                 meshcat,
                 scenario_data=DEFAULT_SCENARIO_URL,
                 time_step=0.001,
                 visualization_config=VisualizationConfig()
    ):
        # meshcat.Delete()
        builder = DiagramBuilder()

        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
        plant: MultibodyPlant = plant # what's the point of this line?

        parser = CustomParser(plant)
        models_id = parser.AddModelsFromUrl(scenario_data)
        robot_id = [instance_index for instance_index in models_id
            if plant.GetModelInstanceName(instance_index) in ["iiwa", "wsg"]]

        # for instance_index in model_iiwa_wsg_id:
        #     model_name = plant.GetModelInstanceName(instance_index)
        #     print(f"Model Instance Index: {instance_index}, Name: {model_name}") 

        plant.set_name("plant")
        filterIiwaCollisionGeometry(scene_graph)
        plant.Finalize()

        SetDefaultIiwaNominalPosition(plant, robot_id)
            
        plant_context = plant.CreateDefaultContext()
        print("Plant init: ", plant.GetPositions(plant_context))
        
        movable_bodies = GetAllMovableBodies(plant, scenario_data)
        fixed_bodies = GetAllFixedBodies(plant)

        AddThingsDefaultPos(plant, movable_bodies)
        AddIiwaDriver(builder, plant, ["iiwa", "wsg"], scenario_data)

        self._add_visuals(builder, scene_graph, meshcat, visualization_config)

        # # Export "cheat" ports.
        builder.ExportOutput(scene_graph.get_query_output_port(), "query_object")
        builder.ExportOutput(
            plant.get_contact_results_output_port(), "contact_results"
        )
        builder.ExportOutput(
            plant.get_state_output_port(), "plant_continuous_state"
        )
        builder.ExportOutput(plant.get_body_poses_output_port(), "body_poses")

        diagram = builder.Build()
        diagram.set_name("environment")
        self.diagram = diagram
        self.meshcat = meshcat
        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            sim_plant_context = env.plant.GetMyContextFromRoot(sim_context)
            env.plant.SetPositionsAndVelocities(sim_plant_context, continuous_state)
            return sim_context
        self.env = Environment(diagram=diagram,
                               plant=plant,
                               scene_graph=scene_graph,
                               models_id=robot_id,
                               floating_bodies=[plant.get_body(body_index) for body_index in plant.GetFloatingBaseBodies()],
                               context_update_function=context_update_function,
                               movable_bodies=movable_bodies,
                               fixed_bodies=fixed_bodies
        )
        self.scenario_data = scenario_data
        self.time_step = time_step

    def default_continuous_state(self):
        default_plant_context = self.env.plant.CreateDefaultContext()
        return self.env.plant.GetPositionsAndVelocities(default_plant_context)

    def construct_welded_sim(self, continuous_state, modify_default_context=True, meshcat=None):
        """
        constructs a plant and scene_graph that will be used by robot's controller
        the idea is that the plant that simulates the world should be different from
        the plant that robot uses to solve trajectory optimization...

        In this new plant we include objects but weld all of them so that
        dof plant == dof robot

        pitfall. This does not work if we have non single rigid body objects (like a chain)
        if you want to support that we have to somehow fix all the joints possible
        """

        sim_builder = DiagramBuilder()

        sim_plant, sim_scene_graph = AddMultibodyPlantSceneGraph(sim_builder, time_step=self.time_step)
        sim_plant: MultibodyPlant = sim_plant # what's the point of this line?

        sim_parser = CustomParser(sim_plant)
        
        models_id = sim_parser.AddModelsFromUrl(self.scenario_data)
        filterIiwaCollisionGeometry(sim_scene_graph)
        sim_robot_id = [instance_index for instance_index in models_id
            if sim_plant.GetModelInstanceName(instance_index) in ["iiwa", "wsg"]]

        # for instance_index in robot_id:
        #     model_name = sim_plant.GetModelInstanceName(instance_index)
        #     print(f"Model Instance Index: {instance_index}, Name: {model_name}") 

        sim_plant.set_name("plant")
        sim_plant_context = self.env.plant.CreateDefaultContext()
        
        # weld everything instead of adding default pos
        """
            todo for the future have the option to weld the object that robot is
            holding to the robot's arm
        """
        self.env.plant.SetPositionsAndVelocities(sim_plant_context, continuous_state)

        # weld everything to the world
        floating_bodies = self._weld_all_floating_bodies(sim_plant, sim_plant_context)
        sim_plant.Finalize()
        # sim_plant_contextv2 = sim_plant.CreateDefaultContext()
        
        # # print("Plant welded: ", sim_plant.GetPositions(sim_plant_context))
        # print("Plant welded: ", sim_plant.GetPositions(sim_plant_contextv2))
        SetDefaultIiwaNominalPosition(sim_plant, sim_robot_id)

        if modify_default_context:
            # set default pos of robot to current position
            # sim_default_context = sim_plant.CreateDefaultContext()
            positions_iiwa = self.env.plant.GetPositionsFromArray(
                model_instance=self.env.models_id[0],
                q=self.env.plant.GetPositions(sim_plant_context))
            
            positions_wsg = self.env.plant.GetPositionsFromArray(
                model_instance=self.env.models_id[1],
                q=self.env.plant.GetPositions(sim_plant_context))
            # velocities = self.env.plant.GetVelocitiesFromArray(
            #     model_instance=self.env.model_id,
            #     v=self.env.plant.GetVelocities(plant_context))
            # todo how to set default velocity as well?
            positions = np.concatenate([positions_iiwa, positions_wsg])
            sim_plant.SetDefaultPositions(positions)

        AddIiwaDriver(sim_builder, sim_plant, ["iiwa", "wsg"], self.scenario_data)
        self._add_visuals(sim_builder, sim_scene_graph, meshcat)

        sim_diagram = sim_builder.Build()
        sim_diagram.set_name("environment_welded")

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            self.env.plant.SetPositionsAndVelocities(sim_plant_context, continuous_state)
            iiwa_position_and_velocity = self.env.plant.GetPositionsAndVelocities(sim_plant_context, model_instance=self.env.models_id[0])
            wsg_position_and_velocity = self.env.plant.GetPositionsAndVelocities(sim_plant_context, model_instance=self.env.models_id[1])
            print("CONTEXT UPDATE FUNCTION iiwa_position_and_velocity: ", iiwa_position_and_velocity)
            sim_plant_context = env.plant.GetMyContextFromRoot(sim_context)
            env.plant.SetPositionsAndVelocities(sim_plant_context, iiwa_position_and_velocity)
            env.plant.SetPositionsAndVelocities(sim_plant_context, wsg_position_and_velocity)
            return sim_context

        return Environment(
            diagram=sim_diagram,
            plant=sim_plant,
            scene_graph=sim_scene_graph,
            models_id=sim_robot_id,
            floating_bodies=floating_bodies,
            context_update_function=context_update_function,
            movable_bodies=self.env.movable_bodies,
            fixed_bodies=self.env.fixed_bodies
        )
    
    def construct_welded_sim_with_object_welded(self, continuous_state, frame_name_to_weld: str, mb: MovableBody, modify_default_context=True, meshcat=None):
        sim_builder = DiagramBuilder()
        sim_plant, sim_scene_graph = AddMultibodyPlantSceneGraph(sim_builder, time_step=self.time_step)
        sim_parser = CustomParser(sim_plant)
        models_id = sim_parser.AddModelsFromUrl(self.scenario_data)
        
        filterIiwaCollisionGeometry(sim_scene_graph)

        sim_plant.set_name("plant")

        sim_robot_id = [instance_index for instance_index in models_id
            if sim_plant.GetModelInstanceName(instance_index) in ["iiwa", "wsg"]]
        
        sim_plant: MultibodyPlant = sim_plant
        sim_plant_context = self.env.plant.CreateDefaultContext()
        self.env.plant.SetPositionsAndVelocities(sim_plant_context, continuous_state)
        print("Continuous state WELDED OBJECT: ", continuous_state)
        
        # weld everything to the world except for mb that we weld to frame
        floating_bodies = []
        for other_mb in self.env.movable_bodies:
            if other_mb.is_same(mb):
                X_WGripper = self.env.plant.GetFrameByName(frame_name_to_weld).CalcPoseInWorld(sim_plant_context)
                X_WObject = other_mb.get_pose(self.env.plant, sim_plant_context)
                print("X_WObject: ", X_WObject)
                X_GO = X_WGripper.inverse() @ X_WObject
                print("X_GO translation: ", X_GO.translation())
                print("X_GO rotation: ", X_GO.rotation())
                sim_plant.WeldFrames(sim_plant.GetFrameByName(frame_name_to_weld),
                                     other_mb.get_body(sim_plant).body_frame(),
                                     X_GO)
            else:
                X_WO = other_mb.get_pose(self.env.plant, sim_plant_context)
                sim_plant.WeldFrames(sim_plant.world_frame(),
                                     other_mb.get_body(sim_plant).body_frame(),
                                     X_WO)
                floating_bodies.append(other_mb.get_body(sim_plant))

        # ignore collisions with current object
        
        fixIiwaGripperCollisionWithObjectInGripper(sim_scene_graph, mb.body_name)
        sim_plant.Finalize()

        SetDefaultIiwaNominalPosition(sim_plant, sim_robot_id)

        if modify_default_context:
            # set default pos of robot to current position
            # sim_default_context = sim_plant.CreateDefaultContext()
            positions_iiwa = self.env.plant.GetPositionsFromArray(
                model_instance=self.env.models_id[0],
                q=self.env.plant.GetPositions(sim_plant_context))
            
            positions_wsg = self.env.plant.GetPositionsFromArray(
                model_instance=self.env.models_id[1],
                q=self.env.plant.GetPositions(sim_plant_context))
            # velocities = self.env.plant.GetVelocitiesFromArray(
            #     model_instance=self.env.model_id,
            #     v=self.env.plant.GetVelocities(plant_context))
            # todo how to set default velocity as well?
            positions = np.concatenate([positions_iiwa, positions_wsg])
            sim_plant.SetDefaultPositions(positions)

        AddIiwaDriver(sim_builder, sim_plant, ["iiwa", "wsg"], self.scenario_data)
        self._add_visuals(sim_builder, sim_scene_graph, meshcat)

        sim_diagram = sim_builder.Build()
        sim_diagram.set_name("environment_welded")

        def context_update_function(env: Environment, sim_context: Context, continuous_state: np.ndarray) -> Context:
            self.env.plant.SetPositionsAndVelocities(sim_plant_context, continuous_state)
            iiwa_position_and_velocity = self.env.plant.GetPositionsAndVelocities(sim_plant_context, model_instance=self.env.models_id[0])
            wsg_position_and_velocity = self.env.plant.GetPositionsAndVelocities(sim_plant_context, model_instance=self.env.models_id[1])
            sim_plant_context = env.plant.GetMyContextFromRoot(sim_context)            
            env.plant.SetPositionsAndVelocities(sim_plant_context, iiwa_position_and_velocity)
            env.plant.SetPositionsAndVelocities(sim_plant_context, wsg_position_and_velocity)
            return sim_context

        return Environment(
            diagram=sim_diagram,
            plant=sim_plant,
            scene_graph=sim_scene_graph,
            models_id=sim_robot_id,
            floating_bodies=floating_bodies,
            context_update_function=context_update_function,
            movable_bodies=self.env.movable_bodies,
            fixed_bodies=self.env.fixed_bodies
        )

    def construct_welded_sim_wo_robot(self):
        pass

    def construct_iiwa_alone_sim(self):
        pass

    def _weld_all_floating_bodies(self, sim_plant: MultibodyPlant, plant_context):
        floating_bodies = []
        for body_index in self.env.plant.GetFloatingBaseBodies():
            # plant ids
            body: Body = self.env.plant.get_body(body_index)
            model_instance = body.model_instance()
            model_instance_name = self.env.plant.GetModelInstanceName(model_instance)
            X_WO = self.env.plant.GetFreeBodyPose(plant_context, body)
            
            # sim indices
            model_instance = sim_plant.GetModelInstanceByName(model_instance_name)
            body = sim_plant.GetBodyByName(body.name(), model_instance)
            floating_bodies.append(body)
            sim_plant.WeldFrames(sim_plant.world_frame(), body.body_frame(), X_WO)
        return floating_bodies

    def _add_visuals(self, sim_builder, sim_scene_graph, meshcat=None, visualization_config=VisualizationConfig(publish_period=0.01)):
        if meshcat:
            visualizer = MeshcatVisualizer.AddToBuilder(
                sim_builder,
                sim_scene_graph.get_query_output_port(),
                meshcat,
                MeshcatVisualizerParams(delete_on_initialization_event=False, role=Role.kIllustration),
                # publish_period = 0.01
            )
            ApplyVisualizationConfig(visualization_config, sim_builder, meshcat=meshcat)
            return visualizer
