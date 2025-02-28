from pydrake.all import (
    StartMeshcat
    )
from time import sleep
from generate_models import Playground
from pydrake.systems.analysis import Simulator
from manipulation.utils import RenderDiagram
from robots import IIWA
import numpy as np
from planner import ActionExecutor, connect_to_the_world
from planner import *
from environment import MovableBody
from utils import choose_closest_heuristic

meshcat = StartMeshcat()
playground = Playground(meshcat=meshcat, time_step=0.001)
iiwa = IIWA(playground.construct_welded_sim(playground.default_continuous_state()))

def get_final_plan(playground: Playground, mb: MovableBody):
    plant_context = iiwa.get_fresh_plant_context()    
    plan = MoveToGrasp(playground, mb).then(
        OpenGripper(playground)
        ).then(
        Grasp(playground, mb)
        ).then(
        CloseGripper(playground)
        ).then(
        PostGrasp(playground, mb)
        ).then(
        MoveToDeliver(playground, mb) 
        ).then(
        CloseGripper(playground)
        ).then(
        OpenGripper(playground))
    # plan = MoveToDeliver(playground, mb) 
    
    return plan


def simulate_env():
    mb = playground.env.movable_bodies[0]

    action_executor = ActionExecutor(get_final_plan(playground, mb))
    # action_executor.action.path
    sim_diagram = connect_to_the_world(playground, action_executor)

    simulator = Simulator(sim_diagram)
    simulator.set_target_realtime_rate(1.0)
    # RenderDiagram(sim_diagram, max_depth=2)
    # meshcat.SetRealtimeRate()
    meshcat.StartRecording(set_visualizations_while_recording=True)
    simulator.AdvanceTo(3.8)
    meshcat.StopRecording()
    meshcat.PublishRecording()

    while True:
        sleep(1)

    # return action_executor.action.path

if __name__ == "__main__":
    simulate_env()