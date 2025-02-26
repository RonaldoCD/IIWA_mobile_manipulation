from time import sleep

from pydrake.all import (
    StartMeshcat,
    Simulator,
)
from generate_models import Playground
from planner import ActionExecutor, connect_to_the_world
from environment import MovableBody
from planner import *

from utils import choose_closest_heuristic
from robots import IIWA

meshcat = StartMeshcat()

playground = Playground(meshcat=meshcat, time_step=0.001)
iiwa = IIWA(playground.construct_welded_sim(playground.default_continuous_state()))

def get_final_plan(playground: Playground, mb: MovableBody):
    plant_context = iiwa.get_fresh_plant_context()
    # gripper_name = choose_closest_heuristic(pr2.l_gripper_frame_name, pr2.r_gripper_frame_name,
    #                                      mb.get_pose(pr2.plant, plant_context).translation(),
    #                                      pr2.plant, plant_context)
    # hand = left_hand if "l_gripper" in gripper_name else right_hand

    plan = MoveToGrasp(playground, mb).th

    return plan


def simulate_env():
    mb = playground.env.movable_bodies[0]

    action_executor = ActionExecutor(get_final_plan(playground, mb))
    sim_diagram = connect_to_the_world(playground, action_executor)

    simulator = Simulator(sim_diagram)

    meshcat.StartRecording(set_visualizations_while_recording=True)
    simulator.AdvanceTo(12.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()

    while True:
        sleep(1)


if __name__ == "__main__":
    simulate_env()
