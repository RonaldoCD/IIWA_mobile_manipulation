from generate_models import Playground
from planner.Action import Action
from planner.Command import Gripper, Command


class OpenGripper(Action):
    """
    Opens the gripper to ready for grasping
    """

    def __init__(self, playground: Playground):
        super(OpenGripper, self).__init__(playground)
        self.gripper = Gripper() 
        self.plan_duration = 0.5
        self.start_time = 0


    def state_init(self):
        self.start_time = self.time

    def run(self, prev_command: Command):
        print("Time open: ", self.time)
        finished = (self.time - self.start_time) > self.plan_duration
        command = self.gripper.open(prev_command)
        # brick = self.playground.env.plant.GetBodyByName("brick_center")           
        # plant_context = self.playground.env.plant.CreateDefaultContext()  # Get the simulation state
        # X_WB = self.playground.env.plant.EvalBodyPoseInWorld(plant_context, brick)  # Pose in world
        # position = X_WB.translation()  # Extract position
        # print("Position of thing_1: ", position)
        return command, finished

