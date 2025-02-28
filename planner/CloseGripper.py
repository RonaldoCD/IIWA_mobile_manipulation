from generate_models import Playground
from planner.Action import Action
from planner.Command import Gripper, Command


class CloseGripper(Action):
    """
    closes the gripper $hand
    """

    def __init__(self, playground: Playground):
        super(CloseGripper, self).__init__(playground)
        self.gripper = Gripper()
        self.plan_duration = 0.5
        self.start_time = 0

    def state_init(self):
        self.start_time = self.time

    def run(self, prev_command: Command):
        # print("Time close: ", self.time)
        run_time = self.time - self.start_time
        finished = run_time > self.plan_duration
        command = self.gripper.close(prev_command, run_time)
        return command, finished