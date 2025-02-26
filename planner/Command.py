import numpy as np
from abc import ABC, abstractmethod

from robots import IIWA
from settings import GRIPPER_BASE_LINK


class Command:
    def __init__(self, iiwa: IIWA, position_command: np.ndarray):
        self.iiwa = iiwa
        self.position_command = position_command

    def new_command_ignore_grippers(self, new_position_command: np.ndarray) -> "Command":
        my_positions = self.position_command.copy()
        self.iiwa.set_ignoring_gripper_position(
            positions_from=new_position_command,
            positions_to=my_positions
        )
        return Command(iiwa=self.iiwa, position_command=my_positions)


class Gripper(ABC):

    def open(self, command: Command) -> Command:
        position = command.position_command
        command.iiwa.get_open_gripper_position(initial_positions=position)
        # print("Position: ", position)
        return Command(iiwa=command.iiwa, position_command=position)

    def close(self, command: Command, run_time) -> Command:
        position = command.position_command
        if run_time < 0.25:    
            command.iiwa.get_close_gripper_position(initial_positions=position, finger_position=0.01)
        else:
            command.iiwa.get_close_gripper_position(initial_positions=position, finger_position=0.01)
        return Command(iiwa=command.iiwa, position_command=position)

    def connecting_frame_name(self):
        return GRIPPER_BASE_LINK

# class LeftHand(Hand):
#     def open(self, command: Command) -> Command:
#         position = command.position_command
#         command.pr2.get_open_gripper_position(
#             gripper_joint_name=command.pr2.l_gripper_joint_name,
#             initial_positions=position
#         )
#         return Command(pr2=command.pr2, position_command=position)

#     def close(self, command: Command) -> Command:
#         position = command.position_command
#         command.pr2.get_close_gripper_position(
#             gripper_joint_name=command.pr2.l_gripper_joint_name,
#             initial_positions=position
#         )
#         return Command(pr2=command.pr2, position_command=position)

#     def connecting_frame_name(self):
#         return self.pr2.l_gripper_base_link_name


# class RightHand(Hand):
#     def open(self, command: Command) -> Command:
#         position = command.position_command
#         command.pr2.get_open_gripper_position(
#             gripper_joint_name=command.pr2.r_gripper_joint_name,
#             initial_positions=position
#         )
#         return Command(pr2=command.pr2, position_command=position)

#     def close(self, command: Command) -> Command:
#         position = command.position_command
#         command.pr2.get_close_gripper_position(
#             gripper_joint_name=command.pr2.r_gripper_joint_name,
#             initial_positions=position
#         )
#         return Command(pr2=command.pr2, position_command=position)

#     def connecting_frame_name(self):
#         return self.pr2.r_gripper_base_link_name
