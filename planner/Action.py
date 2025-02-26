from abc import ABC, abstractmethod

from generate_models import Playground
from planner.Command import Command
from manipulation.meshcat_utils import AddMeshcatTriad
import numpy as np


class Action(ABC):
    def __init__(self, playground: Playground):
        self.playground = playground
        self.continuous_state = None
        self.time = None
        self.torque_external = None
        self.__is_first_call = True
        self.ee_path = np.empty((3, 0))

    def run_or_init(self, command: Command):
        if self.__is_first_call:
            self.state_init_verbose()
            self.__is_first_call = False
        return self.run(command)

    def get_name(self):
        return self.__class__.__name__

    @abstractmethod
    def state_init(self):
        pass

    def state_init_verbose(self):
        print("initializing", self.get_name())
        self.state_init()

    def state_finished(self):
        self.__is_first_call = True

    @abstractmethod
    def run(self, prev_command: Command):
        """
        output: new_command, finished
        this gets called until we return finished=True
        """
        pass

    def set_data(self, continuous_state, time, torque_external):
        self.continuous_state = continuous_state
        self.time = time
        self.torque_external = torque_external

    def then(self, next_action: "Action"):
        return ComposedAction(self, next_action)
    
    def display_frame(self, X_WF, frame_name):
        AddMeshcatTriad(self.playground.meshcat, path=frame_name, X_PT=X_WF, opacity=0.5, length=0.1, radius=0.01)
    
    def display_path(self):
        self.playground.meshcat.SetLine(
            path="path_ee", 
            vertices=self.ee_path, 
            # color=[1, 0, 0],  # Red color
            line_width=2.0
        )

class ComposedAction(Action):
    def __init__(self, first: Action, second: Action):
        assert first.playground == second.playground
        super(ComposedAction, self).__init__(first.playground)
        self.first = first
        self.second = second
        self.is_first_finished = False
        self.is_second_started = False
        print("composeed")

    def get_name(self):
        return f"{self.first.get_name()} -> {self.second.get_name()}"

    def state_init(self):
        pass

    def run(self, command: Command):
        
        if self.is_first_finished:
            return self.second.run_or_init(command)
        else:
            command, finished = self.first.run_or_init(command)
            if finished:
                self.is_first_finished = True
                self.first.state_finished()
            return command, False

    def state_finished(self):
        super(ComposedAction, self).state_finished()
        self.second.state_finished()

    def set_data(self, *args, **kwargs):
        super(ComposedAction, self).set_data(*args, **kwargs)
        if self.is_first_finished:
            self.second.set_data(*args, **kwargs)
        else:
            self.first.set_data(*args, **kwargs)

        
        # X_Custom = RigidTransform(RotationMatrix(), [0.7, 0.0, 0.65])
        # # AddMeshcatTriad(self.playground.meshcat, path="custom_frame", X_PT=X_Custom, opacity=0.5)
