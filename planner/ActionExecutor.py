import numpy as np
from pydrake.all import LeafSystem, DiagramBuilder, QueryObject, AbstractValue

from generate_models import Playground

from robots import IIWA
from .Action import Action
from .Command import Command
from pydrake.all import Demultiplexer



class ActionExecutor(LeafSystem):
    def __init__(self, action: Action):
        LeafSystem.__init__(self)

        playground = action.playground
        self.action = action

        ################################
        # declare input/output
        print("  State output port plant size: ", playground.env.plant.get_state_output_port().size())
        print("  State output port plant: ", playground.env.plant.get_state_output_port().Eval(playground.env.get_fresh_plant_context()))
        

        self.DeclareVectorInputPort(
            name="plant_continuous_state",  # assume this is q, dq
            size=playground.env.plant.get_state_output_port().size()
        )
        self.DeclareVectorInputPort(
            name="iiwa.torque_external",
            size=playground.env.plant.get_generalized_contact_forces_output_port(playground.env.models_id[0]).size()
        )

        self.DeclareVectorInputPort(
            name="wsg.torque_external",
            size=playground.env.plant.get_generalized_contact_forces_output_port(playground.env.models_id[1]).size()
        )

        self.DeclareAbstractInputPort(
            name="query_object",
            model_value=AbstractValue.Make(QueryObject())
        )

        self.DeclareVectorOutputPort(
            name="iiwa_position_command",
            size=playground.env.plant.num_actuated_dofs(),
            # size=7,
            calc=self.CalcPositionCommand
        )

        # self.DeclareVectorOutputPort(
        #     name="feedforward_torque",
        #     size=playground.env.plant.num_actuated_dofs(),
        #     calc=self.CalcFeedForwardTorque
        # )
        # end declare input/output
        ################################
        self.is_finished = False

        # this part is the only non-beautiful part about this framework which is a result of technical debt
        # we need to construct iiwa in order to make the command
        iiwa = IIWA(playground.construct_welded_sim(playground.default_continuous_state()))
        self.command = Command(iiwa=iiwa, position_command=playground.construct_welded_sim(playground.default_continuous_state()).plant.GetDefaultPositions()) # default command is all zeros
        print("Action executor")

    def CalcPositionCommand(self, context, output):
        if self.is_finished:
            return
        continuous_state = self.GetInputPort("plant_continuous_state").Eval(context)
        # print("Continuous state: ", continuous_state)
        self.action.set_data(
            continuous_state=continuous_state,
            time=context.get_time(),
            torque_external=self.GetInputPort("iiwa.torque_external").Eval(context)
        )
        self.command, done = self.action.run_or_init(self.command)
        # print("self command: ", self.command.position_command)
        if done:
            self.action.state_finished()
        # print("self command possition comamnd: ", self.command.position_command)
        output.SetFromVector(self.command.position_command)
        self.is_finished = self.is_finished or done

    # todo
    # def CalcFeedForwardTorque(self, context, output):
    #     # maybe use later for gripper
    #     output.SetFromVector(np.zeros(output.size()))


def connect_to_the_world(playground: Playground, action_executor: ActionExecutor):
    builder = DiagramBuilder()
    inner_diagram = builder.AddSystem(playground.env.diagram)
    builder.AddSystem(action_executor)

    print("CONNECT TO THE WORLD: ")
    print("iiwa_position_command", action_executor.GetOutputPort("iiwa_position_command"))

    # 1. Add a Demultiplexer to split the 9D position command into 7D (iiwa) and 2D (wsg)
    demux = builder.AddSystem(Demultiplexer(output_ports_sizes=[7, 1, 1]))

    builder.Connect(
        action_executor.GetOutputPort("iiwa_position_command"),
        demux.get_input_port()
    )

    # 3. Connect Demultiplexer outputs to respective input ports
    builder.Connect(
        demux.get_output_port(0),  # First 7 elements for IIWA
        inner_diagram.GetInputPort("iiwa.position")
    )
    builder.Connect(
        demux.get_output_port(1),  # Last 2 elements for WSG gripper
        inner_diagram.GetInputPort("wsg.position")
    )

    # builder.Connect(
    #     action_executor.GetOutputPort("iiwa_position_command"),
    #     inner_diagram.GetInputPort("iiwa.position")
    # )

    # builder.Connect(
    #     action_executor.GetOutputPort("iiwa_position_command"),
    #     inner_diagram.GetInputPort("wsg.position")
    # )

    # builder.Connect(
    #     action_executor.GetOutputPort("feedforward_torque"),
    #     inner_diagram.GetInputPort("iiwa.feedforward_torque")
    # )
    builder.Connect(
        inner_diagram.GetOutputPort("query_object"),
        action_executor.GetInputPort("query_object")
    )
    builder.Connect(
        inner_diagram.GetOutputPort("plant_continuous_state"),
        action_executor.GetInputPort("plant_continuous_state")
    )
    builder.Connect(
        inner_diagram.GetOutputPort("iiwa.torque_external"),
        action_executor.GetInputPort("iiwa.torque_external")
    )
    return builder.Build()
