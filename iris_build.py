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
from iris_utils import *

## Setup IRIS

iris_filename = "my_iris.yaml"
iris_regions = dict()
q = []

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

#####

meshcat = StartMeshcat()
playground = Playground(meshcat=meshcat, time_step=0.001)
iiwa = IIWA(playground.construct_welded_sim(playground.default_continuous_state()))

if os.path.isfile(iris_filename):
    iris_regions.update(LoadIrisRegionsYamlFile(iris_filename))
    print(f"Loaded iris regions from {iris_filename}.")
else:
    print(f"{iris_filename} not found. No previously computed regions were loaded.")

CollisionGeometryReport(iiwa.plant, iiwa.diagram)

seeds = OrderedDict()
seeds["Home Position"] = iiwa.get_robot_nominal_position()
seeds["Front Shelve 1"] = np.array([0.75778046, -0.15025176, -0.64249042, 
                              -1.87726127, -2.51179728, 0.23095827,
                              -0.70537215, -0.05, 0.05])
seeds["Top Rack Shelve 1"] = np.array([0.8577218127779863, 1.0667040654704312, 1.33937485889043,
                                    1.4461052605134126, 2.3363107715669296, -0.7493105522005948,
                                    -1.1611574898891013, -0.05, 0.05])


iris_regions = dict()
GenerateRegions(seeds)