import os

PROJECT_PATH = os.path.dirname(__file__)
MODELS_PATH = os.path.join(PROJECT_PATH, "models/")
PACKAGE_XML_PATH = os.path.join(PROJECT_PATH, "package.xml")
CACHE_PATH = os.path.join(PROJECT_PATH, "cache")
DEFAULT_ENV_URL = "package://Project/models/main_environment.dmd.yaml"
EASY_ENV_URL = "package://Project/models/main_environment.dmd.yaml"
# IIWA_MODEL_URL = "package://drake_models/iiwa_description/sdf/iiwa7_no_collision.sdf"
IIWA_MODEL_URL = "package://Project/models/iiwa_gripper.dmd.yaml"
DEFAULT_SCENARIO_URL = "package://Project/models/main_environment.dmd.yaml"
DEFAULT_WELDED_SCENARIO_URL = "package://Project/models/welded_object_main_environment.dmd.yaml"

# PR2_MODEL_URL = "package://Shelve_Bot/models/pr2_collisions_filtered.urdf"
GRIPPER_MODEL_URL = "package://manipulation/hydro/schunk_wsg_50_with_tip.sdf"
GRIPPER_BASE_LINK = "body"

IIWA_MAIN_LINK = "iiwa_link_0"
MAX_GRASP_CANDIDATE_GENERATION = 20
FLOOR_FOR_BRICKS = 2  # must be between 0 and 2 corresponding to 1 and 3
PRESEEDED_IK = True # set to False to solve all IKs with random init

# make the necessary folders:
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)
