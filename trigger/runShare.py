import os
import glob
import sys

# Add the ../watersharing/code/ directory to sys.path
watersharing_code_dir = os.path.abspath("../watersharing/code")
if watersharing_code_dir not in sys.path:
    sys.path.insert(0, watersharing_code_dir)

# Now we can directly import the modules as if they were in the current directory
import water_sharing_JSON as WS
from WSP_Visualization import SummaryPlot

def run_watersharing(in_path_sharing, ex_path_sharing):
    print("RUN SHARE STARTED")
    path = os.path.expanduser(ex_path_sharing + "/*.json")
    list_of_files = glob.glob(path)

    if len(list_of_files) == 0:
        raise Exception("Error: Target folder is empty or does not exist")

    latest_file = max(list_of_files, key=os.path.getmtime)
    
    gwpc_dir = os.path.join(in_path_sharing, "GWPC_V2")
    if not os.path.exists(gwpc_dir):
        os.makedirs(gwpc_dir)
    
    DISTANCE_JSON = os.path.join(gwpc_dir, "GWPC_V2_distance.json")
    MATCHES_JSON = os.path.join(gwpc_dir, "GWPC_V2_matches.json")

    # Run water sharing optimization model
    WS.run_optimization_models(latest_file, DISTANCE_JSON, MATCHES_JSON, Update_distance_matrix=True)
    SummaryPlot(ex_path_sharing, MATCHES_JSON, latest_file, DISTANCE_JSON)