import os
import glob
import sys

from datetime import datetime

# Add the ../watersharing/code/ directory to sys.path
watersharing_code_dir = os.path.abspath("../watersharing_analogue/code")
if watersharing_code_dir not in sys.path:
    sys.path.insert(0, watersharing_code_dir)

# Add the ../watertrading/code/ directory to sys.path (for trading and shared modules)
watertrading_code_dir = os.path.abspath("../watertrading/code")
if watertrading_code_dir not in sys.path:
    sys.path.insert(0, watertrading_code_dir)

import ASAMainScript as WS
import MainScript as WT

def run_matching(optimizer_module, in_path, ex_path):
    """
    Common logic for running water trading or water sharing optimization.

    Args:
        optimizer_module: The module to use (WT or WS)
        in_path: Input path for optimization results
        ex_path: Export path to find latest request file
    """
    path = os.path.expanduser(ex_path + "/*.json")
    list_of_files = glob.glob(path)

    if len(list_of_files) == 0:
        raise Exception("Error: Target folder is empty or does not exist")

    now = datetime.now()
    date_time = now.strftime('%Y-%m-%d_%H-%M-%S')

    latest_file = max(list_of_files, key=os.path.getmtime)
    file_names = {
        "requests": latest_file,
        "distance": "_distance_" + date_time + ".json",
        "matches": "_matches_" + date_time + ".json",
        "profits": "_profits_" + date_time + ".json",
    }

    match_detail_dir = os.path.join(in_path, "match-detail")

    # Run optimization model
    optimizer_module.run_optimization_models(
        data_dir=in_path,
        output_dir=match_detail_dir,
        file_names=file_names,
        Update_distance_matrix=True
    )

def run_watertrading(in_path_trading, ex_path_trading):
    """Run water trading optimization model."""
    run_matching(WT, in_path_trading, ex_path_trading)

def run_watersharing(in_path_sharing, ex_path_sharing):
    """Run water sharing optimization model."""
    run_matching(WS, in_path_sharing, ex_path_sharing)