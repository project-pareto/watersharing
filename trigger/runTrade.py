import os
import glob
import sys

from datetime import datetime

# Add the ../watertrading/code/ directory to sys.path
watertrading_code_dir = os.path.abspath("../watertrading/code")
if watertrading_code_dir not in sys.path:
    sys.path.insert(0, watertrading_code_dir)

# Now we can directly import the module as if it were in the current directory
import MainScript as WT

def run_watertrading(in_path_trading, ex_path_trading):
    path = os.path.expanduser(ex_path_trading + "/*.json")
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

    # Run water trading optimization model
    WT.run_optimization_models(data_dir=in_path_trading, output_dir=in_path_trading, file_names=file_names, Update_distance_matrix=True)