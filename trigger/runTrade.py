import os
import glob
import sys

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

    latest_file = max(list_of_files, key=os.path.getmtime)
    output_dir = os.path.join(ex_path_trading, "match-detail")
    file_names = {
        "requests": "_requests.json",
        "distance": "_distance.json",
        "matches": "_matches.json",
        "profits": "_profits.json",
    }

    # Run water trading optimization model
    WT.run_optimization_models(data_dir=ex_path_trading, output_dir=output_dir, file_names=file_names, Update_distance_matrix=True)
