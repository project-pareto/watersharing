import os
import time
import subprocess
import sys

import yaml
import argparse
import glob
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import json
import pandas as pd
import water_sharing_JSON as WS
import watertrading.code.Mainscript as WT
from WSP_Visualization import SummaryPlot

# Set Matplotlib backend to Agg to prevent GUI errors
import matplotlib
matplotlib.use('Agg')


class EventHandler(FileSystemEventHandler):
    def __init__(self, in_path_sharing, ex_path_sharing, in_path_trading, ex_path_trading):
        super().__init__()
        self.ex_path_sharing = ex_path_sharing
        self.in_path_sharing = in_path_sharing
        self.ex_path_trading = ex_path_trading
        self.in_path_trading = in_path_trading
    
    def on_any_event(self, event):
        if event.event_type == 'created':
            # Determine which directory the event occurred in
            if self.ex_path_sharing in event.src_path:
                self.run_watersharing(event.src_path)
            elif self.ex_path_trading in event.src_path:
                self.run_watertrading(event.src_path)

    def run_watersharing(self, event_path):
        # Check if water_sharing_JSON.py process is already running
        if not is_process_running("water_sharing_JSON.py"): 
            # Get all json files and create a list of them
            path = os.path.expanduser((self.ex_path_sharing + "/*.json"))
            list_of_files = glob.glob(path)

            if len(list_of_files) == 0:
                raise Exception("Error: Target folder is empty or does not exist")

            latest_file = max(list_of_files, key=os.path.getmtime)

            # Set up output file paths
            DISTANCE_JSON = os.path.join(self.in_path_sharing, "GWPC_V2", "GWPC_V2_distance.json")
            MATCHES_JSON = os.path.join(self.in_path_sharing, "GWPC_V2", "GWPC_V2_matches.json")

            # Run water sharing optimization model
            WS.run_optimization_models(latest_file, DISTANCE_JSON, MATCHES_JSON, Update_distance_matrix=True)
            SummaryPlot(self.ex_path_sharing, MATCHES_JSON, latest_file, DISTANCE_JSON)

    def run_watertrading(self, event_path):
        # Check if Mainscript.py process is already running
        if not is_process_running("Mainscript.py"):
            # Get all json files and create a list of them
            path = os.path.expanduser((self.ex_path_trading + "/*.json"))
            list_of_files = glob.glob(path)

            if len(list_of_files) == 0:
                raise Exception("Error: Target folder is empty or does not exist")

            latest_file = max(list_of_files, key=os.path.getmtime)

            # Define the output directory and file names
            output_dir = os.path.join(self.in_path_trading, "match-detail")
            file_names = {
                "requests": "_requests.json",
                "distance": "_distance.json",
                "matches": "_matches.json",
                "profits": "_profits.json",
            }

            # Run water trading optimization model
            WT.run_optimization_models(data_dir=self.ex_path_trading, output_dir=output_dir,
                                       file_names=file_names, Update_distance_matrix=True)

def is_process_running(process_name):
    output = subprocess.Popen(["pgrep", "-f", process_name], stdout=subprocess.PIPE).communicate()[0]
    return len(output.strip()) > 0

# Load paths from config.yaml
def load_config(config_file_path, mode):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if mode == 'local':
        return config['local_vars']
    elif mode == 'production':
        return config['production_vars']
    else:
        raise ValueError("Invalid mode. Use 'local' or 'production'.")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Parses command line arguments')
    parser.add_argument('--mode', choices=['local', 'production'], default='production', help='Set the environment mode (local or production)')
    args = parser.parse_args()
    config = load_config('config.yaml', args.mode)
    
    ex_path_sharing = config['watersharing']['ex_path']
    in_path_sharing = config['watersharing']['in_path']
    ex_path_trading = config['watertrading']['ex_path']
    in_path_trading = config['watertrading']['in_path']

    # Check if the specified paths exist
    for path_var in [ex_path_sharing, in_path_sharing, ex_path_trading, in_path_trading]:
        path = os.path.expanduser(path_var)
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path_var}' does not exist.")

    # Set up the observer for both directories
    event_handler = EventHandler(in_path_sharing, ex_path_sharing, in_path_trading, ex_path_trading) 
    observer = Observer()
    observer.schedule(event_handler, os.path.expanduser(ex_path_sharing), recursive=True)
    observer.schedule(event_handler, os.path.expanduser(ex_path_trading), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
