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
from WSP_Visualization import SummaryPlot

# Set Matplotlib backend to Agg to prevent GUI errors
import matplotlib
matplotlib.use('Agg')


class EventHandler(FileSystemEventHandler):
    def __init__(self, in_path_var, ex_path_var):
        super().__init__()
        self.ex_path_var = ex_path_var
        self.in_path_var = in_path_var
    
    def on_any_event(self, event):
        if event.event_type == 'created':
            #Do not run the script if there is already a matching process going on
            if not is_process_running("water_sharing_JSON.py"): 
                #Get all json files and create a list of them
                path = os.path.expanduser((self.ex_path_var + "/*.json")) #Normalize path for unix and windows
                list_of_files = glob.glob(path)

                print(len(list_of_files))

                #Error handling for if json export folder is empty or non-existant
                if(len(list_of_files)==0):
                    raise Exception("Error: Target folder is empty or does not exist")

                #Get most recent file
                latest_file = max(list_of_files, key=os.path.getmtime) 
                
                #Set up json file directory
                REQUESTS_GWPC_V2 = latest_file

                #Set up output file paths
                DISTANCE_GWPC_V2 = os.path.join(in_path_var,"GWPC_V2\\GWPC_V2_distance.json")
                MATCHES_GWPC_V2 = os.path.join(in_path_var,"GWPC_V2\\GWPC_V2_matches.json")

                WS.run_optimization_models(REQUESTS_GWPC_V2,DISTANCE_GWPC_V2,MATCHES_GWPC_V2,Update_distance_matrix=True)
                SummaryPlot(ex_path_var,MATCHES_GWPC_V2,REQUESTS_GWPC_V2,DISTANCE_GWPC_V2)

def is_process_running(process_name):
    output = subprocess.Popen(["pgrep", "-f", process_name], stdout=subprocess.PIPE).communicate()[0]
    return len(output.strip()) > 0

#Load paths from config.yaml
def load_config(config_file_path, mode):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    
    if mode == 'local': #Local Dir
        return config['local_vars']
    elif mode == 'production': #Wordpress Dir
        return config['production_vars']
    else:
        raise ValueError("Invalid mode. Use 'local' or 'production'.")

if __name__ == "__main__":
    
    #Sets up arg parser and loads production/local path depending on cmd arguments
    parser = argparse.ArgumentParser(description='Parses command line arguments')
    parser.add_argument('--mode', choices=['local', 'production'], default='production', help='Set the environment mode (local or production)')
    args = parser.parse_args()
    config = load_config('config.yaml', args.mode)
    
    ex_path_var = config['ex_path_var']
    in_path_var = config['in_path_var']

    path1 = os.path.expanduser(ex_path_var)

    path2 = os.path.expanduser(in_path_var)

    #Error handling for if an unknown json file path is being used
    if not os.path.exists(path1):
        raise FileNotFoundError("The path \'" + ex_path_var + "\' does not exist.")
    
    if not os.path.exists(path2):
        raise FileNotFoundError("The path \'" + in_path_var + "\' does not exist.")

    #Use the event handler to periodically check for new files
    event_handler = EventHandler(in_path_var,ex_path_var) 
    observer = Observer()
    observer.schedule(event_handler, path1, recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()