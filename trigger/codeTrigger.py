import os
import time
import argparse
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from runShare import run_watersharing
from runTrade import run_watertrading

class EventHandler(FileSystemEventHandler):
    def __init__(self, in_path_sharing, ex_path_sharing, in_path_trading, ex_path_trading):
        super().__init__()
        self.ex_path_sharing = os.path.abspath(os.path.expanduser(ex_path_sharing))
        self.in_path_sharing = os.path.abspath(os.path.expanduser(in_path_sharing))
        self.ex_path_trading = os.path.abspath(os.path.expanduser(ex_path_trading))
        self.in_path_trading = os.path.abspath(os.path.expanduser(in_path_trading))
    
    def on_any_event(self, event):

        event_path = os.path.abspath(event.src_path)

        if event.event_type == 'created' and event.src_path.endswith('.json'):
            if self.ex_path_sharing in event_path:
                print("Running water sharing function") 
                try:
                    run_watersharing(self.in_path_sharing, self.ex_path_sharing)
                except Exception as e:
                    print(f"Error during water sharing function: {e}")
            elif self.ex_path_trading in event_path:
                print("Running water trading function")  
                try:
                    run_watertrading(self.in_path_trading, self.ex_path_trading)
                except Exception as e:
                    print(f"Error during water trading function: {e}")
            else:
                print("Error while tracing file creation")

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

    for path_var in [ex_path_sharing, in_path_sharing, ex_path_trading, in_path_trading]:
        path = os.path.expanduser(path_var)
        if not os.path.exists(path):
            raise FileNotFoundError(f"The path '{path_var}' does not exist.")

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