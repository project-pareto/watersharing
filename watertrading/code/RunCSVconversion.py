import json
import pandas as pd
import Utilities

input_dir = "C:\\Users\\Philip\\Documents\\ProjectPARETO\\WSBidding_cases\\NE_PA_04"

# Northwestern PA demo problem, converted for bidding
Utilities.csv_2_json(input_dir,"Demo_Producers.csv","Demo_Consumers.csv","Demo_Midstreams.csv",name="_requests.json")