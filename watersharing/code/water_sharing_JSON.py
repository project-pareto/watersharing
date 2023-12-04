# COMMON PYTHON PACKAGES
import json
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from ipywidgets import Layout,HTML,SelectionSlider, interact
from ipyleaflet import Map, basemaps, basemap_to_tiles, CircleMarker, Marker, MarkerCluster, AwesomeIcon, Popup,Polyline, LayerGroup
import ipywidgets as widgets

# PARETO-SPECIFIC
from pyomo.opt import TerminationCondition
from pyomo.environ import value
from pyomo.opt import SolverFactory
import PARETO_water_sharing_model_JSON_reform as pws



# run the optimization model
def run_optimization_models(REQUESTS_JSON,DISTANCE_JSON,MATCHES_JSON,Update_distance_matrix,verbose=True):

    df_producer, df_consumer, df_restrictions, df_distance, df_time = pws.get_data(REQUESTS_JSON,DISTANCE_JSON,Update_distance_matrix)

    # Add some output and formatting
    spacer="#"*50

    # Run first optimization model
    output_s = f"""
    {spacer}
    # Starting First Optimization: Maximizing Match Volume
    {spacer}
    """
    if verbose: print(output_s)
    objective_type = "Volume"
    prev_obj = 0
    water_sharing_volume = pws.create_model(df_restrictions,df_producer, df_consumer, df_distance, df_time, objective_type, prev_obj,default={})
    solver_volume = SolverFactory('scip', solver_io='nl')
    results_volume = solver_volume.solve(water_sharing_volume, tee=True)
    results_volume.write(format="json")

    # Run second optimization model
    if results_volume.solver.termination_condition == TerminationCondition.optimal:
        output_s = f"""
        {spacer}
        # Starting Second Optimization: Minimizing Distance & Volume
        {spacer}
        """
        if verbose: print(output_s)
        objective_type = "Match"
        prev_obj = value(water_sharing_volume.objective)
        water_sharing_match = pws.create_model(df_restrictions,df_producer, df_consumer, df_distance, df_time, objective_type, prev_obj,default={})
        solver_match = SolverFactory('scip', solver_io='nl')
        results_match = solver_match.solve(water_sharing_match, tee=True)
        results_match.write()

        # Output result 
        if results_match.solver.termination_condition == TerminationCondition.optimal:
            print('results are optimal')
            print("Total volume matched: ", value(water_sharing_volume.objective))
            #water_sharing_match.v_match.display()
            #for key in water_sharing_match.s_LLT:
            #    print(water_sharing_match.v_match[key])
            print("Number of matches: ", value(sum(water_sharing_match.v_match[pi,ci] for (pi,ci) in water_sharing_match.s_LLT)))
            MatchProcess(water_sharing_match, df_producer, df_consumer, df_distance, MATCHES_JSON)
            return None #df_match # Not needed; just need to save file
        else:
            print('results are invalid')
            return None
        

def MatchProcess(water_sharing_match, df_producer, df_consumer, df_distance, MATCHES_JSON):
    """
    Processes output match dataframe from second-round optimization
    Inputs:
    - df_matches: dataframe containing match data
    - df_distance: dataframe containing distance data
    Returns:
    - None
    Note:
    - Saves processed match file to JSON format
    """
    # Convert model output to dataframe
    df_match = pws.create_dataframe(water_sharing_match)

    # Generate distance column and add it to matches the dataframe
    distances = []
    for index, row in df_match.iterrows():
        dist = df_distance.loc[row["From wellpad"],row["To wellpad"]]
        distances.append(dist)
    df_match["Distance"] = distances

    # use Pandas merge() to add latitude and longitude data to the matches dataframe, then use that to plot lines colored by date.
    df_match = pd.merge(df_match,df_producer[["Index","Longitude","Latitude"]],left_on="From index", right_on="Index", how="left")
    df_match.rename(columns={"Longitude": "From Longitude", "Latitude": "From Latitude"},inplace=True,errors='raise')
    df_match = pd.merge(df_match,df_consumer[["Index","Longitude","Latitude"]],left_on="To index",right_on="Index",how="left")
    df_match.rename(columns={"Longitude": "To Longitude", "Latitude": "To Latitude"},inplace=True,errors='raise')

    # Convert matches dataframe to dictionary and output with json.dump
    # note: this gives the desired output format; df -> json directly seemed not to
    d_match = df_match.to_dict(orient='records')
    with open(MATCHES_JSON, "w") as data_file:
        json.dump(d_match, data_file, indent=2)
    return None