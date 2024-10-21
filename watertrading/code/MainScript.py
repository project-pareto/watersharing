##################################################
# PYTHON IMPORTS
import json
import pandas as pd
from datetime import datetime, timedelta
from pyomo.opt import TerminationCondition
import pyomo.environ
from pyomo.environ import value
from pyomo.opt import SolverFactory
import BuildModel
from os import getcwd
from os.path import join


##################################################
# RUN OPTIMIZATION ALGORITHM
def run_optimization_models(
    data_dir=getcwd(),
    file_names={
        "requests":"_requests.json",
        "distance":"_distance.json",
        "matches":"_matches.json",
        "profits":"_profits.json"
    },
    Update_distance_matrix=True,
    filter_by_date=None,
):

    # Set up file path names
    request_dir = join(data_dir,file_names["requests"])
    distance_dir = join(data_dir,file_names["distance"])
    matches_dir = join(data_dir,file_names["matches"])
    profits_dir = join(data_dir,file_names["profits"])

    # Build model 
    (
        df_producer,
        df_consumer,
        df_restrictions,
        df_distance,
        df_time,
    ) = BuildModel.get_data(
        request_dir,
        distance_dir,
        Update_distance_matrix,
        filter_by_date=filter_by_date,
    )

    # Add some output and formatting
    spacer = "#" * 50

    # Run first optimization model
    output_s = f"""
    {spacer}
    # Starting Optimization Run: Clearing Formulation
    {spacer}
    """
    print(output_s)

    water_sharing = BuildModel.create_model(
        df_restrictions,
        df_producer,
        df_consumer,
        df_distance,
        df_time,
        default={},
    )
    # solver = SolverFactory('scip', solver_io='nl')
    # solver = SolverFactory("gurobi")
    solver = SolverFactory("glpk")
    results = solver.solve(water_sharing, tee=True)
    results.write(format="json")

    # Output result
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("results are optimal")
        water_sharing = BuildModel.PostSolve(water_sharing)
        BuildModel.jsonize_outputs(water_sharing, matches_dir)
        #BuildModel.jsonize_profits(water_sharing, profits_dir)
        #BuildModel.DataViews(water_sharing, data_dir)
        #BuildModel.PostSolveViews(water_sharing, data_dir)
        return None
    else:
        print("results are invalid")
        return None
