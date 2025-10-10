#####################################################################################################
# PARETO was produced under the DOE Produced Water Application for Beneficial Reuse Environmental
# Impact and Treatment Optimization (PARETO), and is copyright (c) 2021-2024 by the software owners:
# The Regents of the University of California, through Lawrence Berkeley National Laboratory, et al.
# All rights reserved.
#
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for
# itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit others to do so.
#####################################################################################################

##################################################
# PYTHON IMPORTS
import json
import pandas as pd
from datetime import datetime, timedelta
from pyomo.opt import TerminationCondition
import pyomo.environ
from pyomo.environ import value
from pyomo.opt import SolverFactory
import ASABuildModel
from Utilities import CreateEmptyOutputJSON
from os import getcwd
from os.path import join


##################################################
# RUN OPTIMIZATION ALGORITHM
def run_optimization_models(
    data_dir=getcwd(),
    output_dir=join(getcwd(), "match-detail"),
    file_names={
        "requests": "_requests.json",
        "distance": "_distance.json",
        "matches": "_matches.json"
    },
    Update_distance_matrix=True,
    filter_by_date=None,
):

    # Set up file path names
    request_dir = join(data_dir, file_names["requests"])
    distance_dir = join(data_dir, file_names["distance"])
    matches_dir = join(data_dir, file_names["matches"])

    # Build model
    (
        df_producer,
        df_consumer,
        df_restrictions,
        df_road_distance,
        df_road_time,
        df_pipe_distance
    ) = ASABuildModel.get_data(
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
    # Starting Optimization Run: Sharing Formulation
    {spacer}
    """
    print(output_s)

    water_sharing = ASABuildModel.create_model(
        df_restrictions,
        df_producer,
        df_consumer,
        df_road_distance,
        df_road_time,
        df_pipe_distance,
        default={}
    )
    # solver = SolverFactory('scip', solver_io='nl')
    # solver = SolverFactory("gurobi")
    solver = SolverFactory("glpk")
    results = solver.solve(water_sharing, tee=True)
    # Check if no matches; if so, return an empty match file
    if water_sharing.objective() <= 0: # solvers sometimes return -0.
        print("No matches!")
        CreateEmptyOutputJSON(matches_dir)
        return None
    # else:
    results.write(format="json")

    # Output result
    if results.solver.termination_condition == TerminationCondition.optimal:
        print("results are optimal")
        ASABuildModel.jsonize_outputs(water_sharing, matches_dir)
        ASABuildModel.OutputMatchesToUsers(matches_dir, output_dir)
        # BuildModel.DataViews(water_sharing, data_dir)
        # BuildModel.PostSolveViews(water_sharing, data_dir)
        return None
    else:
        print("results are invalid")
        return None
