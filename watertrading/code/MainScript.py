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


##################################################
# RUN OPTIMIZATION ALGORITHM
def run_optimization_models(REQUESTS,DISTANCE,MATCHES,PROFITS,Update_distance_matrix=True,save_dir=getcwd(),verbose=True,OnlineMode=False,filter_ON=False,filter_date=None):

    df_producer, df_consumer, df_midstream, df_restrictions, df_distance, df_time = BuildModel.get_data(REQUESTS,DISTANCE,Update_distance_matrix,OnlineMode=OnlineMode,filter_ON=filter_ON,filter_date=filter_date)

    # Add some output and formatting
    spacer="#"*50

    # Run first optimization model
    output_s = f"""
    {spacer}
    # Starting Optimization Run: Clearing Formulation
    {spacer}
    """
    if verbose: print(output_s)
    
    water_sharing = BuildModel.create_model(df_restrictions, df_producer, df_consumer, df_midstream, df_distance, df_time,default={})
    #solver = SolverFactory('scip', solver_io='nl')
    #solver = SolverFactory("gurobi")
    solver = SolverFactory("glpk")
    results = solver.solve(water_sharing, tee=True)
    results.write(format="json")

    # Output result 
    if results.solver.termination_condition == TerminationCondition.optimal:
        print('results are optimal')
        #print("Total volume matched: ", value(water_sharing_volume.objective))
        #water_sharing.v_match.display()
        #for key in water_sharing_match.s_LLT:
        #    print(water_sharing_match.v_match[key])
        water_sharing = BuildModel.PostSolve(water_sharing)
        BuildModel.jsonize_outputs(water_sharing,MATCHES)
        BuildModel.jsonize_profits(water_sharing, PROFITS)
        BuildModel.DataViews(water_sharing, save_dir)
        BuildModel.PostSolveViews(water_sharing, save_dir)
        return None
    else:
        print('results are invalid')
        return None