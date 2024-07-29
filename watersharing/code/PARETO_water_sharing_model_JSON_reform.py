'''
|   Parse input files for the matching optimization model
|   
'''
import os.path 
import pandas as pd
import numpy as np
import json

# Import
from cmath import nan
#from unittest import result

from pyomo.environ import (
    Var,
    Param,
    Set,
    ConcreteModel,
    Constraint,
    Objective,
    minimize,
    maximize,
    NonNegativeReals,
    Reals,
    Binary,
    Any,
    units as pyunits,
    Block,
    Suffix,
    SolverFactory,
    TransformationFactory,
    value,
)

from pyomo.core.base.constraint import simple_constraint_rule
from pyomo.core.expr.current import identify_variables

# from gurobipy import *
from pyomo.common.config import ConfigBlock, ConfigValue, In, Bool
from enum import Enum
from pyomo.opt import TerminationCondition
from GetDistance_JSON import get_driving_distance
from WSP_Utilities_JSON import df_to_dict_helper

# def input_data():
#inputs
# self.input_dir = input_dir
class Objectives(Enum):
    volume = 1
    match = 2
    cost = 3

# create config dictionary
CONFIG = ConfigBlock()
CONFIG.declare(
    "has_pipeline_constraints",
    ConfigValue(
        default=True,
        domain=Bool,
        description="build pipeline constraints",
        doc="""Indicates whether holdup terms should be constructed or not.
**default** - True.
**Valid values:** {
**True** - construct pipeline constraints,
**False** - do not construct pipeline constraints}""",
    ),
)
CONFIG.declare(
    "has_trucking_constraints",
    ConfigValue(
        default=True,
        domain=Bool,
        description="build pipeline constraints",
        doc="""Indicates whether holdup terms should be constructed or not.
**default** - True.
**Valid values:** {
**True** - construct pipeline constraints,
**False** - do not construct pipeline constraints}""",
    ),
)
CONFIG.declare(
    "objective",
    ConfigValue(
        default=Objectives.volume,
        domain=In(Objectives),
        description="alternate objectives selection",
        doc="""Alternate objective functions (i.e., minimize cost, maximize reuse)
**default** - Objectives.cost
**Valid values:** {
**Objectives.cost** - cost objective
**Objectives.matches** - matching objective}""",
    ),
)





def get_data(REQUESTS_JSON,DISTANCE_JSON,Update_distance_matrix=False):
    """
    Inputs:
    > REQUESTS_JSON - directory to requests JSON file
    > DISTANCE_JSON - directory to distance JSON file
    > Update_distance_matrix - optional kwarg; if true, creates a distance matrix, if false, assumes the distance matrix exists and is CORRECT
    Outputs:
    > df_producer - producer request dataframe
    > df_consumer - consumer request dataframe
    > df_distance - drive distance dataframe
    > df_time - drive time dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
                print("Data directory not found")

    # Pull in JSON request data
    with open(REQUESTS_JSON, "r") as read_file:
        request_data = json.load(read_file)

    # Producer inputs
    df_producer = pd.DataFrame(data=request_data["Producers"])
    # convert time inputs to datetime
    df_producer['Start Date'] = pd.to_datetime(df_producer['Start Date'], format ="%Y/%m/%d")
    df_producer['End Date'] = pd.to_datetime(df_producer['End Date'], format ="%Y/%m/%d")

    # Consumer inputs
    df_consumer = pd.DataFrame(data=request_data["Consumers"])
    # convert time inputs to datetime
    df_consumer['Start Date'] = pd.to_datetime(df_consumer['Start Date'], format ="%Y/%m/%d")
    df_consumer['End Date'] = pd.to_datetime(df_consumer['End Date'], format ="%Y/%m/%d")

    # Matching restrictions
    df_restrictions = pd.DataFrame(data=request_data["Restrictions"])

    # Initialize dataframes (producer and consumer dataframes now created directly from JSON data)
    df_distance = pd.DataFrame()
    df_time = pd.DataFrame()
  
    # If Update_distance_matrix then get_driving_distance() else load existing distance JSON
    if Update_distance_matrix: # generate new distance matrix using API
        df_time, df_distance = get_driving_distance(DISTANCE_JSON,df_producer,df_consumer)

    else: # load existing JSON distance matrix
        with open(DISTANCE_JSON, "r") as read_file:
            distance_data = json.load(read_file)
    
        # Distance matrix parsing
        df_distance = pd.DataFrame(data=distance_data["DriveDistances"])
        df_time = pd.DataFrame(data=distance_data["DriveTimes"])

    print("input data instance created")
    return df_producer, df_consumer, df_restrictions, df_distance, df_time
  



def create_model(restricted_set, df_producer, df_consumer, df_distance, df_time, objective_type, prev_obj, default={}):
    model = ConcreteModel()

    # import config dictionary
    model.config = CONFIG(default)
    model.type = "water_sharing"

    model.df_producer = df_producer
    model.df_consumer = df_consumer
    model.df_distance = df_distance
    model.df_time = df_time

    model.model_units = {
        "volume": pyunits.koil_bbl,
        "distance": pyunits.mile,
        "diameter": pyunits.inch,
        "concentration": pyunits.kg / pyunits.liter,
        # "currency": pyunits.kUSD,
        "time": pyunits.day,
        "volume_time": pyunits.koil_bbl / pyunits.day,
    }

    
    # DETERMINE DATE RANGE - assume need to optimize from the earliest date to the latest date in data
    first_date = min(df_producer['Start Date'].append(df_consumer['Start Date']))
    last_date = max(df_producer['End Date'].append(df_consumer['End Date']))
    #first_date = min(pd.concat([df_producer['Start Date'].dt.strftime("%Y/%m/%d"),df_consumer['Start Date'].dt.strftime("%Y/%m/%d")], axis=0))
    #last_date = max(pd.concat([df_producer['Start Date'].dt.strftime("%Y/%m/%d"),df_consumer['Start Date'].dt.strftime("%Y/%m/%d")], axis=0))
    #first_date = min(pd.concat([df_producer['Start Date'],df_consumer['Start Date']], axis=0))
    #last_date = max(pd.concat([df_producer['Start Date'],df_consumer['Start Date']], axis=0))

    # map dates to a dictionary with index
    time_list = pd.date_range(first_date, 
                                 periods=(last_date-first_date).days+1).tolist()
    model.d_t = {time_list[i]:"T_"+str(i+1) for i in range(len(time_list))}
    s_t = ["T_"+str(i+1) for i in range(len(time_list))]
    
    model.new_d_t = dict([(value, key) for key, value in model.d_t.items()])

    # ------------------- sets and parameters -----------------------
    model.s_T = Set(
        initialize=s_t, 
        doc="Time Periods", 
        ordered=True
    )

    model.s_PI = Set(
        initialize=model.df_producer['Index'], 
        doc="Producer entry index")
    
    model.s_CI = Set(
        initialize=model.df_consumer['Index'], 
        doc="Consumer entry index")
    
    model.s_PP = Set(
        initialize=model.df_producer['Wellpad'], 
        doc="Producer wellpad")
    
    model.s_CP = Set(
        initialize=model.df_consumer['Wellpad'], 
        doc="Consumer wellpad")
    

    

    # producers
    Producer_Wellpad = dict(zip(model.df_producer.Index, model.df_producer.Wellpad))
    model.p_ProducerPad = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Wellpad,
        doc="Map producer wellpad to the entry index",
    )

    Producer_Operator = dict(zip(model.df_producer.Index, model.df_producer.Operator))
    model.p_ProducerOperator = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Operator,
        doc="Map producer operator to the entry index",
    )

    Producer_Start = dict(zip(model.df_producer.Index, pd.Series([model.d_t[date] for date in df_producer['Start Date']])))
    model.p_ProducerStart = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Start,
        doc="Producer entry production start date",
    )

    Producer_End = dict(zip(model.df_producer.Index, pd.Series([model.d_t[date] for date in df_producer['End Date']])))
    model.p_ProducerEnd = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_End,
        doc="Producer entry production end date",
    )

   
    Producer_Rate = dict(zip(model.df_producer.Index, model.df_producer.Rate))
    model.p_ProducerRate = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Rate,
        units=model.model_units["volume_time"],
        doc="Producer water supply forecast [volume/time]",
    )

    Producer_Maxdis = dict(zip(model.df_producer.Index, model.df_producer['Max Transport']))
    model.p_ProducerMaxdis = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Maxdis,
        units=model.model_units["distance"],
        doc="Maximum producer trucking distance [distance]",
    )
   
    # consumers
    Consumer_Wellpad = dict(zip(model.df_consumer.Index, model.df_consumer.Wellpad))
    model.p_ConsumerPad = Param(
        model.s_CI,
        within=Any, # to suppress Pyomo warning
        initialize=Consumer_Wellpad,
        doc="Map consumer wellpad to the entry index",
    )

    Consumer_Operator = dict(zip(model.df_consumer.Index, model.df_consumer.Operator))
    model.p_ConsumerOperator = Param(
        model.s_CI,
        within=Any, # to suppress Pyomo warning
        initialize=Consumer_Operator,
        doc="Map consumer operator to the entry index",
    )

    Consumer_Start = dict(zip(model.df_consumer.Index, pd.Series([model.d_t[date] for date in df_consumer['Start Date']])))
    model.p_ConsumerStart = Param(
        model.s_CI,
        within=Any, # to suppress Pyomo warning
        initialize=Consumer_Start,
        doc="Consumer entry demand start date",
    )

    Consumer_End = dict(zip(model.df_consumer.Index, pd.Series([model.d_t[date] for date in df_consumer['End Date']])))
    model.p_ConsumerEnd = Param(
        model.s_CI,
        within=Any, # to suppress Pyomo warning
        initialize=Consumer_End,
        doc="Consumer entry demand end date",
    )

   
    Consumer_Rate = dict(zip(model.df_consumer.Index, model.df_consumer.Rate))
    model.p_ConsumerRate = Param(
        model.s_CI,
        within=Any, # to suppress Pyomo warning
        initialize=Consumer_Rate,
        units=model.model_units["volume_time"],
        doc="Consumer water demand forecast [volume/time]",
    )

    # distance
    # Convert dataframe to dictionary (for pyomo compatibility & parameter initialization)
    d_distance = df_to_dict_helper(df_distance)
    # Create initialization function for parameter
    def Arc_Distance(model, p, p_tilde):
        return d_distance[p, p_tilde]
    # Parameter definition
    model.p_ArcDistance = Param(
        model.s_PP,
        model.s_CP,
        within=NonNegativeReals, # to suppress Pyomo warning
        initialize=Arc_Distance,
        units=model.model_units["distance"],
        doc="arc trucking distance [distance]",
    )

    # time
    # Convert dataframe to dictionary (for pyomo compatibility & parameter initialization)
    d_time = df_to_dict_helper(df_time)
    # Create initialization function for parameter
    def Arc_Time(model, p, p_tilde):
        return d_time[p, p_tilde]
    # Parameter definition
    model.p_ArcTime = Param(
        model.s_PP,
        model.s_CP,
        within=NonNegativeReals, # to suppress Pyomo warning
        initialize=Arc_Time,
        units=model.model_units["time"],
        doc="arc trucking time [time]",
    )
    #model.p_ArcTime.display()

    # allowed trucking arcs - all producer to consumer arcs are allowed except
    # if they're the same wellpad, can be changed later
    """
    Note: due to a function deprecation, building sets from other sets issues a warning
    related to the use of unordered data. This can be avoided by using "list(model.s_LLT.ordered_data())"
    which is how Pyomo will work in future versions
    """
    model.s_LLT = Set(
        initialize={(pi,ci)
                    for pi in list(model.s_PI.ordered_data())
                    for ci in list(model.s_CI.ordered_data())
                    if model.p_ConsumerPad[ci]!=model.p_ProducerPad[pi] 
                    and model.p_ArcDistance[model.p_ProducerPad[pi],model.p_ConsumerPad[ci]].value 
                    <= model.p_ProducerMaxdis[pi].value},
        doc="Valid Trucking Arcs"
    )

    # MODELING PARAMETERS 
    model.DEGRADATION_LIMIT = 1.00 #0.90  # allow some degree of optimality reduction 
                                    #from 1st opt model to the 2nd model to improve 2nd model objective 
    
    max_flow_prod = max(value(model.p_ProducerRate[pi])*(
                        list(model.s_T.ordered_data()).index(model.p_ProducerEnd[pi])
                        - list(model.s_T.ordered_data()).index(model.p_ProducerStart[pi])) 
                                for pi in model.s_PI)
    max_flow_cons = max(value(model.p_ConsumerRate[ci])*(
                        list(model.s_T.ordered_data()).index(model.p_ConsumerEnd[ci])
                        - list(model.s_T.ordered_data()).index(model.p_ConsumerStart[ci])) 
                                for ci in model.s_CI)
    model.FLOWUB = max(max_flow_prod, max_flow_cons)


    ### ---------------------- Variables and set bounds -------------------------------------
 
    model.v_F_Trucked = Var(
        model.s_LLT,
        model.s_T,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water quantity trucked from location l to location l [volume/time]",
    )

    
    ### Variable Bounds
    ## start v_F_Trucked upper bound
    def p_F_Trucked_UB_init(model, pi, ci, t):
        # set upper bound default to p_gamma_Completions[p] if between p_gamma_demand_start and p_gamma_demand_end, else set to zero
        if (pi,ci) in model.s_LLT: 
            if list(model.s_T.ordered_data()).index(t) >= list(model.s_T.ordered_data()).index(model.p_ConsumerStart[ci]) \
            and list(model.s_T.ordered_data()).index(t) <= list(model.s_T.ordered_data()).index(model.p_ConsumerEnd[ci]):
                if list(model.s_T.ordered_data()).index(t) >= list(model.s_T.ordered_data()).index(model.p_ProducerStart[pi]) \
                    and list(model.s_T.ordered_data()).index(t) <= list(model.s_T.ordered_data()).index(model.p_ProducerEnd[pi]): 
                    return min(model.p_ConsumerRate[ci].value,model.p_ProducerRate[pi].value)
                else:
                    return 0
            else: 
                return 0
        else:
            return 0
        
    model.p_F_Trucked_UB = Param(
        model.s_PI,
        model.s_CI,
        model.s_T,
        within=Any,
        default=None,
        mutable=True,
        initialize=p_F_Trucked_UB_init,
        units=model.model_units["volume_time"],
        doc="Maximum trucking capacity between nodes [volume_time]",
    )
   

    for pi in model.s_PI:
        for ci in model.s_CI:
            if (pi,ci) in model.s_LLT:
                for t in model.s_T:
                    model.v_F_Trucked[(pi, ci), t].setub(model.p_F_Trucked_UB[pi, ci, t])
  
        
    ### ---------------------- Constraints ---------------------------------------------

    # -------- base constraints ----------
    # demand balance 
    def ConsumerDemandBalanceRule(model, ci, t):
        if list(model.s_T.ordered_data()).index(t)  >= list(model.s_T.ordered_data()).index(model.p_ConsumerStart[ci]) \
            and list(model.s_T.ordered_data()).index(t) <= list(model.s_T.ordered_data()).index(model.p_ConsumerEnd[ci]):
            return (sum(model.v_F_Trucked[(pi, ci), t] for pi in model.s_PI if ((pi,ci) in model.s_LLT))
                            <= model.p_ConsumerRate[ci])
            
        else:
            return Constraint.Skip

    model.ConsumerDemandBalance = Constraint(
        model.s_CI,
        model.s_T,
        rule=ConsumerDemandBalanceRule,
        doc="Consumer demand balance"
    )


    # supply balance 
    def ProducerSupplyBalanceRule(model, pi, t):
        if list(model.s_T.ordered_data()).index(t) >= list(model.s_T.ordered_data()).index(model.p_ProducerStart[pi]) \
            and list(model.s_T.ordered_data()).index(t) <= list(model.s_T.ordered_data()).index(model.p_ProducerEnd[pi]):
            return (sum(model.v_F_Trucked[(pi, ci), t] for ci in model.s_CI if ((pi,ci) in model.s_LLT))
                            <= model.p_ProducerRate[pi])
            
        else:
            return Constraint.Skip

    model.ProducerSupplyBalance = Constraint(
        model.s_PI,
        model.s_T,
        rule=ProducerSupplyBalanceRule,
        doc="Producer Supply balance"
    )

    # restrict trucking from declined match
    def TruckingUpperBound(model, pi, ci, t):
        if ((model.p_ProducerOperator[pi],model.p_ProducerPad[pi], model.p_ConsumerOperator[ci], model.p_ConsumerPad[ci]) in restricted_set):
            return model.v_F_Trucked[pi, ci, t] == 0
        else:
            return Constraint.Skip
    
    model.TruckingUpperBound = Constraint(
        model.s_PI,
        model.s_CI,
        model.s_T,
        rule=TruckingUpperBound,
        doc="Trucking Upper Bound"
    )

    # -------- add'l constraints to solve 2nd opt: minimum number of matches------------
        
    # Add binary variable to count matched entries
    model.v_match = Var(
        model.s_PI,
        model.s_CI,
        within=Binary,
        units=None,
        doc="Binary matching variable [unitless]",
    )

    # Add reformulation variables to linearize bilinear terms
    model.v1 = Var(
        model.s_PI,
        model.s_CI,
        model.s_T,
        within=NonNegativeReals,
        units=None,
        doc="Binary matching variable [unitless]",
    )
    model.v2 = Var(
        model.s_PI,
        model.s_CI,
        model.s_T,
        within=NonNegativeReals,
        units=None,
        doc="Binary matching variable [unitless]",
    )

    # add constraint to bound previous objective degradation 
    def FirstObjDegradationRule(model):
        if objective_type == "Match":
            return (sum(model.v_F_Trucked[(pi, ci), t]  for (pi,ci) in model.s_LLT for t in model.s_T)
                    >= model.DEGRADATION_LIMIT*prev_obj)
        else:
            return Constraint.Skip
                  

    model.FirstObjDegradation = Constraint(
         rule = FirstObjDegradationRule,
         doc = "bound first optimization objective in the second model"
    )

    # add constraint to determine if a match is found
    def PadtoPadMatchingRule(model,pi,ci,t):
        if objective_type == "Match":
            if (pi,ci) in model.s_LLT:
                return (model.v_F_Trucked[(pi, ci), t] <= model.FLOWUB*model.v_match[pi,ci])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip 
                  

    model.PadtoPadMatching = Constraint(
         model.s_PI,
         model.s_CI,
         model.s_T,
         rule = PadtoPadMatchingRule,
         doc = "match pad to pad"
    )

    ### Add reformulation constraints ###
    def ReformRuleNo1(model,pi,ci,t):
        if objective_type == "Match":
            if (pi,ci) in model.s_LLT:
                return (model.v1[pi,ci,t] + model.v2[pi,ci,t] == model.v_F_Trucked[pi, ci, t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.ReformConstraintNo1 = Constraint(
         model.s_PI,
         model.s_CI,
         model.s_T,
         rule = ReformRuleNo1,
         doc = "Glover reformulation constraint number 1"
    )

    def ReformRuleNo2(model,pi,ci,t):
        if objective_type == "Match":
            if (pi,ci) in model.s_LLT:
                return (model.v1[pi,ci,t] <= model.v_match[pi,ci]*model.p_F_Trucked_UB[pi, ci, t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.ReformConstraintNo2 = Constraint(
         model.s_PI,
         model.s_CI,
         model.s_T,
         rule = ReformRuleNo2,
         doc = "Glover reformulation constraint number 2"
    )

    def ReformRuleNo3(model,pi,ci,t):
        if objective_type == "Match":
            if (pi,ci) in model.s_LLT:
                return (model.v2[pi,ci,t] <= (1-model.v_match[pi,ci])*model.p_F_Trucked_UB[pi, ci, t])
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip
    model.ReformConstraintNo3 = Constraint(
         model.s_PI,
         model.s_CI,
         model.s_T,
         rule = ReformRuleNo3,
         doc = "Glover reformulation constraint number 3"
    )


    #-------------------- Define objectives ----------------------

    # Objectives
    def DefineObjective(model, objective_type):
         # maximize volume matched
        if objective_type == "Volume":
            return (sum(model.v_F_Trucked[(pi, ci), t] 
                            for (pi,ci) in model.s_LLT
                            for t in model.s_T))

        # minimizing number of matches between entries
        elif objective_type == "Match":
            return (-sum(model.v1[pi,ci,t]*model.p_ArcDistance[model.p_ProducerPad[pi],model.p_ConsumerPad[ci]]
                            for (pi,ci) in model.s_LLT for t in model.s_T))

        else:
             return 1
             

    model.objective = Objective(
        expr= DefineObjective(model,objective_type),
        sense=maximize, 
        doc='Objective function'
        )

    return model

def create_dataframe(model):
    df_v_Truck = pd.DataFrame({ 'From index': key[0],
                                'From wellpad': model.p_ProducerPad[key[0]],
                                'From operator': model.p_ProducerOperator[key[0]],
                                'To index': key[1],
                                'To wellpad': model.p_ConsumerPad[key[1]],
                                'To operator': model.p_ConsumerOperator[key[1]],
                                'Date': model.new_d_t[key[2]],
                                'Rate': value(model.v_F_Trucked[key])} 
                                        for key in model.v_F_Trucked 
                                        if value(model.v_F_Trucked[key])>0.01)
    # Timestamps (i.e., Pandas datetime format) are not JSON serializable; convert to string
    df_v_Truck['Date']= df_v_Truck['Date'].dt.strftime("%Y/%m/%d")
    return df_v_Truck