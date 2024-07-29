##################################################
# PYTHON IMPORTS
import os.path 
import pandas as pd
import numpy as np
import json

from pyomo.environ import (
    Var,
    Param,
    Set,
    ConcreteModel,
    Constraint,
    Objective,
    maximize,
    NonNegativeReals,
    Reals,
    Any,
    units as pyunits,
    value,
    BuildAction,
    Suffix,
)
from pyomo.common.config import ConfigBlock, ConfigValue, Bool
from GetDistance import get_driving_distance
from Utilities import df_to_dict_helper, add_dataframe_distance, add_dataframe_prices, GetUniqueDFData, GetDataFromGSheet


##################################################
# CREATE CONFIG DICTIONARY
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
    )
)


##################################################
# GET DATA FUNCTION
def get_data(REQUESTS,DISTANCE,Update_distance_matrix=False,OnlineMode=False,filter_ON=False,filter_date=None):
    """
    Inputs:
    > REQUESTS - directory to requests JSON file
    > DISTANCE - directory to distance JSON file
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

    if OnlineMode:
        df_consumer, df_producer, df_midstream = GetDataFromGSheet()
        df_restrictions = pd.DataFrame() # No functionality in the online mode for restrictions
        #print(filter_ON)
        #print(filter_date)
        if filter_ON:
            df_producer,df_consumer,df_midstream = FilterRequests(df_producer,df_consumer,df_midstream, filter_date)
        jsonize_GSheets_requests(REQUESTS,df_producer,df_consumer,df_midstream,df_restrictions) # save for replicability
    elif not OnlineMode:
        # Pull in JSON request data
        with open(REQUESTS, "r") as read_file:
            request_data = json.load(read_file)

            # Place producer, consumer, and midstream data into dataframes
            df_producer = pd.DataFrame(data=request_data["Producers"])
            df_consumer = pd.DataFrame(data=request_data["Consumers"])
            df_midstream = pd.DataFrame(data=request_data["Midstreams"])
            df_restrictions = pd.DataFrame(data=request_data["Restrictions"])
            if filter_ON:
                FilterRequests(df_producer,df_consumer,df_midstream, filter_date)
    else:
        print("Please enter a valid model selection.")

    # Clean and process input data
    # convert time inputs to datetime
    df_producer['Start Date'] = pd.to_datetime(df_producer['Start Date'], format ="%m/%d/%Y", errors="coerce")
    df_producer['End Date'] = pd.to_datetime(df_producer['End Date'], format ="%m/%d/%Y", errors="coerce")
    
    # convert time inputs to datetime
    df_consumer['Start Date'] = pd.to_datetime(df_consumer['Start Date'], format ="%m/%d/%Y", errors="coerce")
    df_consumer['End Date'] = pd.to_datetime(df_consumer['End Date'], format ="%m/%d/%Y", errors="coerce")
    
    # convert time inputs to datetime
    df_midstream['Start Date'] = pd.to_datetime(df_midstream['Start Date'], format ="%m/%d/%Y", errors="coerce")
    df_midstream['End Date'] = pd.to_datetime(df_midstream['End Date'], format ="%m/%d/%Y", errors="coerce")

    # Initialize dataframes (producer and consumer dataframes now created directly from JSON data)
    df_distance = pd.DataFrame()
    df_time = pd.DataFrame()
  
    # If Update_distance_matrix then get_driving_distance() else load existing distance JSON
    if Update_distance_matrix: # generate new distance matrix using API
        df_time, df_distance = get_driving_distance(DISTANCE,df_producer,df_consumer)

    else: # load existing JSON distance matrix
        with open(DISTANCE, "r") as read_file:
            distance_data = json.load(read_file)
    
        # Distance matrix parsing
        df_distance = pd.DataFrame(data=distance_data["DriveDistances"])
        df_time = pd.DataFrame(data=distance_data["DriveTimes"])

    print("input data instance created")
    return df_producer, df_consumer, df_midstream, df_restrictions, df_distance, df_time


##################################################
# CREATE MODEL FUNCTION
def create_model(restricted_set, df_producer, df_consumer, df_midstream, df_distance, df_time, default={}):
    model = ConcreteModel()

    # import config dictionary
    model.config = CONFIG(default)
    model.type = "water_clearing"

    # enable use of duals
    model.dual = Suffix(direction=Suffix.IMPORT)

    # add data frames to model
    model.df_producer = df_producer
    model.df_consumer = df_consumer
    model.df_midstream = df_midstream
    model.df_distance = df_distance
    model.df_time = df_time
    
    pyunits.load_definitions_from_strings(["USD = [currency]"])

    model.model_units = {
        "volume": pyunits.oil_bbl,
        "distance": pyunits.mile,
        #"diameter": pyunits.inch,
        #"concentration": pyunits.kg / pyunits.liter,
        "time": pyunits.day,
        "volume_time": pyunits.oil_bbl / pyunits.day,
        "currency":pyunits.USD,
        "currency_volume": pyunits.USD / pyunits.oil_bbl,
        "currency_volume-distance": pyunits.USD /(pyunits.oil_bbl*pyunits.mile),
    }

    
    # DETERMINE DATE RANGE - assume need to optimize from the earliest date to the latest date in data
    first_date = min(df_producer["Start Date"].append(df_consumer["Start Date"]).append(df_midstream["Start Date"]))
    last_date = max(df_producer['End Date'].append(df_consumer['End Date']).append(df_midstream["End Date"]))

    # map dates to a dictionary with index
    time_list = pd.date_range(first_date, periods=(last_date-first_date).days+1).tolist()
    s_t = ["T_"+str(i+1) for i in range(len(time_list))] # set of time index strings
    model.d_t = dict(zip(time_list, s_t)) # get time index from date_time value
    model.d_T = dict(zip(s_t, time_list)) # get date_time value from time index
    model.d_T_ord = dict(zip(s_t, range(len(time_list)))) # get ordinate from time index
    
    # ------------------- SETS & PARAMETERS ----------------------- #
    # SETS
    model.s_T = Set(
        initialize=s_t, #NOTE: this sets up the time index based on index strings 
        ordered=True,
        doc="Time Periods"
    )

    model.s_PI = Set(
        initialize=model.df_producer['Index'], 
        doc="Producer entry index"
    )
    
    model.s_CI = Set(
        initialize=model.df_consumer['Index'], 
        doc="Consumer entry index"
    )

    model.s_MI = Set(
        initialize=model.df_midstream["Index"],
        doc="Midstream entry index"
    )
    
    model.s_PP = Set(
        initialize=model.df_producer['Wellpad'], 
        doc="Producer wellpad"
    )
    
    model.s_CP = Set(
        initialize=model.df_consumer['Wellpad'], 
        doc="Consumer wellpad"
    )

    ### PARAMETERS
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
   
    Producer_Rate = dict(zip(model.df_producer.Index, model.df_producer["Supply Rate (bpd)"]))
    model.p_ProducerRate = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Rate,
        units=model.model_units["volume_time"],
        doc="Producer water supply forecast [volume/time]",
    )

    Producer_Supply_Bid = dict(zip(model.df_producer.Index, model.df_producer["Supplier Bid (USD/bbl)"]))
    model.p_ProducerSupplyBid = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Supply_Bid,
        units=model.model_units["currency_volume"],
        doc="Producer water supply bid [currency_volume]",
    )

    Producer_Transport_Bid = dict(zip(model.df_producer.Index, model.df_producer["Transport Bid (USD/bbl)"]))
    model.p_ProducerTransportBid = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Transport_Bid,
        units=model.model_units["currency_volume-distance"],
        doc="Producer water supply bid [currency_volume-distance]",
    )

    Producer_Maxdis = dict(zip(model.df_producer.Index, model.df_producer["Max Transport (bbl)"]))
    model.p_ProducerMaxdis = Param(
        model.s_PI,
        within=Any, # to suppress Pyomo warning
        initialize=Producer_Maxdis,
        units=model.model_units["distance"],
        doc="Maximum producer trucking distance [distance]",
    )

    # Dictionaries mapping unique wellpad entries to corresponding Lat/Lon coordinates
    model.p_ProducerWellpadLon = GetUniqueDFData(model.df_producer,"Wellpad","Longitude")
    model.p_ProducerWellpadLat = GetUniqueDFData(model.df_producer,"Wellpad","Latitude")
    model.p_ConsumerWellpadLon = GetUniqueDFData(model.df_consumer,"Wellpad","Longitude")
    model.p_ConsumerWellpadLat = GetUniqueDFData(model.df_consumer,"Wellpad","Latitude")
   
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
   
    Consumer_Rate = dict(zip(model.df_consumer.Index, model.df_consumer["Demand Rate (bpd)"]))
    model.p_ConsumerRate = Param(
        model.s_CI,
        within=Any, # to suppress Pyomo warning
        initialize=Consumer_Rate,
        units=model.model_units["volume_time"],
        doc="Consumer water demand forecast [volume/time]",
    )

    Consumer_Demand_Bid = dict(zip(model.df_consumer.Index, model.df_consumer["Consumer Bid (USD/bbl)"]))
    model.p_ConsumerDemandBid = Param(
        model.s_CI,
        within=Any, # to suppress Pyomo warning
        initialize=Consumer_Demand_Bid,
        units=model.model_units["currency_volume"],
        doc="Consumer water demand bid [currency_volume]",
    )

    # Midstreams
    Midstream_Operator = dict(zip(model.df_midstream.Index, model.df_midstream.Operator))
    model.p_MidstreamOperator = Param(
        model.s_MI,
        within=Any, # to suppress Pyomo warning
        initialize=Midstream_Operator,
        doc="Map midstream operator to the entry index",
    )

    Midstream_Start = dict(zip(model.df_midstream.Index, pd.Series([model.d_t[date] for date in df_midstream['Start Date']])))
    model.p_MidstreamStart = Param(
        model.s_MI,
        within=Any, # to suppress Pyomo warning
        initialize=Midstream_Start,
        doc="Midstream entry production start date",
    )

    Midstream_End = dict(zip(model.df_midstream.Index, pd.Series([model.d_t[date] for date in df_midstream['End Date']])))
    model.p_MidstreamEnd = Param(
        model.s_MI,
        within=Any, # to suppress Pyomo warning
        initialize=Midstream_End,
        doc="Midstream entry production end date",
    )
   
    Midstream_Rate = dict(zip(model.df_midstream.Index, model.df_midstream["Total Capacity (bbl)"]))
    model.p_MidstreamRate = Param(
        model.s_MI,
        within=Any, # to suppress Pyomo warning
        initialize=Midstream_Rate,
        units=model.model_units["volume_time"],
        doc="Midstream total capacity in bpd [volume/time]",
    )
    
    Midstream_Bid = dict(zip(model.df_midstream.Index, model.df_midstream["Transport Bid (USD/bbl)"]))
    model.p_MidstreamBid = Param(
        model.s_MI,
        within=Any, # to suppress Pyomo warning
        initialize=Midstream_Bid,
        units=model.model_units["currency_volume-distance"],
        doc="Midstream water transport bid [currency_volume-distance]",
    )

    Midstream_Maxdis = dict(zip(model.df_midstream.Index, model.df_midstream["Max Transport (bbl)"]))
    model.p_MidstreamMaxdis = Param(
        model.s_MI,
        within=Any, # to suppress Pyomo warning
        initialize=Midstream_Maxdis,
        units=model.model_units["distance"],
        doc="Maximum midstream trucking distance [distance]",
    )

    Midstream_Lag = dict(zip(model.df_midstream.Index, model.df_midstream["Lag (days)"]))
    model.p_MidstreamLag = Param(
        model.s_MI,
        within=Any,
        initialize=Midstream_Lag,
        units=model.model_units["time"],
        doc="Midstream delivery lag [time]"
    )
    
    # Distances
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

    # Times
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



    # Derived Bound Values
    max_flow_prod = max(value(model.p_ProducerRate[pi])*(
                        list(model.s_T.ordered_data()).index(model.p_ProducerEnd[pi])
                        - list(model.s_T.ordered_data()).index(model.p_ProducerStart[pi])) 
                                for pi in model.s_PI)
    max_flow_cons = max(value(model.p_ConsumerRate[ci])*(
                        list(model.s_T.ordered_data()).index(model.p_ConsumerEnd[ci])
                        - list(model.s_T.ordered_data()).index(model.p_ConsumerStart[ci])) 
                                for ci in model.s_CI)
    model.FLOWUB = max(max_flow_prod, max_flow_cons)


    ### DERIVED SETS AND SUBSETS
    """
    Note: due to a function deprecation, building sets from other sets issues a warning
    related to the use of unordered data. This can be avoided by using "list(model.s_LLT.ordered_data())"
    which is how Pyomo will work in future versions. Given we have a specific version held for PARETO
    development, this means we may need to update our code if we ever update our Pyomo version.
    """

    # Sets of time points defined by each request
    model.s_T_ci = Set(
        model.s_CI,
        dimen=1,
        initialize=dict((ci, s_t[model.d_T_ord[Consumer_Start[ci]]:(model.d_T_ord[Consumer_End[ci]]+1)]) for ci in model.s_CI),
        doc="Valid time points for each ci in s_CI"
    )
    model.s_T_pi = Set(
        model.s_PI,
        dimen=1,
        initialize=dict((pi, s_t[model.d_T_ord[Producer_Start[pi]]:(model.d_T_ord[Producer_End[pi]]+1)]) for pi in model.s_PI),
        doc="Valid time points for each pi in s_PI"
    )
    model.s_T_mi = Set(
        model.s_MI,
        dimen=1,
        initialize=dict((mi, s_t[model.d_T_ord[Midstream_Start[mi]]:(model.d_T_ord[Midstream_End[mi]]+1)]) for mi in model.s_MI),
        doc="Valid time points for each mi in s_MI"
    )

    # Create list of elements to initialize set s_LLT
    L_LLT = []
    for pi in list(model.s_PI.ordered_data()):
        for ci in list(model.s_CI.ordered_data()):
            if model.p_ProducerPad[pi] != model.p_ConsumerPad[ci] and model.p_ArcDistance[Producer_Wellpad[pi],Consumer_Wellpad[ci]].value <= model.p_ProducerMaxdis[pi].value:
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc:
                            L_LLT.append((pi,Producer_Wellpad[pi],Consumer_Wellpad[ci],tp))
    L_LLT = list(set(L_LLT))
    model.s_LLT = Set(
        dimen=4,
        initialize=L_LLT,
        doc="Valid Trucking Arcs"
    )

    # Set for arcs inbound on (cp,t)
    def s_LP_in_ct_INIT(model,cp,tc):
        elems = []
        for (pi,p,c,t) in model.s_LLT.ordered_data():
            if c == cp and t == tc:
                elems.append((pi,p,c,t))
        return elems
    model.s_LP_in_ct = Set(
        model.s_CP,
        model.s_T,
        dimen=4,
        initialize=s_LP_in_ct_INIT,
        doc="Inbound arcs on completions pad cp at time t"
    )
    print("Inbound arc set complete")
    
    # set for arcs outbound from (pp,t)
    def s_LP_out_pt_INIT(model,pp,tp):
        elems = []
        for (pi,p,c,t) in model.s_LLT.ordered_data():
            if p == pp and t == tp:
                elems.append((pi,p,c,t))
        return elems
    model.s_LP_out_pt = Set(
        model.s_PP,
        model.s_T,
        dimen=4,
        initialize=s_LP_out_pt_INIT,
        doc="Outbound arcs from production pad pp at time t"
    )
    print("Outbound arc set complete")

    # set for producer-node mapping
    def ProducerNodeMapINIT(model,p):
        pies = []
        for pi in model.s_PI:
            if Producer_Wellpad[pi] == p:
                pies.append(pi)
        return pies
    model.ProducerNodeMap = Set(
        model.s_PP,
        dimen = 1,
        initialize=ProducerNodeMapINIT,
        doc="Mapping from producer node p in s_PP to pi in s_PI (one-to-many)"
    )

    # set for producer-node-time mapping
    def ProducerNodeTimeMapINIT(model,p,t):
        peties = []
        for pi in model.s_PI:
            if Producer_Wellpad[pi] == p and t in model.s_T_pi[pi]:
                peties.append(pi)
        return peties
    model.ProducerNodeTimeMap = Set(
        model.s_PP,
        model.s_T,
        dimen = 1,
        initialize=ProducerNodeTimeMapINIT,
        doc="Mapping from producer node p in s_PP active at time t in s_T to pi in s_PI (one-to-many)"
    )

    # set for consumer-node mapping
    def ConsumerNodeMapINIT(model,c):
        cies = []
        for ci in model.s_CI:
            if Consumer_Wellpad[ci] == c:
                cies.append(ci)
        return cies
    model.ConsumerNodeMap = Set(
        model.s_CP,
        dimen = 1,
        initialize=ConsumerNodeMapINIT,
        doc="Mapping from consumer node c in s_CP to ci in s_CI (one-to-many)"
    )

    # set for consumer-node-time mapping
    def ConsumerNodeTimeMapINIT(model,c,t):
        ceties = []
        for ci in model.s_CI:
            if Consumer_Wellpad[ci] == c and t in model.s_T_ci[ci]:
                ceties.append(ci)
        return ceties
    model.ConsumerNodeTimeMap = Set(
        model.s_CP,
        model.s_T,
        dimen = 1,
        initialize=ConsumerNodeTimeMapINIT,
        doc="Mapping from consumer node c in s_CP active at time t in s_T to ci in s_CI (one-to-many)"
    )

    # set yielding active midstreams by time point
    def MidstreamTimeMapINIT(model,t):
        mies = []
        for mi in model.s_MI:
            if t in model.s_T_mi[mi]:
                mies.append(mi)
        return mies
    model.MidstreamTimeMap = Set(
        model.s_T,
        dimen=1,
        initialize=MidstreamTimeMapINIT,
        doc="All midstreams active at time t in s_T (one-to-many)"
    )

    print("Starting midstream arc set")
    # set yielding valid time lags for each midstream mi from each start point tp
    def LLM_INIT(model):
        LLMs = []
        for pi in list(model.s_PI.ordered_data()):
            for ci in list(model.s_CI.ordered_data()):
                for mi in list(model.s_MI.ordered_data()):
                    p = model.p_ProducerPad[pi]
                    c = model.p_ConsumerPad[ci]
                    if p != c:
                        for tp in model.s_T_pi[pi]:
                            for tc in model.s_T_ci[ci]:
                                if bool(model.MidstreamTimeMap[tp]) and bool(model.ProducerNodeTimeMap[p,tp]) and bool(model.ConsumerNodeTimeMap[c,tc]):
                                    if model.d_T_ord[tc] - model.d_T_ord[tp] >= 0 and model.d_T_ord[tc] - model.d_T_ord[tp] <= model.p_MidstreamLag[mi].value and model.p_ArcDistance[p,c].value <= model.p_MidstreamMaxdis[mi].value:
                                        LLMs.append((mi,p,c,tp,tc))
        """
        # Works, but slow and inefficient
        for (mi,p,c,tp,tc) in model.s_MI*model.s_PP*model.s_CP*model.s_T*model.s_T:
            if bool(model.MidstreamTimeMap[tp]) and bool(model.ProducerNodeTimeMap[p,tp]) and bool(model.ConsumerNodeTimeMap[c,tc]):
                if model.d_T_ord[tc] - model.d_T_ord[tp] <= model.p_MidstreamLag[mi].value:
                    LLMs.append((mi,p,c,tp,tc))
                else: Set.Skip
            else:
                Set.Skip
        """
        s_LLM = set(LLMs)
        return s_LLM
    model.s_LLM = Set(
        within=model.s_MI*model.s_PP*model.s_CP*model.s_T*model.s_T,
        dimen=5,
        initialize=LLM_INIT,
        doc="Valid midstream transportation arcs"
    )
    print("Midstream arc set complete")

    # Set for arcs inbound on (cp,t)
    def s_LM_in_ct_INIT(model, c_prime, t_prime):
        elems = []
        for (mi,p,c,tp,tc) in model.s_LLM.ordered_data():
            if c == c_prime and tc == t_prime:
                elems.append((mi,p,c,tp,tc))
        #elems = [(mi,p,c,tp,tc) in model.s_LLM if c == c_prime and tc == t_prime]
        return elems
    model.s_LM_in_ct = Set(
        model.s_CP,
        model.s_T,
        dimen=5,
        initialize=s_LM_in_ct_INIT,
        doc="Inbound midstream arcs on completions pad cp at time t"
    )
    print("Inbound arc set complete")
    
    # set for arcs outbound from (pp,t)
    def s_LM_out_pt_INIT(model,p_prime,t_prime):
        elems = []
        for (mi,p,c,tp,tc) in model.s_LLM:
            if p == p_prime and tp == t_prime:
                elems.append((mi,p,c,tp,tc))
        # elems = [(mi,p,c,tp,tc) in model.s_LLM if p == p_prime and tp == t_prime]
        return elems
    model.s_LM_out_pt = Set(
        model.s_PP,
        model.s_T,
        dimen=5,
        initialize=s_LM_out_pt_INIT,
        doc="Outbound arcs from production pad pp at time t"
    )
    print("Outbound arc set complete")

    # set for all outbound arcs at time tp
    def s_LM_out_mit_INIT(model, mi_prime, t_prime):
        elems = []
        for (mi,p,c,tp,tc) in model.s_LLM:
            if mi == mi_prime and tp == t_prime:
                elems.append((mi,p,c,tp,tc))
        return elems
    model.s_LM_out_mit = Set(
        model.s_MI,
        model.s_T,
        within=model.s_LLM,
        initialize=s_LM_out_mit_INIT,
        doc="Outbound arcs from all pads pp at t for mi",
    )

    # All sets done
    print("All sets complete")

    ### ---------------------- VARIABLES & BOUNDS ---------------------- #
    model.v_FP_Trucked = Var(
        model.s_LLT,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water quantity trucked from location l to location l' [volume/time] by producer pi",
    )

    model.v_FM_Trucked = Var(
        model.s_LLM,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water quantity trucked from location l to location l' [volume/time] by midstream mi",
    )

    model.v_Supply = Var(
        model.s_PI,
        model.s_T,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water volume supplied by producer request index pi at time t"
    )

    model.v_Demand = Var(
        model.s_CI,
        model.s_T,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water volume consumed by consumer request index ci at time t"
    )
    print("All variables set up")
    
    ### Variable Bounds
    """
    # v_FP_Trucked upper bound
    NodalSupply = dict(((p,t),0.0) for p in model.s_PP for t in model.s_T)
    for pi in model.s_PI:
        for t in model.s_T_pi[pi]:
            NodalSupply[Producer_Wellpad[pi],t] += Producer_Rate[pi]
    
    NodalDemand = dict(((c,t),0.0) for c in model.s_CP for t in model.s_T)
    for ci in model.s_CI:
        for t in model.s_T_ci[ci]:
            NodalDemand[Consumer_Wellpad[ci],t] += Consumer_Rate[ci]
    
    def p_FP_Trucked_UB_init(model, p, c, t):
        return min(NodalSupply[p,t], NodalDemand[c,t])
    """
    
    # update for new arc indexing:
    def p_FP_Trucked_UB_init_V2(model, pi, p, c, t):
        if tp in model.s_T_pi[pi]:
            return model.p_ProducerRate[pi]
        else:
            return 0
        
    model.p_FP_Trucked_UB = Param(
        model.s_LLT,
        within=Any,
        default=None,
        mutable=True,
        #initialize=p_FP_Trucked_UB_init,
        initialize=p_FP_Trucked_UB_init_V2,
        units=model.model_units["volume_time"],
        doc="Maximum trucking capacity between nodes [volume_time]",
    )

    # Producer transport bound
    for (pi,p,c,t) in model.s_LLT:
        model.v_FP_Trucked[pi, p, c, t].setub(model.p_FP_Trucked_UB[pi, p, c, t])

    # Midstream transport bound
    for (mi,p,c,tp,tc) in model.s_LLM:
        model.v_FM_Trucked[mi, p, c, tp, tc].setub(model.p_MidstreamRate[mi])

    # Supply bound
    for pi in model.s_PI:
        for t in model.s_T:
            if t in model.s_T_pi[pi]:
                model.v_Supply[pi, t].setub(model.p_ProducerRate[pi])
            else:
                model.v_Supply[pi, t].setub(0)

    # Demand bound
    for ci in model.s_CI:
        for t in model.s_T:
            if t in model.s_T_ci[ci]:
                model.v_Demand[ci, t].setub(model.p_ConsumerRate[ci])
            else:
                model.v_Demand[ci, t].setub(0)

    """
    # NOTE: Will need to reformulate the restriction set if it is desired in the bidding model; by going with a clearing model, we have sacrificed request-level control over shipping; we now work at the node-level, meaning we would need to block transport between nodes if a request is declined.
    # restrict trucking from declined match
    def TruckingUpperBound(model, pi, ci, t):
        if ((model.p_ProducerOperator[pi],model.p_ProducerPad[pi], model.p_ConsumerOperator[ci], model.p_ConsumerPad[ci]) in restricted_set):
            return model.v_FP_Trucked[pi, ci, t] == 0
        else:
            return Constraint.Skip
    
    model.TruckingUpperBound = Constraint(
        model.s_PI,
        model.s_CI,
        model.s_T,
        rule=TruckingUpperBound,
        doc="Trucking Upper Bound"
    )
    """
    print("All variable bounds set")

    
    ### ---------------------- Constraints ---------------------------------------------

    # -------- base constraints ----------
    # demand balance
    def ConsumerDemandBalanceRule(model, c, t):
        if bool(model.ConsumerNodeTimeMap[c,t]):
            if bool(model.s_LM_in_ct[c,t]) and bool(model.s_LP_in_ct[c,t]):
                return sum(model.v_FP_Trucked[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_in_ct[c,t]) + \
                    sum(model.v_FM_Trucked[mi, p, ca, tp, tc] for (mi, p, ca, tp, tc) in model.s_LM_in_ct[c,t]) - \
                    sum(model.v_Demand[ci,t] for ci in model.ConsumerNodeTimeMap[c,t]) == 0
            elif bool(model.s_LP_in_ct[c,t]):
                return sum(model.v_FP_Trucked[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_in_ct[c,t]) + \
                    sum(model.v_Demand[ci,t] for ci in model.ConsumerNodeTimeMap[c,t]) == 0
            elif bool(model.s_LM_in_ct[c,t]):
                return sum(model.v_FM_Trucked[mi, pa, ca, tp, tc] for (mi, pa, ca, tp, tc) in model.s_LM_in_ct[c,t]) - \
                    sum(model.v_Demand[ci,t] for ci in model.ConsumerNodeTimeMap[c,t]) == 0
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip

    model.ConsumerDemandBalance = Constraint(
        model.s_CP,
        model.s_T,
        rule=ConsumerDemandBalanceRule,
        doc="Consumer demand balance"
    )

    # supply balance 
    def ProducerSupplyBalanceRule(model, p, t):
        if bool(model.ProducerNodeTimeMap[p,t]):
            if bool(model.s_LM_out_pt[p,t]) and bool(model.s_LP_out_pt[p,t]):
                return sum(model.v_Supply[pi,t] for pi in model.ProducerNodeTimeMap[p,t]) -\
                    sum(model.v_FP_Trucked[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_out_pt[p,t]) -\
                    sum(model.v_FM_Trucked[mi, pa, ca, tp, tc] for (mi, pa, ca, tp, tc) in model.s_LM_out_pt[p,t]) == 0
            elif bool(model.s_LP_out_pt[p,t]):
                return sum(model.v_Supply[pi,t] for pi in model.ProducerNodeTimeMap[p,t]) -\
                    sum(model.v_FP_Trucked[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_out_pt[p,t]) == 0
            elif bool(model.s_LM_out_pt[p,t]):
                return sum(model.v_Supply[pi,t] for pi in model.ProducerNodeTimeMap[p,t]) -\
                    sum(model.v_FM_Trucked[mi, pa, c, tp, tc] for (mi, pa, c, tp, tc) in model.s_LM_out_pt[p,t]) == 0
            else:
                return Constraint.Skip
        else:
            return Constraint.Skip

    model.ProducerSupplyBalance = Constraint(
        model.s_PP,
        model.s_T,
        rule=ProducerSupplyBalanceRule,
        doc="Producer Supply balance"
    )

    # supply constraints
    def ProducerSupplyMaxINIT(model, pi, t):
        if t in model.s_T_pi[pi]:
            return model.v_Supply[pi,t] <= model.p_ProducerRate[pi]
        else:
            return Constraint.Skip
    model.ProducerSupplyMax = Constraint(
        model.s_PI,
        model.s_T,
        rule=ProducerSupplyMaxINIT,
        doc="Producer Supply Maximum"
    )

    # demand constraints
    def ConsumerDemandMaxINIT(model, ci, t):
        if t in model.s_T_ci[ci]:
            return model.v_Demand[ci,t] <= model.p_ConsumerRate[ci]
        else:
            return Constraint.Skip
    model.ConsumerDemandMax = Constraint(
        model.s_CI,
        model.s_T,
        rule=ConsumerDemandMaxINIT,
        doc="Producer Supply Maximum"
    )

    # midstream individual constraints
    def MidstreamRouteMaxINIT(model, mi, p, c, tp, tc):
        if (mi, p, c, tp, tc) in model.s_LLM:
            return model.v_FM_Trucked[mi, p, c, tp, tc] <= model.p_MidstreamRate[mi]
        else:
            return Constraint.Skip
    model.MidstreamRouteMax = Constraint(
        model.s_MI,
        model.s_PP,
        model.s_CP,
        model.s_T,
        model.s_T,
        rule=MidstreamRouteMaxINIT,
        doc="Midstream transport route maximum",
    )
    
    def MidstreamTotalMaxINIT(model, mi_prime, tp_prime):
        if tp_prime in model.s_T_mi[mi]:
            return sum(model.v_FM_Trucked[mi, p, c, tp, tc] for (mi,p,c,tp,tc) in model.s_LM_out_mit[mi_prime,tp_prime]) <= model.p_MidstreamRate[mi_prime]
        else:
            return Constraint.Skip
    model.MidstreamTotalMax = Constraint(
        model.s_MI,
        model.s_T,
        rule=MidstreamTotalMaxINIT,
        doc="Midstream transport total maximum",
    )

    # producer transport constraints
    def ProducerTransportMaxINIT(model, pi, p, c, t):
        if (pi,p,c,t) in model.s_LLT:
            return model.v_FP_Trucked[pi, p, c, t] <= model.p_FP_Trucked_UB[pi, p, c, t]
        else:
            return Constraint.Skip
    model.ProducerTransportMax = Constraint(
        model.s_PI,
        model.s_PP,
        model.s_CP,
        model.s_T,
        rule=ProducerTransportMaxINIT,
        doc="Producer transport maximum",
    )


    #-------------------- Define objectives ----------------------

    # Objective
    ClearingObjExpr = sum(model.p_ConsumerDemandBid[ci]*model.v_Demand[ci,t] for ci in model.s_CI for t in model.s_T_ci[ci]) - \
                    sum(model.p_ProducerSupplyBid[pi]*model.v_Supply[pi,t] for pi in model.s_PI for t in model.s_T_pi[pi]) - \
                    sum(model.p_ProducerTransportBid[pi]*model.v_FP_Trucked[pi,p,c,t] for (pi,p,c,t) in model.s_LLT) -\
                    sum(model.p_MidstreamBid[mi]*model.v_FM_Trucked[mi,p,c,tp,tc] for (mi,p,c,tp,tc) in model.s_LLM)
                    # NOTE: Old version using transport bids with units of (USD/bbl.mile)
                    #sum(model.p_ProducerTransportBid[pi]*model.p_ArcDistance[p,c]*model.v_FP_Trucked[pi,p,c,t] for (pi,p,c,t) in model.s_LLT) -\
                    #sum(model.p_MidstreamBid[mi]*model.p_ArcDistance[p,c]*model.v_FM_Trucked[mi,p,c,tp,tc] for (mi,p,c,tp,tc) in model.s_LLM)

    model.objective = Objective(
        expr= ClearingObjExpr,
        sense=maximize, 
        doc='Objective function'
        )

    print("Model setup complete")
    return model

# Create a requests JSON file (i.e., in Online Mode) to replicate runs
def jsonize_GSheets_requests(REQUESTS,df_producer,df_consumer,df_midstream,df_restrictions):
    """
    Given case data downloaded during Online Demonstration, i.e., from Google Sheets, saves that data to json for replication using OnlineMode
    """
    d_producer = df_producer.to_dict(orient='records')
    d_consumer = df_consumer.to_dict(orient='records')
    d_midstream = df_midstream.to_dict(orient='records')
    d_restrictions = df_restrictions.to_dict(orient='records')
    with open(REQUESTS, "w") as data_file:
        json.dump({"Producers":d_producer, "Consumers":d_consumer, "Midstreams":d_midstream, "Restrictions":d_restrictions}, data_file, indent=2)
    return None

# Create output dataframe
def jsonize_outputs(model, MATCHES):
    df_vP_Truck = pd.DataFrame({"Carrier Index": key[0],
                                "From wellpad": model.p_ProducerPad[key[0]],
                                "To wellpad": key[2],
                                "Date Index": key[3],
                                "Date": model.d_T[key[3]],
                                "Rate": value(model.v_FP_Trucked[key])}
                                        for key in model.v_FP_Trucked 
                                        if value(model.v_FP_Trucked[key])>0.01)
    df_vP_Truck["Distance"] = add_dataframe_distance(df_vP_Truck, model.df_distance)
    #'To index': key[1],
    #'From operator': model.p_ProducerOperator[key[0]],
    #'To operator': model.p_ConsumerOperator[key[2]],
    # Timestamps (i.e., Pandas datetime format) are not JSON serializable; convert to string
    if not df_vP_Truck.empty:
        df_vP_Truck["Price"] = [model.p_ConsumerNodalPrice[row["To wellpad"],row["Date Index"]].value -
                                model.p_ProducerNodalPrice[row["From wellpad"],row["Date Index"]].value
                                for ind,row in df_vP_Truck.iterrows()]
        df_vP_Truck["Date"] = df_vP_Truck["Date"].dt.strftime("%m/%d/%Y")
        df_vP_Truck["From Longitude"] = [model.p_ProducerWellpadLon[pad] for pad in df_vP_Truck.loc[:,"From wellpad"]]
        df_vP_Truck["From Latitude"] = [model.p_ProducerWellpadLat[pad] for pad in df_vP_Truck.loc[:,"From wellpad"]]
        df_vP_Truck["To Longitude"] = [model.p_ConsumerWellpadLon[pad] for pad in df_vP_Truck.loc[:,"To wellpad"]]
        df_vP_Truck["To Latitude"] = [model.p_ConsumerWellpadLat[pad] for pad in df_vP_Truck.loc[:,"To wellpad"]]
        df_vP_Truck["Operator"] = [model.p_ProducerOperator[index] for index in df_vP_Truck["Carrier Index"]]

    df_vM_Truck = pd.DataFrame({"Midstream Index": key[0],
                                "From wellpad": key[1],
                                "To wellpad": key[2],
                                "Departure Index": key[3],
                                "Arrival Index": key[4],
                                "Departure Date": model.d_T[key[3]],
                                "Arrival Date": model.d_T[key[4]],
                                "Rate": value(model.v_FM_Trucked[key])}
                                        for key in model.v_FM_Trucked 
                                        if value(model.v_FM_Trucked[key])>0.01)
    df_vM_Truck["Distance"] = add_dataframe_distance(df_vM_Truck, model.df_distance)
    if not df_vM_Truck.empty:
        df_vM_Truck["Price"] = [model.p_ConsumerNodalPrice[row["To wellpad"],row["Arrival Index"]].value -
                                model.p_ProducerNodalPrice[row["From wellpad"],row["Departure Index"]].value
                                for ind,row in df_vM_Truck.iterrows()]
        df_vM_Truck["Departure Date"] = df_vM_Truck["Departure Date"].dt.strftime("%m/%d/%Y")
        df_vM_Truck["Arrival Date"] = df_vM_Truck["Arrival Date"].dt.strftime("%m/%d/%Y")
        df_vM_Truck["From Longitude"] = [model.p_ProducerWellpadLon[pad] for pad in df_vM_Truck.loc[:,"From wellpad"]]
        df_vM_Truck["From Latitude"] = [model.p_ProducerWellpadLat[pad] for pad in df_vM_Truck.loc[:,"From wellpad"]]
        df_vM_Truck["To Longitude"] = [model.p_ConsumerWellpadLon[pad] for pad in df_vM_Truck.loc[:,"To wellpad"]]
        df_vM_Truck["To Latitude"] = [model.p_ConsumerWellpadLat[pad] for pad in df_vM_Truck.loc[:,"To wellpad"]]
        df_vM_Truck["Operator"] = [model.p_MidstreamOperator[index] for index in df_vM_Truck["Midstream Index"]]

    df_v_Supply = pd.DataFrame({"Supplier Index": key[0],
                                "Supplier Wellpad": model.p_ProducerPad[key[0]],
                                "Date Index": key[1],
                                "Date": model.d_T[key[1]],
                                "Rate": value(model.v_Supply[key])}
                                for key in model.v_Supply
                                if value(model.v_Supply[key])>0)
    if not df_v_Supply.empty:
        df_v_Supply["Date"]= df_v_Supply["Date"].dt.strftime("%m/%d/%Y")
        df_v_Supply["Longitude"] = [model.p_ProducerWellpadLon[pad] for pad in df_v_Supply.loc[:,"Supplier Wellpad"]]
        df_v_Supply["Latitude"] = [model.p_ProducerWellpadLat[pad] for pad in df_v_Supply.loc[:,"Supplier Wellpad"]]
        df_v_Supply["Nodal Price"] = add_dataframe_prices(df_v_Supply, model.p_ProducerNodalPrice, "Supplier")
        df_v_Supply["Operator"] = [model.p_ProducerOperator[index] for index in df_v_Supply["Supplier Index"]]

    df_v_Demand = pd.DataFrame({"Consumer Index": key[0],
                                "Consumer Wellpad": model.p_ConsumerPad[key[0]],
                                "Date Index": key[1],
                                "Date": model.d_T[key[1]],
                                "Rate": value(model.v_Demand[key])}
                                for key in model.v_Demand
                                if value(model.v_Demand[key])>0)
    if not df_v_Demand.empty:
        df_v_Demand['Date']= df_v_Demand['Date'].dt.strftime("%m/%d/%Y")
        df_v_Demand["Longitude"] = [model.p_ConsumerWellpadLon[pad] for pad in df_v_Demand.loc[:,"Consumer Wellpad"]]
        df_v_Demand["Latitude"] = [model.p_ConsumerWellpadLat[pad] for pad in df_v_Demand.loc[:,"Consumer Wellpad"]]
        df_v_Demand["Nodal Price"] = add_dataframe_prices(df_v_Demand, model.p_ConsumerNodalPrice, "Consumer")
        df_v_Demand["Operator"] = [model.p_ConsumerOperator[index] for index in df_v_Demand["Consumer Index"]]

    if df_v_Supply.empty & df_v_Demand.empty:
        print("***************************************")
        print("***************************************")
        print("*** No transactions; market is dry! ***")
        print("***************************************")
        print("***************************************")
        return None

    # convert dataframes to dictionaries for easy json output
    d_vP_Truck = df_vP_Truck.to_dict(orient='records')
    d_vM_Truck = df_vM_Truck.to_dict(orient='records')
    d_v_Supply = df_v_Supply.to_dict(orient='records')
    d_v_Demand = df_v_Demand.to_dict(orient='records')
    with open(MATCHES, "w") as data_file:
        #json.dump([d_v_Supply, d_v_Demand, d_v_Truck], data_file, indent=2)
        json.dump({"Supply":d_v_Supply, "Demand":d_v_Demand, "Transport (Producer)":d_vP_Truck, "Transport (Midstream)":d_vM_Truck}, data_file, indent=2)
    return None

# Create secondary dataframe for outputting profit
def jsonize_profits(model, PROFITS):
    # Create dataframes from paramter data
    df_ProducerProfit = pd.DataFrame({"Producer Index":key,"Profit":model.p_ProducerProfit[key].value,"Operator":model.p_ProducerOperator[key]} for key in model.p_ProducerProfit)
    df_ConsumerProfit = pd.DataFrame({"Consumer Index":key,"Profit":model.p_ConsumerProfit[key].value,"Operator":model.p_ConsumerOperator[key]} for key in model.p_ConsumerProfit)
    df_MidstreamProfit = pd.DataFrame({"Midstream Index":key,"Profit":model.p_MidstreamTotalProfit[key].value,"Operator":model.p_MidstreamOperator[key]} for key in model.p_MidstreamTotalProfit)
    df_ProducerTransportProfit = pd.DataFrame({"Producer Index":key,"Profit":model.p_ProducerTotalTransportProfit[key].value,"Operator":model.p_ProducerOperator[key]} for key in model.p_ProducerTotalTransportProfit)

    # convert to dictionaries for easy json output
    d_ProducerProfit = df_ProducerProfit.to_dict(orient='records')
    d_ConsumerProfit = df_ConsumerProfit.to_dict(orient='records')
    d_MidstreamProfit = df_MidstreamProfit.to_dict(orient='records')
    d_ProducerTransportProfit = df_ProducerTransportProfit.to_dict(orient='records')
    with open(PROFITS, "w") as data_file:
        json.dump({"Supply":d_ProducerProfit, "Demand":d_ConsumerProfit, "Transport (Producer)":d_ProducerTransportProfit, "Transport (Midstream)":d_MidstreamProfit}, data_file, indent=2)
    return None


# Post-solve calculations
def PostSolve(model):
    # Producer Nodal Price
    #ProducerNodalPriceINIT = dict.fromkeys([(p,t) for p in model.s_PP for t in model.s_T], None)
    #for index in model.ProducerSupplyBalance:
    #    ProducerNodalPriceINIT[index] = model.dual[model.ProducerSupplyBalance[index]]
    def ProducerNodalPriceINIT(model, p, t):
        try:
        # NOTE: Dual variables might be negative; may need to multiply by -1 to get meaningful results
            return -1*model.dual[model.ProducerSupplyBalance[p,t]]
        except:
            return None
    model.p_ProducerNodalPrice = Param(
        model.s_PP,
        model.s_T,
        within=Any,
        initialize=ProducerNodalPriceINIT,
        units=model.model_units["currency_volume"],
        doc="Producer Nodal Price [currency_volume]",
    )

    # Consumer Nodal Price
    def ConsumerNodalPriceINIT(model, c, t):
        try:
            return -1*value(model.dual[model.ConsumerDemandBalance[c,t]])
        except:
            return None
    model.p_ConsumerNodalPrice = Param(
        model.s_CP,
        model.s_T,
        within=Any,
        initialize=ConsumerNodalPriceINIT,
        units=model.model_units["currency_volume"],
        doc="Consumer Nodal Price [currency_volume]",
    )

    # Transport Price
    def TransportPriceINIT(model, p, c, tp, tc):
        try:
            return value(-1*model.dual[model.ConsumerDemandBalance[c,tc]] - (-1)*model.dual[model.ProducerSupplyBalance[p,tp]])
        except:
            return None
    model.p_TransportPrice = Param(
        model.s_PP,
        model.s_CP,
        model.s_T,
        model.s_T,
        within=Any,
        initialize=TransportPriceINIT,
        units=model.model_units["currency_volume"],
        doc="Transportation Price [currency_volume]",
    )

    # Supplier Profit
    def ProducerProfitINIT(model, pi):
        p = model.p_ProducerPad[pi]
        try:
            return sum((model.p_ProducerNodalPrice[p,t].value - model.p_ProducerSupplyBid[pi].value)*model.v_Supply[pi,t].value for t in model.s_T_pi[pi])
        except:
            return 0
    model.p_ProducerProfit = Param(
        model.s_PI,
        within=Any,
        initialize=ProducerProfitINIT,
        units=model.model_units["currency"],
        doc="Producer profit [currency]",
    )

    # Consumer Profit
    def ConsumerProfitINIT(model, ci):
        c = model.p_ConsumerPad[ci]
        try:
            return sum((model.p_ConsumerDemandBid[ci].value - model.p_ConsumerNodalPrice[c,t].value)*model.v_Demand[ci,t].value for t in model.s_T_ci[ci])
        except:
            return 0
    model.p_ConsumerProfit = Param(
        model.s_CI,
        within=Any,
        initialize=ConsumerProfitINIT,
        units=model.model_units["currency"],
        doc="Consumer profit [currency]",
    )

    # Midstream per-route Profit
    def MidstreamRouteProfitINIT(model, mi, p, c, tp, tc):
        try:
            if tp in model.s_T_mi[mi] and tc in model.s_T_mi[mi]:
                return (model.p_TransportPrice[p,c,tp,tc].value - model.p_MidstreamBid[mi].value)*model.v_FM_Trucked[mi,p,c,tp,tc].value
                #return (model.p_TransportPrice[p,c,tp,tc].value - model.p_MidstreamBid[mi].value*model.p_ArcDistance[p,c].value)*model.v_FM_Trucked[mi,p,c,tp,tc].value
            else:
                return 0
        except:
            return 0
    model.p_MidstreamRouteProfit = Param(
        model.s_MI,
        model.s_PP,
        model.s_CP,
        model.s_T,
        model.s_T,
        within=Any,
        initialize=MidstreamRouteProfitINIT,
        units=model.model_units["currency"],
        doc="Midstream profit [currency]",
    )

    # Midstream total Profit
    def MidstreamTotalProfitINIT(model, mi):
        return sum(model.p_MidstreamRouteProfit[mii,p,c,tp,tc] for (mii,p,c,tp,tc) in model.s_LLM if mii == mi)
    model.p_MidstreamTotalProfit = Param(
        model.s_MI,
        within=Any,
        initialize=MidstreamTotalProfitINIT,
        units=model.model_units["currency"],
        doc="Midstream total profit [currency]",
    )

    # Producer Transport per-route Profit
    def ProducerRouteProfitINIT(model, pi, p, c, t):
        try:
            if t in model.s_T_pi[pi]:
                return (model.p_TransportPrice[p,c,t,t].value - model.p_ProducerTransportBid[pi].value)*model.v_FP_Trucked[pi,p,c,t].value
                #return (model.p_TransportPrice[p,c,t,t].value - model.p_ProducerTransportBid[pi].value*model.p_ArcDistance[p,c].value)*model.v_FP_Trucked[pi,p,c,t].value
            else:
                return 0
        except:
            return 0
    model.p_ProducerRouteProfit = Param(
        model.s_PI,
        model.s_PP,
        model.s_CP,
        model.s_T,
        within=Any,
        initialize=ProducerRouteProfitINIT,
        units=model.model_units["currency"],
        doc="Producer per-route transport profit [currency]",
    )

    # Producer Transport Total Profit
    def ProducerTotalProfitINIT(model, pi):
        return sum(model.p_ProducerRouteProfit[pii,p,c,t] for (pii,p,c,t) in model.s_LLT if pii == pi)
    model.p_ProducerTotalTransportProfit = Param(
        model.s_PI,
        within=Any,
        initialize=ProducerTotalProfitINIT,
        units=model.model_units["currency"],
        doc="Producer total transport profit [currency]",
    )

    return model


# Basic data views
def DataViews(model, save_dir):
    filename = os.path.join(save_dir,"_data.txt")
    PrintSpacer = "#"*50
    fo = open(filename, "w")

    # Time points
    fo.write("\n"+PrintSpacer)
    fo.write("\nTime Index")
    for (t) in model.s_T:
        fo.write("\n"+t)

    # Suppliers
    fo.write("\n"+PrintSpacer)
    fo.write("\nSupplier Index")
    for (pi) in model.s_PI:
        fo.write("\n"+pi)

    # Consumers
    fo.write("\n"+PrintSpacer)
    fo.write("\nConsumer Index")
    for (ci) in model.s_CI:
        fo.write("\n"+ci)

    # Producer transport arcs
    fo.write("\n"+PrintSpacer)
    fo.write("\nProducer Transportation Arcs")
    for (pi,p,c,t) in model.s_LLT:
        fo.write("\n"+",".join((pi,p,c,t)))

    # Midstream transport arcs
    fo.write("\n"+PrintSpacer)
    fo.write("\nMidstream Transportation Arcs")
    for (mi,p,c,tp,tc) in model.s_LLM:
        fo.write("\n"+",".join((mi,p,c,tp,tc)))

    # Consumer set
    fo.write("\n"+PrintSpacer)
    fo.write("\nConsumer index mapping set")
    for c in model.s_CP:
        for t in model.s_T:
            fo.write("\n"+c+","+t+": "+ ",".join(model.ConsumerNodeTimeMap[c,t]))

    """
    # Consumer inbound nodes set
    fo.write("\n"+PrintSpacer)
    fo.write("\nConsumer inbound arcs index set")
    for c in model.s_CP:
        for t in model.s_T:
            fo.write("\n"+c+","+t+": "+ ",".join(model.s_LM_in_ct[c,t]))
            """
    
    fo.close()
    return None


# Basic Post-solve views
def PostSolveViews(model, save_dir):
    filename = os.path.join(save_dir,"_postsolve.txt")
    PrintSpacer = "#"*50
    fo = open(filename, "w")

    # Objective value
    fo.write("\n"+PrintSpacer)
    fo.write("\nObjective")
    fo.write("\n"+ str(value(model.objective)))

    # Producer Profits
    fo.write("\n"+PrintSpacer)
    fo.write("\nProducer Profits [currency]")
    for pi in model.s_PI:
        fo.write("\n"+ pi + ": " + str(model.p_ProducerProfit[pi].value))

    # Consumer Profits
    fo.write("\n"+PrintSpacer)
    fo.write("\nConsumer Profits [currency]")
    for ci in model.s_CI:
        fo.write("\n"+ ci + ": " + str(model.p_ConsumerProfit[ci].value))

    # Midstream per-route Profits
    fo.write("\n"+PrintSpacer)
    fo.write("\nMidstream Profits [currency]")
    for (mi,p,c,tp,tc) in model.s_LLM:
        fo.write("\n"+ ",".join((mi,p,c,tp,tc)) + ": " + str(model.p_MidstreamRouteProfit[mi,p,c,tp,tc].value))
    
    # Producer per-route Profits
    fo.write("\n"+PrintSpacer)
    fo.write("\nProducer Transportation Profits (per-route) [currency]")
    for (pi,p,c,t) in model.s_LLT:
        fo.write("\n"+ ",".join((pi,p,c,t)) + ": " + str(model.p_ProducerRouteProfit[pi,p,c,t].value))

    # Producer variable value
    fo.write("\n"+PrintSpacer)
    fo.write("\nProducer Supply Allocations [volume]")
    for pi in model.s_PI:
        for t in model.s_T_pi[pi]:
            fo.write("\n"+ pi + "," + model.p_ProducerPad[pi] + "," + t + ": " + str(model.v_Supply[pi,t].value))

    # Consumer variable value
    fo.write("\n"+PrintSpacer)
    fo.write("\nConsumer Demand Allocations [volume]")
    for ci in model.s_CI:
        for t in model.s_T_ci[ci]:
            fo.write("\n"+ ci + "," + model.p_ConsumerPad[ci] + "," + t + ": " + str(model.v_Demand[ci,t].value))

    # Midstream variable value
    fo.write("\n"+PrintSpacer)
    fo.write("\nMidstream Transport Allocations [volume]")
    for (mi,p,c,tp,tc) in model.s_LLM:
        fo.write("\n"+ ",".join((mi,p,c,tp,tc)) + ": " + str(model.v_FM_Trucked[mi,p,c,tp,tc].value))

    # Producer transport variable value
    fo.write("\n"+PrintSpacer)
    fo.write("\nProducer Transport Allocations [volume]")
    for (pi,p,c,t) in model.s_LLT:
        fo.write("\n"+ ",".join((pi,p,c,t)) + ": " + str(model.v_FP_Trucked[pi,p,c,t].value))

    # Producer nodal prices
    fo.write("\n"+PrintSpacer)
    fo.write("\nProducer Nodal Prices [currency_volume]")
    for (p,t) in model.s_PP*model.s_T:
        fo.write("\n"+ ",".join((p,t)) + ": " + str(model.p_ProducerNodalPrice[p,t].value))

    # Consumer nodal prices
    fo.write("\n"+PrintSpacer)
    fo.write("\nConsumer Nodal Prices [currency_volume]")
    for (c,t) in model.s_CP*model.s_T:
        fo.write("\n"+ ",".join((c,t)) + ": " + str(model.p_ConsumerNodalPrice[c,t].value))

    # Transportation prices
    fo.write("\n"+PrintSpacer)
    fo.write("\nTransportation Prices [currency_volume]")
    for (p,c,tp,tc) in model.s_PP*model.s_CP*model.s_T*model.s_T:
        fo.write("\n"+ ",".join((p,c,tp,tc)) + ": " + str(model.p_TransportPrice[p,c,tp,tc].value))
    
    fo.close()
    return None


# filter requests by date
def FilterRequests(df_producer,df_consumer,df_midstream, filter_date):
    # Filter by date here
    df_producer = df_producer.loc[(df_producer["Start Date"] == filter_date) & (df_producer["End Date"] == filter_date)]
    df_consumer = df_consumer.loc[(df_consumer["Start Date"] == filter_date) & (df_consumer["End Date"] == filter_date)]
    df_midstream = df_midstream.loc[(df_midstream["Start Date"] == filter_date) & (df_midstream["End Date"] == filter_date)]
    return df_producer,df_consumer,df_midstream