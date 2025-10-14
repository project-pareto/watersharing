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
    Suffix,
)
from pyomo.common.config import ConfigBlock, ConfigValue, Bool
from GetDistance import get_driving_distance, get_pipeline_distance, estimate_driving_distance
from Utilities import (
    df_to_dict_helper,
    GetUniqueDFData,
    DFRateSum,
)
import WTQuality

##################################################
# CREATE CONFIG DICTIONARY
# NOTE: Currently unsued, other than assigned modl type; remove in future?
CONFIG = ConfigBlock()
CONFIG.declare(
    "has_pipeline_constraints",
    ConfigValue(
        default=True,
        domain=Bool,
        description="build pipeline constraints",
        doc="""Indicates whether pipeline constraints should be constructed or not.
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
        description="build trucking constraints",
        doc="""Indicates whether trucking constraints should be constructed or not.
**default** - True.
**Valid values:** {
**True** - construct trucking constraints,
**False** - do not construct trucking constraints}""",
    ),
)


##################################################
# GET DATA FUNCTION
def get_data(
    request_dir,
    distance,
    update_distance_matrix=False,
    filter_by_date=None,
):
    """
    Inputs:
    > request_dir - directory to _requests.json file
    > distance_dir - directory to _distance.json file
    > update_distance_matrix - optional kwarg; if true, creates a distance matrix, if false, assumes the distance matrix exists and is CORRECT
    > filter_by_date - date by which to filter data; if None, all dates are processed; default: None
    Outputs:
    > df_producer - producer request dataframe
    > df_consumer - consumer request dataframe
    > df_road_distance - drive distance dataframe
    > df_road_time - drive time dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
        raise Exception("Data directory does not exist as received! Please check the data path.")

    # Pull in JSON request data
    with open(request_dir, "r") as read_file:
        request_data = json.load(read_file)

        # Place producer and consumer data into dataframes; force index column to string type to avoid concatenation issues later; set Index colun to index
        df_producer = pd.DataFrame(data=request_data["Producers"])
        df_producer["Index"] = df_producer["Index"].astype(str)
        df_producer.set_index("Index", inplace=True)

        df_consumer = pd.DataFrame(data=request_data["Consumers"])
        df_consumer["Index"] = df_consumer["Index"].astype(str)
        df_consumer.set_index("Index", inplace=True)

        df_restrictions = pd.DataFrame(data=request_data["Restrictions"])
        if filter_by_date is not None:
            FilterRequests(df_producer, df_consumer, filter_by_date)

    # Clean and process input data
    # convert time inputs to datetime
    df_producer["Start Date"] = pd.to_datetime(
        df_producer["Start Date"], format="%Y/%m/%d", errors="coerce"
    )
    df_producer["End Date"] = pd.to_datetime(
        df_producer["End Date"], format="%Y/%m/%d", errors="coerce"
    )

    # convert time inputs to datetime
    df_consumer["Start Date"] = pd.to_datetime(
        df_consumer["Start Date"], format="%Y/%m/%d", errors="coerce"
    )
    df_consumer["End Date"] = pd.to_datetime(
        df_consumer["End Date"], format="%Y/%m/%d", errors="coerce"
    )

    # add unique keys for wellpads as userID-wellpadID-Lon-Lat (in case multiple users have duplicate names)
    df_producer["WellpadUnique"] = (
        df_producer.index
        + "|"
        + df_producer["Wellpad"]
        + "|"
        + df_producer["Longitude"].astype(str)
        + "|"
        + df_producer["Latitude"].astype(str)
    )
    df_consumer["WellpadUnique"] = (
        df_consumer.index
        + "|"
        + df_consumer["Wellpad"]
        + "|"
        + df_consumer["Longitude"].astype(str)
        + "|"
        + df_consumer["Latitude"].astype(str)
    )

    # Initialize dataframes ()
    df_road_distance = pd.DataFrame()
    df_road_time = pd.DataFrame()

    # If update_distance_matrix then get_driving_distance() and get_pipeline_distance() else load existing distance JSON
    if update_distance_matrix:  # generate new distance matrix using API
        try: # try to retrieve accurate road distances from various API
            df_road_time, df_road_distance = get_driving_distance(distance, df_producer, df_consumer)
        except: # if for any reason we cannot retrieve the real distances, use a good estimate
            df_road_time, df_road_distance = estimate_driving_distance(distance, df_producer, df_consumer)
        df_pipe_distance = get_pipeline_distance(distance, df_producer, df_consumer)

    else:  # load existing JSON distance matrix
        with open(distance, "r") as read_file:
            distance_data = json.load(read_file)

        # Distance matrix parsing
        df_road_distance = pd.DataFrame(data=distance_data["DriveDistances"])
        df_road_time = pd.DataFrame(data=distance_data["DriveTimes"])
        df_pipe_distance = pd.DataFrame(data=distance_data["PipeDistances"])

    print("input data instance created")
    return df_producer, df_consumer, df_restrictions, df_road_distance, df_road_time, df_pipe_distance


##################################################
# CREATE MODEL FUNCTION
def create_model(
    restricted_set,
    df_producer,
    df_consumer,
    df_road_distance,
    df_road_time,
    df_pipe_distance,
    default={},
):
    model = ConcreteModel()

    # import config dictionary
    model.config = CONFIG(default)
    model.type = "water_sharing_analogue"

    # add data frames to model
    model.df_producer = df_producer
    model.df_consumer = df_consumer
    model.df_road_distance = df_road_distance
    model.df_road_time = df_road_time
    model.df_pipe_distance = df_pipe_distance

    pyunits.load_definitions_from_strings(["USD = [currency]"])

    model.model_units = {
        "volume": pyunits.oil_bbl,
        "distance": pyunits.mile,
        "time": pyunits.day,
        "volume_time": pyunits.oil_bbl / pyunits.day,
        "currency": pyunits.USD,
        "currency_volume": pyunits.USD / pyunits.oil_bbl,
        "currency_volume-distance": pyunits.USD / (pyunits.oil_bbl * pyunits.mile),
    }

    # DETERMINE DATE RANGE - assume need to optimize from the earliest date to the latest date in data
    first_date = min(df_producer["Start Date"].append(df_consumer["Start Date"]))
    last_date = max(df_producer["End Date"].append(df_consumer["End Date"]))

    # map dates to a dictionary with index
    time_list = pd.date_range(
        first_date, periods=(last_date - first_date).days + 1
    ).tolist()
    s_t = [
        "T_" + str(i + 1) for i in range(len(time_list))
    ]  # set of time index strings
    model.d_t = dict(zip(time_list, s_t))  # get time index from date_time value
    model.d_T = dict(zip(s_t, time_list))  # get date_time value from time index
    model.d_T_ord = dict(
        zip(s_t, range(len(time_list)))
    )  # get ordinate from time index

    # ------------------- SETS & PARAMETERS ----------------------- #
    # SETS
    model.s_T = Set(
        initialize=s_t,  # NOTE: this sets up the time index based on index strings
        ordered=True,
        doc="Time Periods",
    )

    model.s_PI = Set(initialize=model.df_producer.index, doc="Producer entry index")

    model.s_CI = Set(initialize=model.df_consumer.index, doc="Consumer entry index")

    model.s_PPUnique = Set(
        initialize=model.df_producer["WellpadUnique"],
        within=Any,
        doc="Producer wellpad name; not necessarily unique",
    )

    model.s_CPUnique = Set(
        initialize=model.df_consumer["WellpadUnique"],
        within=Any,
        doc="Consumer wellpad name; not necessarily unique",
    )

    ### PARAMETERS
    # Producer parameters
    Producer_Wellpad = dict(zip(model.df_producer.index, model.df_producer.Wellpad))
    model.p_ProducerPad = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Wellpad,
        doc="Map producer wellpad to the entry index",
    )

    Producer_WellpadUnique = dict(
        zip(model.df_producer.index, model.df_producer.WellpadUnique)
    )
    model.p_ProducerPadUnique = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_WellpadUnique,
        doc="Map producer wellpad to the entry index using unique keys",
    )

    # Add a reverse-lookup to the model; just to simplify finding the original name
    Producer_WellMap = dict(
        zip(model.df_producer.WellpadUnique, model.df_producer.Wellpad)
    )
    model.p_ProducerWellMap = Param(
        model.s_PPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_WellMap,
        doc="Reverse-lookup from wellpad unique ID to original name",
    )

    # Add a reverse-lookup unique wellpad to request ID
    Producer_WellIDMap = dict(
        zip(model.df_producer.WellpadUnique, model.df_producer.index)
    )
    model.p_ProducerWellIDMap = Param(
        model.s_PPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_WellIDMap,
        doc="Reverse-lookup from wellpad unique ID to request ID",
    )

    Producer_Operator = dict(zip(model.df_producer.index, model.df_producer.Operator))
    model.p_ProducerOperator = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Operator,
        doc="Map producer operator to the entry index",
    )

    Producer_Start = dict(
        zip(
            model.df_producer.index,
            pd.Series([model.d_t[date] for date in df_producer["Start Date"]]),
        )
    )
    model.p_ProducerStart = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Start,
        doc="Producer entry production start date",
    )

    Producer_End = dict(
        zip(
            model.df_producer.index,
            pd.Series([model.d_t[date] for date in df_producer["End Date"]]),
        )
    )
    model.p_ProducerEnd = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_End,
        doc="Producer entry production end date",
    )

    Producer_Rate = dict(
        zip(model.df_producer.index, model.df_producer["Supply Rate (bpd)"])
    )
    model.p_ProducerRate = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Rate,
        units=model.model_units["volume_time"],
        doc="Producer water supply forecast [volume/time]",
    )

    # Boolean values for 3rd party transport providers
    Producer_Trucks_Accepted = dict(
        zip(model.df_producer["WellpadUnique"], model.df_producer["Trucks Accepted"])
    )
    model.p_Producer_Trucks_Accepted = Param(
        model.s_PPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Trucks_Accepted,
        doc="Producer accepts 3rd party trucks [Bool]",
    )
    Producer_Pipes_Accepted = dict(
        zip(model.df_producer["WellpadUnique"], model.df_producer["Pipes Accepted"])
    )
    model.p_Producer_Pipes_Accepted = Param(
        model.s_PPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Pipes_Accepted,
        doc="Producer accepts 3rd party pipes [Bool]",
    )

    ProducerMaxRangeTruck = dict(
        zip(model.df_producer.index, model.df_producer["Truck Max Dist (mi)"])
    )
    model.p_ProducerMaxRangeTruck = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerMaxRangeTruck,
        units=model.model_units["distance"],
        doc="Maximum producer trucking range [distance]",
    )

    ProducerTransportCapacityTruck = dict(
        zip(model.df_producer.index, model.df_producer["Trucking Capacity (bpd)"])
    )
    model.p_ProducerTransportCapacityTruck = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerTransportCapacityTruck,
        units=model.model_units["volume_time"],
        doc="Producer water trucking capacity [volume_time]",
    )

    ProducerMaxRangePipel = dict(
        zip(model.df_producer.index, model.df_producer["Pipe Max Dist (mi)"])
    )
    model.p_ProducerMaxRangePipel = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerMaxRangePipel,
        units=model.model_units["distance"],
        doc="Maximum producer pipeline range [distance]",
    )

    ProducerTransportCapacityPipel = dict(
        zip(model.df_producer.index, model.df_producer["Pipeline Capacity (bpd)"])
    )
    model.p_ProducerTransportCapacityPipel = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerTransportCapacityPipel,
        units=model.model_units["volume_time"],
        doc="Producer water pipeline capacity [volume_time]",
    )

    # Consumer parameters
    Consumer_Wellpad = dict(zip(model.df_consumer.index, model.df_consumer.Wellpad))
    model.p_ConsumerPad = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Wellpad,
        doc="Map consumer wellpad to the entry index",
    )

    Consumer_WellpadUnique = dict(
        zip(model.df_consumer.index, model.df_consumer.WellpadUnique)
    )
    model.p_ConsumerPadUnique = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_WellpadUnique,
        doc="Map consumer wellpad to the entry index using unique keys",
    )

    # Add a reverse-lookup to the model; just to simplify finding the original name
    Consumer_WellMap = dict(
        zip(model.df_consumer.WellpadUnique, model.df_consumer.Wellpad)
    )
    model.p_ConsumerWellMap = Param(
        model.s_CPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_WellMap,
        doc="Reverse-lookup from wellpad unique ID to original name",
    )

    # Add a reverse-lookup unique wellpad to request ID
    Consumer_WellIDMap = dict(
        zip(model.df_consumer.WellpadUnique, model.df_consumer.index)
    )
    model.p_ConsumerWellIDMap = Param(
        model.s_CPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_WellIDMap,
        doc="Reverse-lookup from wellpad unique ID to request ID",
    )

    Consumer_Operator = dict(zip(model.df_consumer.index, model.df_consumer.Operator))
    model.p_ConsumerOperator = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Operator,
        doc="Map consumer operator to the entry index",
    )

    Consumer_Start = dict(
        zip(
            model.df_consumer.index,
            pd.Series([model.d_t[date] for date in df_consumer["Start Date"]]),
        )
    )
    model.p_ConsumerStart = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Start,
        doc="Consumer entry demand start date",
    )

    Consumer_End = dict(
        zip(
            model.df_consumer.index,
            pd.Series([model.d_t[date] for date in df_consumer["End Date"]]),
        )
    )
    model.p_ConsumerEnd = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_End,
        doc="Consumer entry demand end date",
    )

    Consumer_Rate = dict(
        zip(model.df_consumer.index, model.df_consumer["Demand Rate (bpd)"])
    )
    model.p_ConsumerRate = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Rate,
        units=model.model_units["volume_time"],
        doc="Consumer water demand forecast [volume/time]",
    )

    # Boolean values for 3rd party transport providers
    Consumer_Trucks_Accepted = dict(
        zip(model.df_consumer["WellpadUnique"], model.df_consumer["Trucks Accepted"])
    )
    model.p_Consumer_Trucks_Accepted = Param(
        model.s_CPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Trucks_Accepted,
        doc="Consumer accepts 3rd party trucks [Bool]",
    )
    Consumer_Pipes_Accepted = dict(
        zip(model.df_consumer["WellpadUnique"], model.df_consumer["Pipes Accepted"])
    )
    model.p_Consumer_Pipes_Accepted = Param(
        model.s_CPUnique,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Pipes_Accepted,
        doc="Consumer accepts 3rd party pipes [Bool]",
    )

    ConsumerMaxRangeTruck = dict(
        zip(model.df_consumer.index, model.df_consumer["Truck Max Dist (mi)"])
    )
    model.p_ConsumerMaxRangeTruck = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerMaxRangeTruck,
        units=model.model_units["distance"],
        doc="Maximum consumer trucking range [distance]",
    )

    ConsumerTransportCapacityTruck = dict(
        zip(model.df_consumer.index, model.df_consumer["Trucking Capacity (bpd)"])
    )
    model.p_ConsumerTransportCapacityTruck = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerTransportCapacityTruck,
        units=model.model_units["volume_time"],
        doc="Consumer water trucking capacity [volume_time]",
    )

    ConsumerMaxRangePipel = dict(
        zip(model.df_consumer.index, model.df_consumer["Pipe Max Dist (mi)"])
    )
    model.p_ConsumerMaxRangePipel = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerMaxRangePipel,
        units=model.model_units["distance"],
        doc="Maximum consumer pipeline range [distance]",
    )

    ConsumerTransportCapacityPipel = dict(
        zip(model.df_consumer.index, model.df_consumer["Pipeline Capacity (bpd)"])
    )
    model.p_ConsumerTransportCapacityPipel = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerTransportCapacityPipel,
        units=model.model_units["volume_time"],
        doc="Consumer water pipeline capacity [volume_time]",
    )

    # Dictionaries mapping unique wellpad entries to corresponding Lat/Lon coordinates
    model.p_ProducerWellpadLon = GetUniqueDFData(
        model.df_producer, "Wellpad", "Longitude"
    )
    model.p_ProducerWellpadLat = GetUniqueDFData(
        model.df_producer, "Wellpad", "Latitude"
    )
    model.p_ConsumerWellpadLon = GetUniqueDFData(
        model.df_consumer, "Wellpad", "Longitude"
    )
    model.p_ConsumerWellpadLat = GetUniqueDFData(
        model.df_consumer, "Wellpad", "Latitude"
    )

    # Road Distances
    # Convert dataframe to dictionary (for pyomo compatibility & parameter initialization)
    d_road_distance = df_to_dict_helper(df_road_distance)
    # Create initialization function for parameter
    def Road_Distance(model, p, p_tilde):
        return d_road_distance[p, p_tilde]

    # Parameter definition
    model.p_RoadDistance = Param(
        model.s_PPUnique,
        model.s_CPUnique,
        within=NonNegativeReals,  # to suppress Pyomo warning
        initialize=Road_Distance,
        units=model.model_units["distance"],
        doc="road arc trucking distance [distance]",
    )

    # Road Times
    # Convert dataframe to dictionary (for pyomo compatibility & parameter initialization)
    d_road_time = df_to_dict_helper(df_road_time)
    # Create initialization function for parameter
    def Road_Time(model, p, p_tilde):
        return d_road_time[p, p_tilde]

    # Parameter definition
    model.p_RoadTime = Param(
        model.s_PPUnique,
        model.s_CPUnique,
        within=NonNegativeReals,  # to suppress Pyomo warning
        initialize=Road_Time,
        units=model.model_units["time"],
        doc="road arc travel time [time]",
    )

    # Pipe Distances
    # Convert dataframe to dictionary (for pyomo compatibility & parameter initialization)
    d_pipe_distance = df_to_dict_helper(df_pipe_distance)
    # Create initialization function for parameter
    def Pipe_Distance(model, p, p_tilde):
        return d_pipe_distance[p, p_tilde]

    # Parameter definition
    model.p_PipeDistance = Param(
        model.s_PPUnique,
        model.s_CPUnique,
        within=NonNegativeReals,  # to suppress Pyomo warning
        initialize=Pipe_Distance,
        units=model.model_units["distance"],
        doc="arc pipeline distance [distance]",
    )

    ### DERIVED SETS AND SUBSETS
    """
    Note: due to a function deprecation, building sets from other sets issues a warning
    related to the use of unordered data. This can be avoided by using "list(model.s_LTP.ordered_data())"
    which is how Pyomo will work in future versions. Given a specific version of Pyomo is used for PARETO
    development, this means the code will need to be updated if the Pyomo version is updated.
    """

    # Sets of time points defined by each request
    model.s_T_ci = Set(
        model.s_CI,
        dimen=1,
        initialize=dict(
            (
                ci,
                s_t[
                    model.d_T_ord[Consumer_Start[ci]] : (
                        model.d_T_ord[Consumer_End[ci]] + 1
                    )
                ],
            )
            for ci in model.s_CI
        ),
        doc="Valid time points for each ci in s_CI",
    )
    model.s_T_pi = Set(
        model.s_PI,
        dimen=1,
        initialize=dict(
            (
                pi,
                s_t[
                    model.d_T_ord[Producer_Start[pi]] : (
                        model.d_T_ord[Producer_End[pi]] + 1
                    )
                ],
            )
            for pi in model.s_PI
        ),
        doc="Valid time points for each pi in s_PI",
    )

    ### Quality setup
    match_qual_dict = {}

    # Set LP(pi,p,c,t) of arcs owned & operated by producers PI
    L_LP_truck = (
        []
    )  # Create list of elements to initialize set s_LP_truck, comprises indices (pi,pp,cp,t)
    L_LP_pipel = (
        []
    )  # Create list of elements to initialize set s_LP_pipel, comprises indices (pi,pp,cp,t)
    # Set LC(ci,p,c,t) of arcs owned & operated by consumers CI
    L_LC_truck = (
        []
    )  # Create list of elements to initialize set s_LC_truck, comprises indices (ci,pp,cp,t)
    L_LC_pipel = (
        []
    )  # Create list of elements to initialize set s_LC_pipel, comprises indices (ci,pp,cp,t)
    for pi in list(model.s_PI.ordered_data()):
        for ci in list(model.s_CI.ordered_data()):
            # LP_truck
            quality_check = WTQuality.match_quality_check(pi,ci,df_producer,df_consumer,match_qual_dict)
            if (
                model.p_ProducerPadUnique[pi]
                != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_RoadDistance[
                    Producer_WellpadUnique[pi], Consumer_WellpadUnique[ci]
                ].value
                <= model.p_ProducerMaxRangeTruck[pi].value
                and model.p_ProducerTransportCapacityTruck[pi].value >= 0
                and quality_check
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc:  # i.e., if any times overlap
                            L_LP_truck.append(
                                (
                                    pi,
                                    Producer_WellpadUnique[pi],
                                    Consumer_WellpadUnique[ci],
                                    tp,
                                )
                            )
            # LP_pipel
            quality_check = WTQuality.match_quality_check(pi,ci,df_producer,df_consumer,match_qual_dict)
            if (
                model.p_ProducerPadUnique[pi]
                != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_PipeDistance[
                    Producer_WellpadUnique[pi], Consumer_WellpadUnique[ci]
                ].value
                <= model.p_ProducerMaxRangePipel[pi].value
                and model.p_ProducerTransportCapacityPipel[pi].value >= 0
                and quality_check
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc:  # i.e., if any times overlap
                            L_LP_pipel.append(
                                (
                                    pi,
                                    Producer_WellpadUnique[pi],
                                    Consumer_WellpadUnique[ci],
                                    tp,
                                )
                            )
            # LC_truck
            quality_check = WTQuality.match_quality_check(pi,ci,df_producer,df_consumer,match_qual_dict)
            if (
                model.p_ProducerPadUnique[pi]
                != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_RoadDistance[
                    Producer_WellpadUnique[pi], Consumer_WellpadUnique[ci]
                ].value
                <= model.p_ConsumerMaxRangeTruck[ci].value
                and model.p_ConsumerTransportCapacityTruck[ci].value >= 0
                and quality_check
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc:  # i.e., if any times overlap
                            L_LC_truck.append(
                                (
                                    ci,
                                    Producer_WellpadUnique[pi],
                                    Consumer_WellpadUnique[ci],
                                    tc,
                                )
                            )
            # LC_pipel
            quality_check = WTQuality.match_quality_check(pi,ci,df_producer,df_consumer,match_qual_dict)
            if (
                model.p_ProducerPadUnique[pi]
                != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_PipeDistance[
                    Producer_WellpadUnique[pi], Consumer_WellpadUnique[ci]
                ].value
                <= model.p_ConsumerMaxRangePipel[ci].value
                and model.p_ConsumerTransportCapacityPipel[ci].value >= 0
                and quality_check
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc:  # i.e., if any times overlap
                            L_LC_pipel.append(
                                (
                                    ci,
                                    Producer_WellpadUnique[pi],
                                    Consumer_WellpadUnique[ci],
                                    tc,
                                )
                            )
    L_LP_truck = list(set(L_LP_truck))  # remove duplciates
    L_LP_pipel = list(set(L_LP_pipel))  # remove duplciates
    L_LC_truck = list(set(L_LC_truck))  # remove duplciates
    L_LC_pipel = list(set(L_LC_pipel))  # remove duplciates
    model.s_LP_truck = Set(
        dimen=4, initialize=L_LP_truck, doc="Valid Producer Trucking Arcs"
    )
    model.s_LP_pipel = Set(
        dimen=4, initialize=L_LP_pipel, doc="Valid Producer Pipeline Arcs"
    )
    model.s_LC_truck = Set(
        dimen=4, initialize=L_LC_truck, doc="Valid Consumer Trucking Arcs"
    )
    model.s_LC_pipel = Set(
        dimen=4, initialize=L_LC_pipel, doc="Valid Consumer Pipeline Arcs"
    )
    model.match_qual_dict = match_qual_dict # will need this later; easiest way is to attach it to the model object
    print("Primary arc sets constructed")

    # Sets for arcs inbound on (cp,t)
    def s_LP_truck_in_ct_INIT(model, cp, tc):
        elems = []
        for (pi, p, c, t) in model.s_LP_truck.ordered_data():
            if c == cp and t == tc and model.p_Consumer_Trucks_Accepted[cp]:
                elems.append((pi, p, c, t))
        return elems

    model.s_LP_truck_in_ct = Set(
        model.s_CPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LP_truck_in_ct_INIT,
        doc="Inbound producer trucking arcs on completions pad cp at time t",
    )

    def s_LP_pipel_in_ct_INIT(model, cp, tc):
        elems = []
        for (pi, p, c, t) in model.s_LP_pipel.ordered_data():
            if c == cp and t == tc and model.p_Consumer_Pipes_Accepted[cp]:
                elems.append((pi, p, c, t))
        return elems

    model.s_LP_pipel_in_ct = Set(
        model.s_CPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LP_pipel_in_ct_INIT,
        doc="Inbound producer pipeline arcs on completions pad cp at time t",
    )

    def s_LC_truck_in_ct_INIT(model, cp, tc):
        elems = []
        for (ci, p, c, t) in model.s_LC_truck.ordered_data():
            if c == cp and t == tc and model.p_Producer_Trucks_Accepted[p]:
                elems.append((ci, p, c, t))
        return elems

    model.s_LC_truck_in_ct = Set(
        model.s_CPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LC_truck_in_ct_INIT,
        doc="Inbound consumer trucking arcs on completions pad cp at time t",
    )

    def s_LC_pipel_in_ct_INIT(model, cp, tc):
        elems = []
        for (ci, p, c, t) in model.s_LC_pipel.ordered_data():
            if c == cp and t == tc and model.p_Producer_Pipes_Accepted[p]:
                elems.append((ci, p, c, t))
        return elems

    model.s_LC_pipel_in_ct = Set(
        model.s_CPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LC_pipel_in_ct_INIT,
        doc="Inbound consumer pipeline arcs on completions pad cp at time t",
    )
    print("Inbound arc sets complete")

    # Sets for arcs outbound from (pp,t)
    def s_LP_truck_out_pt_INIT(model, pp, tp):
        elems = []
        for (pi, p, c, t) in model.s_LP_truck.ordered_data():
            if p == pp and t == tp and model.p_Consumer_Trucks_Accepted[c]:
                elems.append((pi, p, c, t))
        return elems

    model.s_LP_truck_out_pt = Set(
        model.s_PPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LP_truck_out_pt_INIT,
        doc="Outbound producer trucking arcs from production pad pp at time t",
    )

    def s_LP_pipel_out_pt_INIT(model, pp, tp):
        elems = []
        for (pi, p, c, t) in model.s_LP_pipel.ordered_data():
            if p == pp and t == tp and model.p_Consumer_Pipes_Accepted[c]:
                elems.append((pi, p, c, t))
        return elems

    model.s_LP_pipel_out_pt = Set(
        model.s_PPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LP_pipel_out_pt_INIT,
        doc="Outbound producer pipeline arcs from production pad pp at time t",
    )

    def s_LC_truck_out_pt_INIT(model, pp, tp):
        elems = []
        for (ci, p, c, t) in model.s_LC_truck.ordered_data():
            if p == pp and t == tp and model.p_Producer_Trucks_Accepted[pp]:
                elems.append((ci, p, c, t))
        return elems

    model.s_LC_truck_out_pt = Set(
        model.s_PPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LC_truck_out_pt_INIT,
        doc="Outbound consumer trucking arcs from production pad pp at time t",
    )

    def s_LC_pipel_out_pt_INIT(model, pp, tp):
        elems = []
        for (ci, p, c, t) in model.s_LC_pipel.ordered_data():
            if p == pp and t == tp and model.p_Producer_Pipes_Accepted[pp]:
                elems.append((ci, p, c, t))
        return elems

    model.s_LC_pipel_out_pt = Set(
        model.s_PPUnique,
        model.s_T,
        dimen=4,
        initialize=s_LC_pipel_out_pt_INIT,
        doc="Outbound consumer pipeline arcs from production pad pp at time t",
    )
    print("Outbound arc sets complete")

    # set for producer-node mapping
    def ProducerNodeMapINIT(model, p):
        pies = []
        for pi in model.s_PI:
            if model.p_ProducerPadUnique[pi] == p:
                pies.append(pi)
        return pies

    model.ProducerNodeMap = Set(
        model.s_PPUnique,
        dimen=1,
        initialize=ProducerNodeMapINIT,
        doc="Mapping from producer node p in s_PP to pi in s_PI (one-to-many)",
    )

    # set for producer-node-time mapping
    def ProducerNodeTimeMapINIT(model, p, t):
        peties = []
        for pi in model.s_PI:
            if model.p_ProducerPadUnique[pi] == p and t in model.s_T_pi[pi]:
                peties.append(pi)
        return peties

    model.ProducerNodeTimeMap = Set(
        model.s_PPUnique,
        model.s_T,
        dimen=1,
        initialize=ProducerNodeTimeMapINIT,
        doc="Mapping from producer node p in s_PP active at time t in s_T to pi in s_PI (one-to-many)",
    )

    # set for consumer-node mapping
    def ConsumerNodeMapINIT(model, c):
        cies = []
        for ci in model.s_CI:
            if model.p_ConsumerPadUnique[ci] == c:
                cies.append(ci)
        return cies

    model.ConsumerNodeMap = Set(
        model.s_CPUnique,
        dimen=1,
        initialize=ConsumerNodeMapINIT,
        doc="Mapping from consumer node c in s_CP to ci in s_CI (one-to-many)",
    )

    # set for consumer-node-time mapping
    def ConsumerNodeTimeMapINIT(model, c, t):
        ceties = []
        for ci in model.s_CI:
            if model.p_ConsumerPadUnique[ci] == c and t in model.s_T_ci[ci]:
                ceties.append(ci)
        return ceties

    model.ConsumerNodeTimeMap = Set(
        model.s_CPUnique,
        model.s_T,
        dimen=1,
        initialize=ConsumerNodeTimeMapINIT,
        doc="Mapping from consumer node c in s_CP active at time t in s_T to ci in s_CI (one-to-many)",
    )

    # All sets done
    print("All sets complete")

    ### ---------------------- VARIABLES & BOUNDS ---------------------- #
    # Variables for water volumes; note: the F in v_F indicates transport (flow) volume, v_FP_truck indicates producer trucking (i.e., v_(node_i,node_j,producer_index)) while v_FC_pipel indicates consumer piping (i.e., v_(node_i,node_j,consumer_index))
    model.v_FP_truck = Var(
        model.s_LP_truck,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water quantity trucked from location l to location l' [volume/time] by producer pi",
    )

    model.v_FP_pipel = Var(
        model.s_LP_pipel,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water quantity piped from location l to location l' [volume/time] by producer pi",
    )

    model.v_FC_truck = Var(
        model.s_LC_truck,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water quantity trucked from location l to location l' [volume/time] by consumer ci",
    )

    model.v_FC_pipel = Var(
        model.s_LC_pipel,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water quantity piped from location l to location l' [volume/time] by consumer ci",
    )

    model.v_Supply = Var(
        model.s_PI,
        model.s_T,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water volume supplied by producer request index pi at time t",
    )

    model.v_Demand = Var(
        model.s_CI,
        model.s_T,
        within=NonNegativeReals,
        initialize=0,
        units=model.model_units["volume_time"],
        doc="Produced water volume consumed by consumer request index ci at time t",
    )
    print("All variables set up")

    ### Variable Bounds
    # Producer trucking bound
    def p_FP_truck_UB_init(model, pi, p, c, tp):
        if tp in model.s_T_pi[pi] and model.p_Consumer_Trucks_Accepted[c]:
            return model.p_ProducerTransportCapacityTruck[pi]
        else:
            return 0

    model.p_FP_truck_UB = Param(
        model.s_LP_truck,
        within=Any,
        default=None,
        mutable=True,
        initialize=p_FP_truck_UB_init,
        units=model.model_units["volume_time"],
        doc="Maximum producer trucking capacity between nodes [volume_time]",
    )

    for (pi, p, c, t) in model.s_LP_truck:
        model.v_FP_truck[pi, p, c, t].setub(model.p_FP_truck_UB[pi, p, c, t])

    # Producer piping bound
    def p_FP_pipel_UB_init(model, pi, p, c, tp):
        if tp in model.s_T_pi[pi] and model.p_Consumer_Pipes_Accepted[c]:
            return model.p_ProducerTransportCapacityPipel[pi]
        else:
            return 0

    model.p_FP_pipel_UB = Param(
        model.s_LP_pipel,
        within=Any,
        default=None,
        mutable=True,
        initialize=p_FP_pipel_UB_init,
        units=model.model_units["volume_time"],
        doc="Maximum producer piping capacity between nodes [volume_time]",
    )

    for (pi, p, c, t) in model.s_LP_pipel:
        model.v_FP_pipel[pi, p, c, t].setub(model.p_FP_pipel_UB[pi, p, c, t])

    # Consumer trucking bound
    def p_FC_truck_UB_init(model, ci, p, c, tc):
        if tc in model.s_T_ci[ci] and model.p_Producer_Trucks_Accepted[p]:
            return model.p_ConsumerTransportCapacityTruck[ci]
        else:
            return 0

    model.p_FC_truck_UB = Param(
        model.s_LC_truck,
        within=Any,
        default=None,
        mutable=True,
        initialize=p_FC_truck_UB_init,
        units=model.model_units["volume_time"],
        doc="Maximum consumer trucking capacity between nodes [volume_time]",
    )

    for (ci, p, c, t) in model.s_LC_truck:
        model.v_FC_truck[ci, p, c, t].setub(model.p_FC_truck_UB[ci, p, c, t])

    # Consumer piping bound
    def p_FC_pipel_UB_init(model, ci, p, c, tc):
        if tc in model.s_T_ci[ci] and model.p_Producer_Pipes_Accepted[p]:
            return model.p_ConsumerTransportCapacityPipel[ci]
        else:
            return 0

    model.p_FC_pipel_UB = Param(
        model.s_LC_pipel,
        within=Any,
        default=None,
        mutable=True,
        initialize=p_FC_pipel_UB_init,
        units=model.model_units["volume_time"],
        doc="Maximum consumer piping capacity between nodes [volume_time]",
    )

    for (ci, p, c, t) in model.s_LC_pipel:
        model.v_FC_pipel[ci, p, c, t].setub(model.p_FC_pipel_UB[ci, p, c, t])

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

    print("All variable bounds set")

    ### ---------------------- Constraints ---------------------------------------------

    # -------- base constraints ----------
    # demand balance
    def ConsumerDemandBalanceRule(model, c, t):
        if bool(model.ConsumerNodeTimeMap[c, t]):
            expr = -sum(model.v_Demand[ci, t] for ci in model.ConsumerNodeTimeMap[c, t])

            if bool(model.s_LP_truck_in_ct[c, t]):
                expr += sum(
                    model.v_FP_truck[pi, pa, ca, t]
                    for (pi, pa, ca, t) in model.s_LP_truck_in_ct[c, t]
                )

            if bool(model.s_LP_pipel_in_ct[c, t]):
                expr += sum(
                    model.v_FP_pipel[pi, pa, ca, t]
                    for (pi, pa, ca, t) in model.s_LP_pipel_in_ct[c, t]
                )

            if bool(model.s_LC_truck_in_ct[c, t]):
                expr += sum(
                    model.v_FC_truck[ci, pa, ca, t]
                    for (ci, pa, ca, t) in model.s_LC_truck_in_ct[c, t]
                )

            if bool(model.s_LC_pipel_in_ct[c, t]):
                expr += sum(
                    model.v_FC_pipel[ci, pa, ca, t]
                    for (ci, pa, ca, t) in model.s_LC_pipel_in_ct[c, t]
                )

            return expr == 0
        else:
            return Constraint.Skip

    model.ConsumerDemandBalance = Constraint(
        model.s_CPUnique,
        model.s_T,
        rule=ConsumerDemandBalanceRule,
        doc="Consumer demand balance",
    )

    # supply balance
    def ProducerSupplyBalanceRule(model, p, t):
        if bool(model.ProducerNodeTimeMap[p, t]):
            expr = sum(model.v_Supply[pi, t] for pi in model.ProducerNodeTimeMap[p, t])

            if bool(model.s_LP_truck_out_pt[p, t]):
                expr += -sum(
                    model.v_FP_truck[pi, pa, ca, t]
                    for (pi, pa, ca, t) in model.s_LP_truck_out_pt[p, t]
                )

            if bool(model.s_LP_pipel_out_pt[p, t]):
                expr += -sum(
                    model.v_FP_pipel[pi, pa, ca, t]
                    for (pi, pa, ca, t) in model.s_LP_pipel_out_pt[p, t]
                )

            if bool(model.s_LC_truck_out_pt[p, t]):
                expr += -sum(
                    model.v_FC_truck[ci, pa, ca, t]
                    for (ci, pa, ca, t) in model.s_LC_truck_out_pt[p, t]
                )

            if bool(model.s_LC_pipel_out_pt[p, t]):
                expr += -sum(
                    model.v_FC_pipel[ci, pa, ca, t]
                    for (ci, pa, ca, t) in model.s_LC_pipel_out_pt[p, t]
                )

            return expr == 0
        else:
            return Constraint.Skip

    model.ProducerSupplyBalance = Constraint(
        model.s_PPUnique,
        model.s_T,
        rule=ProducerSupplyBalanceRule,
        doc="Producer Supply balance",
    )

    # supply constraints
    def ProducerSupplyMaxINIT(model, pi, t):
        if t in model.s_T_pi[pi]:
            return model.v_Supply[pi, t] <= model.p_ProducerRate[pi]
        else:
            return Constraint.Skip

    model.ProducerSupplyMax = Constraint(
        model.s_PI, model.s_T, rule=ProducerSupplyMaxINIT, doc="Producer Supply Maximum"
    )

    # demand constraints
    def ConsumerDemandMaxINIT(model, ci, t):
        if t in model.s_T_ci[ci]:
            return model.v_Demand[ci, t] <= model.p_ConsumerRate[ci]
        else:
            return Constraint.Skip

    model.ConsumerDemandMax = Constraint(
        model.s_CI, model.s_T, rule=ConsumerDemandMaxINIT, doc="Producer Supply Maximum"
    )

    # producer transport constraints
    def ProducerTruckingMaxINIT(model, pi, p, c, t):
        if (pi, p, c, t) in model.s_LP_truck:
            return model.v_FP_truck[pi, p, c, t] <= model.p_FP_truck_UB[pi, p, c, t]
        else:
            return Constraint.Skip

    model.ProducerTruckingMax = Constraint(
        model.s_PI,
        model.s_PPUnique,
        model.s_CPUnique,
        model.s_T,
        rule=ProducerTruckingMaxINIT,
        doc="Producer trucking maximum",
    )

    def ProducerPipingMaxINIT(model, pi, p, c, t):
        if (pi, p, c, t) in model.s_LP_pipel:
            return model.v_FP_pipel[pi, p, c, t] <= model.p_FP_pipel_UB[pi, p, c, t]
        else:
            return Constraint.Skip

    model.ProducerPipingMax = Constraint(
        model.s_PI,
        model.s_PPUnique,
        model.s_CPUnique,
        model.s_T,
        rule=ProducerPipingMaxINIT,
        doc="Producer piping maximum",
    )

    # consumer transport constraints
    def ConsumerTruckingMaxINIT(model, ci, p, c, t):
        if (ci, p, c, t) in model.s_LC_truck:
            return model.v_FC_truck[ci, p, c, t] <= model.p_FC_truck_UB[ci, p, c, t]
        else:
            return Constraint.Skip

    model.ConsumerTruckingMax = Constraint(
        model.s_CI,
        model.s_PPUnique,
        model.s_CPUnique,
        model.s_T,
        rule=ConsumerTruckingMaxINIT,
        doc="Consumer trucking maximum",
    )

    def ConsumerPipingMaxINIT(model, ci, p, c, t):
        if (ci, p, c, t) in model.s_LC_pipel:
            return model.v_FC_pipel[ci, p, c, t] <= model.p_FC_pipel_UB[ci, p, c, t]
        else:
            return Constraint.Skip

    model.ConsumerPipingMax = Constraint(
        model.s_CI,
        model.s_PPUnique,
        model.s_CPUnique,
        model.s_T,
        rule=ConsumerPipingMaxINIT,
        doc="Consumer piping maximum",
    )

    # -------------------- Define objectives ----------------------

    # Objective
    SharingObjExpr = (
        sum(
            ((1+model.p_ProducerTransportCapacityTruck[pi])/(1+model.p_RoadDistance[p,c])) * model.v_FP_truck[pi, p, c, t]
            for (pi, p, c, t) in model.s_LP_truck
        )
        + sum(
            ((1+model.p_ProducerTransportCapacityPipel[pi])/(1+model.p_PipeDistance[p,c])) * model.v_FP_pipel[pi, p, c, t]
            for (pi, p, c, t) in model.s_LP_pipel
        )
        + sum(
            ((1+model.p_ConsumerTransportCapacityTruck[ci])/(1+model.p_RoadDistance[p,c])) * model.v_FC_truck[ci, p, c, t]
            for (ci, p, c, t) in model.s_LC_truck
        )
        + sum(
            ((1+model.p_ConsumerTransportCapacityPipel[ci])/(1+model.p_PipeDistance[p,c])) * model.v_FC_pipel[ci, p, c, t]
            for (ci, p, c, t) in model.s_LC_pipel
        )
    )

    model.objective = Objective(
        expr=SharingObjExpr, sense=maximize, doc="Objective function"
    )

    print("Model setup complete")
    return model


# Create output dataframe
def jsonize_outputs(
    model,
    matches_dir,
    match_threshold=0.0,
    na_string="-",
    temp_qual_val="All specifications met",
):
    # Convert output variables to DataFrames for filtering
    df_FP_truck = pd.DataFrame(
        {
            "Supplier Index": key[0],
            "Supplier Wellpad": key[1],
            "Consumer Wellpad": key[2],
            "Date Index": key[3],
            "Rate": value(model.v_FP_truck[key]),
        }
        for key in model.v_FP_truck
        if value(model.v_FP_truck[key]) > match_threshold
    )
    df_FP_pipel = pd.DataFrame(
        {
            "Supplier Index": key[0],
            "Supplier Wellpad": key[1],
            "Consumer Wellpad": key[2],
            "Date Index": key[3],
            "Rate": value(model.v_FP_pipel[key]),
        }
        for key in model.v_FP_pipel
        if value(model.v_FP_pipel[key]) > match_threshold
    )
    df_FC_truck = pd.DataFrame(
        {
            "Consumer Index": key[0],
            "Supplier Wellpad": key[1],
            "Consumer Wellpad": key[2],
            "Date Index": key[3],
            "Rate": value(model.v_FC_truck[key]),
        }
        for key in model.v_FC_truck
        if value(model.v_FC_truck[key]) > match_threshold
    )
    df_FC_pipel = pd.DataFrame(
        {
            "Consumer Index": key[0],
            "Supplier Wellpad": key[1],
            "Consumer Wellpad": key[2],
            "Date Index": key[3],
            "Rate": value(model.v_FC_pipel[key]),
        }
        for key in model.v_FC_pipel
        if value(model.v_FC_pipel[key]) > match_threshold
    )
    df_v_Supply = pd.DataFrame(
        {
            "Supplier Index": key[0],
            "Supplier Wellpad": model.p_ProducerPadUnique[key[0]],
            "Date Index": key[1],
            "Date": model.d_T[key[1]],
            "Rate": value(model.v_Supply[key]),
        }
        for key in model.v_Supply
        if value(model.v_Supply[key]) > match_threshold
    )
    df_v_Supply_totals = pd.pivot_table(
        df_v_Supply, values="Rate", index="Supplier Index", columns=None, aggfunc="sum"
    )
    df_v_Demand = pd.DataFrame(
        {
            "Consumer Index": key[0],
            "Consumer Wellpad": model.p_ConsumerPadUnique[key[0]],
            "Date Index": key[1],
            "Date": model.d_T[key[1]],
            "Rate": value(model.v_Demand[key]),
        }
        for key in model.v_Demand
        if value(model.v_Demand[key]) > 0
    )
    df_v_Demand_totals = pd.pivot_table(
        df_v_Demand, values="Rate", index="Consumer Index", columns=None, aggfunc="sum"
    )

    # If both supply and demand are empty (either they both are, or neither will be, actually) end here; the market is dry.
    if df_v_Supply.empty and df_v_Demand.empty:
        print("***************************************")
        print("***************************************")
        print("*** No matches satisfy the requests ***")
        print("***************************************")
        print("***************************************")
        return None

    # Iterate over requests, identify matches, and add lines to output supply match dataframe
    d_supply_match = {
        "Index": [],
        "Pair Index": [],
        "Operator": [],
        "UserID": [],
        "Wellpad": [],
        "Longitude": [],
        "Latitude": [],
        "Start Date": [],
        "End Date": [],
        "Supply Rate (bpd)": [],
        "Trucks Accepted": [],
        "Pipes Accepted": [],
        "Truck Max Dist (mi)": [],
        "Trucking Capacity (bpd)": [],
        "Pipe Max Dist (mi)": [],
        "Pipeline Capacity (bpd)": [],
        "Matches": [],
        "Match Total Volume (bbl)":[],
    }

    d_demand_match = {
        "Index": [],
        "Pair Index": [],
        "Operator": [],
        "UserID": [],
        "Wellpad": [],
        "Longitude": [],
        "Latitude": [],
        "Start Date": [],
        "End Date": [],
        "Demand Rate (bpd)": [],
        "Trucks Accepted": [],
        "Pipes Accepted": [],
        "Truck Max Dist (mi)": [],
        "Trucking Capacity (bpd)": [],
        "Pipe Max Dist (mi)": [],
        "Pipeline Capacity (bpd)": [],
        "Matches": [],
        "Match Total Volume (bbl)":[],
    }

    # iterate over supplier requests
    for pi in model.s_PI:
        # for future reference: the monstrosity below checks whether any of the elements in v_Supply_totals' index column are the element in question. The element must be in square brackets, and the axis input to any() is required to return a scalar boolean value.
        if (
            df_v_Supply_totals.index.isin([pi]).any(axis=None)
            and df_v_Supply_totals.loc[pi, "Rate"] > 0
        ):  # there's a match for this request; we just need to figure out within which of the four transport tables
            p = model.p_ProducerPadUnique[pi]
            for c in model.s_CPUnique:
                # Filter all the matches to consumer pads c; these may be empty (v_Supply(pi)>0 does not imply every c will be matched)
                if not df_FP_truck.empty:
                    df_FP_truck_filter = df_FP_truck.loc[
                        (df_FP_truck["Supplier Index"] == pi)
                        & (df_FP_truck["Consumer Wellpad"] == c)
                    ]
                else: # if it is empty
                    df_FP_truck_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                if not df_FP_pipel.empty:
                    df_FP_pipel_filter = df_FP_pipel.loc[
                        (df_FP_pipel["Supplier Index"] == pi)
                        & (df_FP_pipel["Consumer Wellpad"] == c)
                    ]
                else: # if it is empty
                    df_FP_pipel_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                # TODO: check to see if this won't double count entries; might need to filter using a set; p might not be associated with this pi
                # This should be addressed by the Unique Wellpad indexing; confirm nevertheless.
                if not df_FC_truck.empty:
                    df_FC_truck_filter = df_FC_truck.loc[
                        (df_FC_truck["Supplier Wellpad"] == p)
                        & (df_FC_truck["Consumer Wellpad"] == c)
                    ]
                else: # if it is empty
                    df_FC_truck_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                if not df_FC_pipel.empty:
                    df_FC_pipel_filter = df_FC_pipel.loc[
                        (df_FC_pipel["Supplier Wellpad"] == p)
                        & (df_FC_pipel["Consumer Wellpad"] == c)
                    ]
                else: # if it is empty
                    df_FC_pipel_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                # Is there a match with this element, c?
                if not (
                    df_FP_truck_filter.empty
                    and df_FP_pipel_filter.empty
                    and df_FC_truck_filter.empty
                    and df_FC_pipel_filter.empty
                ):  # there's a match with this destination
                    # populate main-level match detail dictionary
                    d_supply_match["Index"].append(pi)
                    d_supply_match["Pair Index"].append(
                        pi + "-" + model.p_ConsumerWellIDMap[c]
                    )
                    d_supply_match["Operator"].append(
                        model.df_producer.loc[pi, "Operator"]
                    )
                    d_supply_match["UserID"].append(
                        model.df_producer.loc[pi, "UserID"]
                    )
                    d_supply_match["Wellpad"].append(
                        model.df_producer.loc[pi, "Wellpad"]
                    )
                    d_supply_match["Longitude"].append(
                        model.df_producer.loc[pi, "Longitude"]
                    )
                    d_supply_match["Latitude"].append(
                        model.df_producer.loc[pi, "Latitude"]
                    )
                    d_supply_match["Start Date"].append(
                        model.df_producer.loc[pi, "Start Date"]
                    )
                    d_supply_match["End Date"].append(
                        model.df_producer.loc[pi, "End Date"]
                    )
                    d_supply_match["Supply Rate (bpd)"].append(
                        model.df_producer.loc[pi, "Supply Rate (bpd)"]
                    )
                    d_supply_match["Trucks Accepted"].append(
                        model.df_producer.loc[pi, "Trucks Accepted"]
                    )
                    d_supply_match["Pipes Accepted"].append(
                        model.df_producer.loc[pi, "Pipes Accepted"]
                    )
                    d_supply_match["Truck Max Dist (mi)"].append(
                        model.df_producer.loc[pi, "Truck Max Dist (mi)"]
                    )
                    d_supply_match["Trucking Capacity (bpd)"].append(
                        model.df_producer.loc[pi, "Trucking Capacity (bpd)"]
                    )
                    d_supply_match["Pipe Max Dist (mi)"].append(
                        model.df_producer.loc[pi, "Pipe Max Dist (mi)"]
                    )
                    d_supply_match["Pipeline Capacity (bpd)"].append(
                        model.df_producer.loc[pi, "Pipeline Capacity (bpd)"]
                    )
                    # create sub-level match detail dictionary
                    d_match_detail = {
                        "Match Index": [],
                        "Match Date Index": [],
                        "Match Date": [],
                        "Match Volume": [],
                        "Providing Trucking Volume": [],
                        "Providing Piping Volume": [],
                        "Obtaining Trucking Volume": [], # edge case; not necessarily a "purchase"
                        "Obtaining Piping Volume": [],
                        "Quality": [],
                    }
                    # Iterate through the possible time points and find matches
                    for t in model.s_T_pi[pi]:
                        # if value(model.v_Supply[pi,t]) > match_threshold: # there's a match for this request at this time
                        if not df_FP_truck.empty:
                            df_FP_truck_filter_t = df_FP_truck.loc[
                                (df_FP_truck["Supplier Index"] == pi)
                                & (df_FP_truck["Consumer Wellpad"] == c)
                                & (df_FP_truck["Date Index"] == t)
                            ]
                        else:
                            df_FP_truck_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        if not df_FP_pipel.empty:
                            df_FP_pipel_filter_t = df_FP_pipel.loc[
                                (df_FP_pipel["Supplier Index"] == pi)
                                & (df_FP_pipel["Consumer Wellpad"] == c)
                                & (df_FP_pipel["Date Index"] == t)
                            ]
                        else:
                            df_FP_pipel_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        # TODO: check to see if this won't double count entries; might need to filter using a set; p might not be associated with this pi
                        if not df_FC_truck.empty:
                            df_FC_truck_filter_t = df_FC_truck.loc[
                                (df_FC_truck["Supplier Wellpad"] == p)
                                & (df_FC_truck["Consumer Wellpad"] == c)
                                & (df_FC_truck["Date Index"] == t)
                            ]
                        else:
                            df_FC_truck_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        if not df_FC_pipel.empty:
                            df_FC_pipel_filter_t = df_FC_pipel.loc[
                                (df_FC_pipel["Supplier Wellpad"] == p)
                                & (df_FC_pipel["Consumer Wellpad"] == c)
                                & (df_FC_pipel["Date Index"] == t)
                            ]
                        else:
                            df_FC_pipel_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        # Is there a match with this element, c, at this time, t?
                        if not (
                            df_FP_truck_filter_t.empty
                            and df_FP_pipel_filter_t.empty
                            and df_FC_truck_filter_t.empty
                            and df_FC_pipel_filter_t.empty
                        ):
                            # calculations NOTE: for supplier, negative price means paying, positive means getting paid, so the negative is consistent
                            FP_truck = DFRateSum(df_FP_truck_filter_t)
                            FP_pipel = DFRateSum(df_FP_pipel_filter_t)
                            FC_truck = DFRateSum(df_FC_truck_filter_t)
                            FC_pipel = DFRateSum(df_FC_pipel_filter_t)
                            match_volume = FP_truck + FP_pipel + FC_truck + FC_pipel
                            # record entries
                            d_match_detail["Match Index"].append(pi + "-" + c + "-" + t)
                            d_match_detail["Match Date Index"].append(t)
                            d_match_detail["Match Date"].append(model.d_T[t])
                            d_match_detail["Match Volume"].append(match_volume)
                            if (
                                FP_truck > 0
                            ):
                                d_match_detail["Providing Trucking Volume"].append(
                                    FP_truck
                                )
                            else:
                                d_match_detail["Providing Trucking Volume"].append(
                                    na_string
                                )
                            if FP_pipel > 0:
                                d_match_detail["Providing Piping Volume"].append(
                                    FP_pipel
                                )
                            else:
                                d_match_detail["Providing Piping Volume"].append(
                                    na_string
                                )
                            if FC_truck > 0:
                                d_match_detail["Obtaining Trucking Volume"].append(
                                    FC_truck
                                )
                            else:
                                d_match_detail["Obtaining Trucking Volume"].append(
                                    na_string
                                )
                            if FC_pipel > 0:
                                d_match_detail["Obtaining Piping Volume"].append(
                                    FC_pipel
                                )
                            else:
                                d_match_detail["Obtaining Piping Volume"].append(
                                    na_string
                                )
                            d_match_detail["Quality"].append(model.match_qual_dict[pi,model.p_ConsumerWellIDMap[c],"supply side"])
                        # end if not [all empty] here
                    # Add integrator terms
                    d_supply_match["Match Total Volume (bbl)"].append(sum(d_match_detail["Match Volume"]))
                    # Finally, push match details to Supply match dictionary
                    d_supply_match["Matches"].append(d_match_detail)

    # iterate over consumer requests
    for ci in model.s_CI:
        if (
            df_v_Demand_totals.index.isin([ci]).any(axis=None)
            and df_v_Demand_totals.loc[ci, "Rate"] > 0
        ):  # there's a match for this request; we just need to figure out within which of the four transport tables
            c = model.p_ConsumerPadUnique[ci]
            for p in model.s_PPUnique:
                # Filter all the matches to producer pads p; these may be empty (v_Demand(ci)>0 does not imply every p will be matched)
                # TODO: check to see if this won't double count entries; might need to filter using a set; c might not be associated with this ci
                if not df_FP_truck.empty:
                    df_FP_truck_filter = df_FP_truck.loc[
                        (df_FP_truck["Supplier Wellpad"] == p)
                        & (df_FP_truck["Consumer Wellpad"] == c)
                    ]
                else: # if it is empty
                    df_FP_truck_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                if not df_FP_pipel.empty:
                    df_FP_pipel_filter = df_FP_pipel.loc[
                        (df_FP_pipel["Supplier Wellpad"] == p)
                        & (df_FP_pipel["Consumer Wellpad"] == c)
                    ]
                else: # if it is empty
                    df_FP_pipel_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                if not df_FC_truck.empty:
                    df_FC_truck_filter = df_FC_truck.loc[
                        (df_FC_truck["Supplier Wellpad"] == p)
                        & (df_FC_truck["Consumer Index"] == ci)
                    ]
                else: # if it is empty
                    df_FC_truck_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                if not df_FC_pipel.empty:
                    df_FC_pipel_filter = df_FC_pipel.loc[
                        (df_FC_pipel["Supplier Wellpad"] == p)
                        & (df_FC_pipel["Consumer Index"] == ci)
                    ]
                else: # if it is empty
                    df_FC_pipel_filter = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                # Is there a match with this element, p?
                if not (
                    df_FP_truck_filter.empty
                    and df_FP_pipel_filter.empty
                    and df_FC_truck_filter.empty
                    and df_FC_pipel_filter.empty
                ):  # there's a match with this destination
                    # populate main-level match detail dictionary
                    d_demand_match["Index"].append(ci)
                    d_demand_match["Pair Index"].append(
                        ci + "-" + model.p_ProducerWellIDMap[p]
                    )
                    d_demand_match["Operator"].append(
                        model.df_consumer.loc[ci, "Operator"]
                    )
                    d_demand_match["UserID"].append(model.df_consumer.loc[ci, "UserID"])
                    d_demand_match["Wellpad"].append(
                        model.df_consumer.loc[ci, "Wellpad"]
                    )
                    d_demand_match["Longitude"].append(
                        model.df_consumer.loc[ci, "Longitude"]
                    )
                    d_demand_match["Latitude"].append(
                        model.df_consumer.loc[ci, "Latitude"]
                    )
                    d_demand_match["Start Date"].append(
                        model.df_consumer.loc[ci, "Start Date"]
                    )
                    d_demand_match["End Date"].append(
                        model.df_consumer.loc[ci, "End Date"]
                    )
                    d_demand_match["Demand Rate (bpd)"].append(
                        model.df_consumer.loc[ci, "Demand Rate (bpd)"]
                    )
                    d_demand_match["Trucks Accepted"].append(
                        model.df_consumer.loc[ci, "Trucks Accepted"]
                    )
                    d_demand_match["Pipes Accepted"].append(
                        model.df_consumer.loc[ci, "Pipes Accepted"]
                    )
                    d_demand_match["Truck Max Dist (mi)"].append(
                        model.df_consumer.loc[ci, "Truck Max Dist (mi)"]
                    )
                    d_demand_match["Trucking Capacity (bpd)"].append(
                        model.df_consumer.loc[ci, "Trucking Capacity (bpd)"]
                    )
                    d_demand_match["Pipe Max Dist (mi)"].append(
                        model.df_consumer.loc[ci, "Pipe Max Dist (mi)"]
                    )
                    d_demand_match["Pipeline Capacity (bpd)"].append(
                        model.df_consumer.loc[ci, "Pipeline Capacity (bpd)"]
                    )
                    # create sub-level match detail dictionary
                    d_match_detail = {
                        "Match Index": [],
                        "Match Date Index": [],
                        "Match Date": [],
                        "Match Volume": [],
                        "Providing Trucking Volume": [],
                        "Providing Piping Volume": [],
                        "Obtaining Trucking Volume": [],
                        "Obtaining Piping Volume": [],
                        "Quality": [],
                    }
                    # Iterate through the possible time points and find matches
                    for t in model.s_T_ci[ci]:
                        # if value(model.v_Demand[ci,t]) > match_threshold: # there's a match for this request at this time
                        # TODO: check to see if this won't double count entries; might need to filter using a set; c might not be associated with this ci
                        if not df_FP_truck.empty:
                            df_FP_truck_filter_t = df_FP_truck.loc[
                                (df_FP_truck["Supplier Wellpad"] == p)
                                & (df_FP_truck["Consumer Wellpad"] == c)
                                & (df_FP_truck["Date Index"] == t)
                            ]
                        else:
                            df_FP_truck_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        if not df_FP_pipel.empty:
                            df_FP_pipel_filter_t = df_FP_pipel.loc[
                                (df_FP_pipel["Supplier Wellpad"] == p)
                                & (df_FP_pipel["Consumer Wellpad"] == c)
                                & (df_FP_pipel["Date Index"] == t)
                            ]
                        else:
                            df_FP_pipel_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        if not df_FC_truck.empty:
                            df_FC_truck_filter_t = df_FC_truck.loc[
                                (df_FC_truck["Supplier Wellpad"] == p)
                                & (df_FC_truck["Consumer Index"] == ci)
                                & (df_FC_truck["Date Index"] == t)
                            ]
                        else:
                            df_FC_truck_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        if not df_FC_pipel.empty:
                            df_FC_pipel_filter_t = df_FC_pipel.loc[
                                (df_FC_pipel["Supplier Wellpad"] == p)
                                & (df_FC_pipel["Consumer Index"] == ci)
                                & (df_FC_pipel["Date Index"] == t)
                            ]
                        else:
                            df_FC_pipel_filter_t = pd.DataFrame() # passing an empty dataframe is the simplest way to proceed
                        # Is there a match with this element, p, at this time, t?
                        if not (
                            df_FP_truck_filter_t.empty
                            and df_FP_pipel_filter_t.empty
                            and df_FC_truck_filter_t.empty
                            and df_FC_pipel_filter_t.empty
                        ):
                            # calculations NOTE: for consumer, negative price means getting paid, positive means paying, so the negative is logically reversed
                            FP_truck = DFRateSum(df_FP_truck_filter_t)
                            FP_pipel = DFRateSum(df_FP_pipel_filter_t)
                            FC_truck = DFRateSum(df_FC_truck_filter_t)
                            FC_pipel = DFRateSum(df_FC_pipel_filter_t)
                            match_volume = FP_truck + FP_pipel + FC_truck + FC_pipel
                            # record entries
                            d_match_detail["Match Index"].append(ci + "-" + p + "-" + t)
                            d_match_detail["Match Date Index"].append(t)
                            d_match_detail["Match Date"].append(model.d_T[t])
                            d_match_detail["Match Volume"].append(match_volume)
                            if (
                                FC_truck > 0
                            ):
                                d_match_detail["Providing Trucking Volume"].append(
                                    FC_truck
                                )
                            else:
                                d_match_detail["Providing Trucking Volume"].append(
                                    na_string
                                )
                            if FC_pipel > 0:
                                d_match_detail["Providing Piping Volume"].append(
                                    FC_pipel
                                )
                            else:
                                d_match_detail["Providing Piping Volume"].append(
                                    na_string
                                )
                            if FP_truck > 0:
                                d_match_detail["Obtaining Trucking Volume"].append(
                                    FP_truck
                                )
                            else:
                                d_match_detail["Obtaining Trucking Volume"].append(
                                    na_string
                                )
                            if FP_pipel > 0:
                                d_match_detail["Obtaining Piping Volume"].append(
                                    FP_pipel
                                )
                            else:
                                d_match_detail["Obtaining Piping Volume"].append(
                                    na_string
                                )
                            d_match_detail["Quality"].append(model.match_qual_dict[model.p_ProducerWellIDMap[p],ci,"demand side"])
                        # end if not [all empty] here
                    # Add integrator terms
                    d_demand_match["Match Total Volume (bbl)"].append(sum(d_match_detail["Match Volume"]))
                    # Finally, push match details to Demand match dictionary
                    d_demand_match["Matches"].append(d_match_detail)

    # convert buildup dictionaries to dataframes
    df_supply_match = pd.DataFrame.from_dict(d_supply_match)
    df_demand_match = pd.DataFrame.from_dict(d_demand_match)

    # convert dataframes to dictionaries for easy json output (and correct format)
    d_supply_match_out = df_supply_match.to_dict(orient="records")
    d_demand_match_out = df_demand_match.to_dict(orient="records")

    with open(matches_dir, "w") as data_file:
        json.dump(
            {
                "Supply": d_supply_match_out,
                "Demand": d_demand_match_out,
            },
            data_file,
            indent=4,
            default=str,
        )
    return None



# Basic data views
def DataViews(model, save_dir):
    filename = os.path.join(save_dir, "_data.txt")
    PrintSpacer = "#" * 50
    fo = open(filename, "w")

    # Time points
    fo.write("\n" + PrintSpacer)
    fo.write("\nTime Index")
    for t in model.s_T:
        fo.write("\n" + t)

    # Suppliers
    fo.write("\n" + PrintSpacer)
    fo.write("\nSupplier Index")
    for pi in model.s_PI:
        fo.write("\n" + pi)

    # Consumers
    fo.write("\n" + PrintSpacer)
    fo.write("\nConsumer Index")
    for ci in model.s_CI:
        fo.write("\n" + ci)

    # Producer transport arcs
    fo.write("\n" + PrintSpacer)
    fo.write("\nProducer Transportation Arcs")
    for (pi, p, c, t) in model.s_LP_truck:
        fo.write("\n" + ",".join((pi, p, c, t)))

    # Consumer set
    fo.write("\n" + PrintSpacer)
    fo.write("\nConsumer index mapping set")
    for c in model.s_CPUnique:
        for t in model.s_T:
            fo.write(
                "\n" + c + "," + t + ": " + ",".join(model.ConsumerNodeTimeMap[c, t])
            )

    fo.close()
    return None


# Basic Post-solve views
def PostSolveViews(model, save_dir):
    filename = os.path.join(save_dir, "_postsolve.txt")
    PrintSpacer = "#" * 50
    fo = open(filename, "w")

    # Objective value
    fo.write("\n" + PrintSpacer)
    fo.write("\nObjective")
    fo.write("\n" + str(value(model.objective)))

    # Producer variable value
    fo.write("\n" + PrintSpacer)
    fo.write("\nProducer Supply Allocations [volume]")
    for pi in model.s_PI:
        for t in model.s_T_pi[pi]:
            fo.write(
                "\n"
                + pi
                + ","
                + model.p_ProducerPadUnique[pi]
                + ","
                + t
                + ": "
                + str(model.v_Supply[pi, t].value)
            )

    # Consumer variable value
    fo.write("\n" + PrintSpacer)
    fo.write("\nConsumer Demand Allocations [volume]")
    for ci in model.s_CI:
        for t in model.s_T_ci[ci]:
            fo.write(
                "\n"
                + ci
                + ","
                + model.p_ConsumerPadUnique[ci]
                + ","
                + t
                + ": "
                + str(model.v_Demand[ci, t].value)
            )

    # Producer transport variable value
    fo.write("\n" + PrintSpacer)
    fo.write("\nProducer Transport Allocations [volume]")
    for (pi, p, c, t) in model.s_LP_truck:
        fo.write(
            "\n"
            + ",".join((pi, p, c, t))
            + ": "
            + str(model.v_FP_truck[pi, p, c, t].value)
        )

    fo.close()
    return None


# filter requests by date
def FilterRequests(df_producer, df_consumer, filter_by_date):
    # Filter by date here
    df_producer = df_producer.loc[
        (df_producer["Start Date"] == filter_by_date)
        & (df_producer["End Date"] == filter_by_date)
    ]
    df_consumer = df_consumer.loc[
        (df_consumer["Start Date"] == filter_by_date)
        & (df_consumer["End Date"] == filter_by_date)
    ]
    return df_producer, df_consumer


# Output user matches to individual folder: /io/watertrading/match-detail/[userID]
def OutputMatchesToUsers(matches_dir, output_dir):
    """
    Outputs match details to folders by UserID within directory output_dir
    Inputs:
    - matches_dir: match file name & directory
    - output_dir: directory to output folder, defaults to match-detail if not specified
    Outputs:
    - None
    """

    # Make sure the output folder is there, or create it if not
    if not os.path.exists(output_dir):
        os.makedirs(os.path.join(output_dir))

    # Load data from JSON file
    with open(matches_dir, "r") as match_data:
        match_data = json.load(match_data)  # load data as dictionary

    # Convert match data to dataframe
    df_matches_s = pd.DataFrame(data=match_data["Supply"])
    n_supply = len(match_data["Supply"])
    for i in range(n_supply):
        # Get User ID
        UserID = match_data["Supply"][i]["UserID"]
        # Get match details to export and convert to dataframe
        df_match_details = pd.DataFrame.from_dict(df_matches_s.loc[i]["Matches"])
        df_match_details.drop(
           columns = ["Match Index","Match Date Index"], axis=1, inplace=True
        )  # remove identifying information
        # Check if output folder exists
        if not os.path.exists(os.path.join(output_dir, UserID)):
            os.mkdir(os.path.join(output_dir, UserID))
        # Create file name from Match Index and export
        output_file_name = os.path.join(
            output_dir, UserID, df_matches_s.loc[i]["Pair Index"] + ".csv"
        )
        df_match_details.to_csv(output_file_name, index=False)

    # Convert match data to dataframe
    df_matches_d = pd.DataFrame(data=match_data["Demand"])
    n_demand = len(match_data["Demand"])
    for i in range(n_demand):
        # Get User ID
        UserID = match_data["Demand"][i]["UserID"]
        # Get match details to export and convert to dataframe
        df_match_details = pd.DataFrame.from_dict(df_matches_d.loc[i]["Matches"])
        df_match_details.drop(
            columns = ["Match Index","Match Date Index"], axis=1, inplace=True
        )  # remove identifying information
        # Check if output folder exists
        if not os.path.exists(os.path.join(output_dir, UserID)):
            os.mkdir(os.path.join(output_dir, UserID))
        # Create file name from Match Index and export
        output_file_name = os.path.join(
            output_dir, UserID, df_matches_d.loc[i]["Pair Index"] + ".csv"
        )
        df_match_details.to_csv(output_file_name, index=False)

    return None
