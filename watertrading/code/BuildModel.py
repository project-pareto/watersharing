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
from Utilities import (
    df_to_dict_helper,
    add_dataframe_distance,
    add_dataframe_prices,
    GetUniqueDFData
)


##################################################
# CREATE CONFIG DICTIONARY
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
    > filter_by_date - date by which to filter data; if None, all dates are processed
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
    with open(request_dir, "r") as read_file:
        request_data = json.load(read_file)

        # Place producer and consumer data into dataframes
        df_producer = pd.DataFrame(data=request_data["Producers"])
        df_consumer = pd.DataFrame(data=request_data["Consumers"])
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
    df_producer["WellpadUnique"] = df_producer["Index"] + df_producer["Wellpad"] + df_producer["Longitude"].astype(str) + df_producer["Latitude"].astype(str)
    df_consumer["WellpadUnique"] = df_consumer["Index"] + df_consumer["Wellpad"] + df_producer["Longitude"].astype(str)+ df_producer["Latitude"].astype(str)

    # Initialize dataframes ()
    df_distance = pd.DataFrame()
    df_time = pd.DataFrame()

    # If update_distance_matrix then get_driving_distance() else load existing distance JSON
    if update_distance_matrix:  # generate new distance matrix using API
        df_time, df_distance = get_driving_distance(distance, df_producer, df_consumer)

    else:  # load existing JSON distance matrix
        with open(distance, "r") as read_file:
            distance_data = json.load(read_file)

        # Distance matrix parsing
        df_distance = pd.DataFrame(data=distance_data["DriveDistances"])
        df_time = pd.DataFrame(data=distance_data["DriveTimes"])

    print("input data instance created")
    return df_producer, df_consumer, df_restrictions, df_distance, df_time


##################################################
# CREATE MODEL FUNCTION
def create_model(
    restricted_set,
    df_producer,
    df_consumer,
    df_distance,
    df_time,
    default={},
):
    model = ConcreteModel()

    # import config dictionary
    model.config = CONFIG(default)
    model.type = "water_clearing"

    # enable use of duals
    model.dual = Suffix(direction=Suffix.IMPORT)

    # add data frames to model
    model.df_producer = df_producer
    model.df_consumer = df_consumer
    model.df_distance = df_distance
    model.df_time = df_time

    pyunits.load_definitions_from_strings(["USD = [currency]"])

    model.model_units = {
        "volume": pyunits.oil_bbl,
        "distance": pyunits.mile,
        # "diameter": pyunits.inch,
        # "concentration": pyunits.kg / pyunits.liter,
        "time": pyunits.day,
        "volume_time": pyunits.oil_bbl / pyunits.day,
        "currency": pyunits.USD,
        "currency_volume": pyunits.USD / pyunits.oil_bbl,
        "currency_volume-distance": pyunits.USD / (pyunits.oil_bbl * pyunits.mile),
    }

    # DETERMINE DATE RANGE - assume need to optimize from the earliest date to the latest date in data
    first_date = min(
        df_producer["Start Date"]
        .append(df_consumer["Start Date"])
    )
    last_date = max(
        df_producer["End Date"]
        .append(df_consumer["End Date"])
    )

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

    model.s_PI = Set(
        initialize=model.df_producer["Index"],
        doc="Producer entry index"
    )

    model.s_CI = Set(
        initialize=model.df_consumer["Index"],
        doc="Consumer entry index"
    )

    model.s_PP = Set(
        initialize=model.df_producer["Wellpad"],
        doc="Producer wellpad"
    )

    model.s_CP = Set(
        initialize=model.df_consumer["Wellpad"],
        doc="Consumer wellpad"
    )

    model.s_PPUnique = Set(
        initialize=model.df_producer["WellpadUnique"],
        doc="Producer wellpad name; not necessarily unique"
    )

    model.s_CPUnique = Set(
        initialize=model.df_consumer["WellpadUnique"],
        doc="Consumer wellpad name; not necessarily unique"
    )

    ### PARAMETERS
    # Producer parameters
    Producer_Wellpad = dict(zip(model.df_producer.Index, model.df_producer.Wellpad))
    model.p_ProducerPad = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Wellpad,
        doc="Map producer wellpad to the entry index",
    )

    Producer_WellpadUnique = dict(zip(model.df_producer.Index, model.df_producer.WellpadUnique))
    model.p_ProducerPadUnique = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_WellpadUnique,
        doc="Map producer wellpad to the entry index using unique keys",
    )

    Producer_Operator = dict(
        zip(
            model.df_producer.Index,
            model.df_producer.Operator
        )
    )
    model.p_ProducerOperator = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Operator,
        doc="Map producer operator to the entry index",
    )

    Producer_Start = dict(
        zip(
            model.df_producer.Index,
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
            model.df_producer.Index,
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
        zip(
            model.df_producer.Index,
            model.df_producer["Supply Rate (bpd)"]
        )
    )
    model.p_ProducerRate = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Rate,
        units=model.model_units["volume_time"],
        doc="Producer water supply forecast [volume/time]",
    )

    Producer_Supply_Bid = dict(
        zip(
            model.df_producer.Index,
            model.df_producer["Supplier Bid (USD/bbl)"]
        )
    )
    model.p_ProducerSupplyBid = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=Producer_Supply_Bid,
        units=model.model_units["currency_volume"],
        doc="Producer water supply bid [currency_volume]",
    )

    ProducerMaxRangeTruck = dict(
        zip(
            model.df_producer.Index,
            model.df_producer["Truck Max Dist (mi)"]
        )
    )
    model.p_ProducerMaxRangeTruck = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerMaxRangeTruck,
        units=model.model_units["distance"],
        doc="Maximum producer trucking range [distance]",
    )

    ProducerTransportCapacityTruck = dict(
        zip(
            model.df_producer.Index,
            model.df_producer["Trucking Capacity (bpd)"]
        )
    )
    model.p_ProducerTransportCapacityTruck = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerTransportCapacityTruck,
        units=model.model_units["volume_time"],
        doc="Producer water trucking capacity [volume_time]",
    )

    ProducerTransportBidTruck = dict(
        zip(
            model.df_producer.Index,
            model.df_producer["Truck Transport Bid (USD/bbl)"]
        )
    )
    model.p_ProducerTransportBidTruck = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerTransportBidTruck,
        units=model.model_units["currency_volume"],
        doc="Producer water trucking bid [currency_volume]",
    )

    ProducerMaxRangePipel = dict(
        zip(
            model.df_producer.Index,
            model.df_producer["Pipe Max Dist (mi)"]
        )
    )
    model.p_ProducerMaxRangePipel = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerMaxRangePipel,
        units=model.model_units["distance"],
        doc="Maximum producer pipeline range [distance]",
    )

    ProducerTransportCapacityPipel = dict(
        zip(
            model.df_producer.Index,
            model.df_producer["Pipeline Capacity (bpd)"]
        )
    )
    model.p_ProducerTransportCapacityPipel = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerTransportCapacityPipel,
        units=model.model_units["volume_time"],
        doc="Producer water pipeline capacity [volume_time]",
    )

    ProducerTransportBidPipel = dict(
        zip(
            model.df_producer.Index,
            model.df_producer["Pipe Transport Bid (USD/bbl)"]
        )
    )
    model.p_ProducerTransportBidPipel = Param(
        model.s_PI,
        within=Any,  # to suppress Pyomo warning
        initialize=ProducerTransportBidPipel,
        units=model.model_units["currency_volume"],
        doc="Producer water pipeline bid [currency_volume]",
    )

    # Consumer parameters
    Consumer_Wellpad = dict(zip(model.df_consumer.Index, model.df_consumer.Wellpad))
    model.p_ConsumerPad = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Wellpad,
        doc="Map consumer wellpad to the entry index",
    )

    Consumer_WellpadUnique = dict(zip(model.df_consumer.Index, model.df_consumer.WellpadUnique))
    model.p_ConsumerPadUnique = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_WellpadUnique,
        doc="Map consumer wellpad to the entry index using unique keys",
    )

    Consumer_Operator = dict(zip(model.df_consumer.Index, model.df_consumer.Operator))
    model.p_ConsumerOperator = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Operator,
        doc="Map consumer operator to the entry index",
    )

    Consumer_Start = dict(
        zip(
            model.df_consumer.Index,
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
            model.df_consumer.Index,
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
        zip(model.df_consumer.Index, model.df_consumer["Demand Rate (bpd)"])
    )
    model.p_ConsumerRate = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Rate,
        units=model.model_units["volume_time"],
        doc="Consumer water demand forecast [volume/time]",
    )

    Consumer_Demand_Bid = dict(
        zip(model.df_consumer.Index, model.df_consumer["Consumer Bid (USD/bbl)"])
    )
    model.p_ConsumerDemandBid = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=Consumer_Demand_Bid,
        units=model.model_units["currency_volume"],
        doc="Consumer water demand bid [currency_volume]",
    )

    ConsumerMaxRangeTruck = dict(
        zip(
            model.df_consumer.Index,
            model.df_consumer["Truck Max Dist (mi)"]
        )
    )
    model.p_ConsumerMaxRangeTruck = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerMaxRangeTruck,
        units=model.model_units["distance"],
        doc="Maximum consumer trucking range [distance]",
    )

    ConsumerTransportCapacityTruck = dict(
        zip(
            model.df_consumer.Index,
            model.df_consumer["Trucking Capacity (bpd)"]
        )
    )
    model.p_ConsumerTransportCapacityTruck = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerTransportCapacityTruck,
        units=model.model_units["volume_time"],
        doc="Consumer water trucking capacity [volume_time]",
    )

    ConsumerTransportBidTruck = dict(
        zip(
            model.df_consumer.Index,
            model.df_consumer["Truck Transport Bid (USD/bbl)"]
        )
    )
    model.p_ConsumerTransportBidTruck = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerTransportBidTruck,
        units=model.model_units["currency_volume"],
        doc="Consumer water trucking bid [currency_volume]",
    )

    ConsumerMaxRangePipel = dict(
        zip(
            model.df_consumer.Index,
            model.df_consumer["Pipe Max Dist (mi)"]
        )
    )
    model.p_ConsumerMaxRangePipel = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerMaxRangePipel,
        units=model.model_units["distance"],
        doc="Maximum consumer pipeline range [distance]",
    )

    ConsumerTransportCapacityPipel = dict(
        zip(
            model.df_consumer.Index,
            model.df_consumer["Pipeline Capacity (bpd)"]
        )
    )
    model.p_ConsumerTransportCapacityPipel = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerTransportCapacityPipel,
        units=model.model_units["volume_time"],
        doc="Consumer water pipeline capacity [volume_time]",
    )

    ConsumerTransportBidPipel = dict(
        zip(
            model.df_consumer.Index,
            model.df_consumer["Pipe Transport Bid (USD/bbl)"]
        )
    )
    model.p_ConsumerTransportBidPipel = Param(
        model.s_CI,
        within=Any,  # to suppress Pyomo warning
        initialize=ConsumerTransportBidPipel,
        units=model.model_units["currency_volume"],
        doc="Consumer water pipeline bid [currency_volume]",
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
        within=NonNegativeReals,  # to suppress Pyomo warning
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
        within=NonNegativeReals,  # to suppress Pyomo warning
        initialize=Arc_Time,
        units=model.model_units["time"],
        doc="arc trucking time [time]",
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

    # Set LP(pi,p,c,t) of arcs owned & operated by producers PI
    L_LP_truck = [] # Create list of elements to initialize set s_LP_truck, comprises indices (pi,pp,cp,t)
    L_LP_pipel = [] # Create list of elements to initialize set s_LP_pipel, comprises indices (pi,pp,cp,t)
    # Set LC(ci,p,c,t) of arcs owned & operated by consumers CI
    L_LC_truck = [] # Create list of elements to initialize set s_LC_truck, comprises indices (ci,pp,cp,t)
    L_LC_pipel = [] # Create list of elements to initialize set s_LC_pipel, comprises indices (ci,pp,cp,t)
    for pi in list(model.s_PI.ordered_data()):
        for ci in list(model.s_CI.ordered_data()):
            # LP_truck
            if (
                model.p_ProducerPadUnique[pi] != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_ArcDistance[
                    Producer_Wellpad[pi], Consumer_Wellpad[ci]
                ].value
                <= model.p_ProducerMaxRangeTruck[pi].value
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc: # i.e., if any times overlap
                            L_LP_truck.append(
                                (pi, Producer_Wellpad[pi], Consumer_Wellpad[ci], tp)
                            )
            # LP_pipel
            if (
                model.p_ProducerPadUnique[pi] != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_ArcDistance[
                    Producer_Wellpad[pi], Consumer_Wellpad[ci]
                ].value
                <= model.p_ProducerMaxRangePipel[pi].value
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc: # i.e., if any times overlap
                            L_LP_pipel.append(
                                (pi, Producer_Wellpad[pi], Consumer_Wellpad[ci], tp)
                            )
            # LC_truck
            if (
                model.p_ProducerPadUnique[pi] != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_ArcDistance[
                    Producer_Wellpad[pi], Consumer_Wellpad[ci]
                ].value
                <= model.p_ConsumerMaxRangeTruck[ci].value
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc: # i.e., if any times overlap
                            L_LC_truck.append(
                                (ci, Producer_Wellpad[pi], Consumer_Wellpad[ci], tc)
                            )
            # LC_pipel
            if (
                model.p_ProducerPadUnique[pi] != model.p_ConsumerPadUnique[ci] # possible edge case
                and model.p_ArcDistance[
                    Producer_Wellpad[pi], Consumer_Wellpad[ci]
                ].value
                <= model.p_ConsumerMaxRangePipel[ci].value
            ):
                for tp in model.s_T_pi[pi]:
                    for tc in model.s_T_ci[ci]:
                        if tp == tc: # i.e., if any times overlap
                            L_LC_pipel.append(
                                (ci, Producer_Wellpad[pi], Consumer_Wellpad[ci], tc)
                            )
    L_LP_truck = list(set(L_LP_truck)) # remove duplciates
    L_LP_pipel = list(set(L_LP_pipel)) # remove duplciates
    L_LC_truck = list(set(L_LC_truck)) # remove duplciates
    L_LC_pipel = list(set(L_LC_pipel)) # remove duplciates
    model.s_LP_truck = Set(dimen=4, initialize=L_LP_truck, doc="Valid Producer Trucking Arcs")
    model.s_LP_pipel = Set(dimen=4, initialize=L_LP_pipel, doc="Valid Producer Pipeline Arcs")
    model.s_LC_truck = Set(dimen=4, initialize=L_LC_truck, doc="Valid Consumer Trucking Arcs")
    model.s_LC_pipel = Set(dimen=4, initialize=L_LC_pipel, doc="Valid Consumer Pipeline Arcs")
    print("Primary arc sets constructed")
    
    # Sets for arcs inbound on (cp,t)
    def s_LP_truck_in_ct_INIT(model, cp, tc):
        elems = []
        for (pi, p, c, t) in model.s_LP_truck.ordered_data():
            if c == cp and t == tc:
                elems.append((pi, p, c, t))
        return elems
    model.s_LP_truck_in_ct = Set(
        model.s_CP,
        model.s_T,
        dimen=4,
        initialize=s_LP_truck_in_ct_INIT,
        doc="Inbound producer trucking arcs on completions pad cp at time t",
    )

    def s_LP_pipel_in_ct_INIT(model, cp, tc):
        elems = []
        for (pi, p, c, t) in model.s_LP_pipel.ordered_data():
            if c == cp and t == tc:
                elems.append((pi, p, c, t))
        return elems
    model.s_LP_pipel_in_ct = Set(
        model.s_CP,
        model.s_T,
        dimen=4,
        initialize=s_LP_pipel_in_ct_INIT,
        doc="Inbound producer pipeline arcs on completions pad cp at time t",
    )

    def s_LC_truck_in_ct_INIT(model, cp, tc):
        elems = []
        for (ci, p, c, t) in model.s_LC_truck.ordered_data():
            if c == cp and t == tc:
                elems.append((ci, p, c, t))
        return elems
    model.s_LC_truck_in_ct = Set(
        model.s_CP,
        model.s_T,
        dimen=4,
        initialize=s_LC_truck_in_ct_INIT,
        doc="Inbound consumer trucking arcs on completions pad cp at time t",
    )

    def s_LC_pipel_in_ct_INIT(model, cp, tc):
        elems = []
        for (ci, p, c, t) in model.s_LC_pipel.ordered_data():
            if c == cp and t == tc:
                elems.append((ci, p, c, t))
        return elems
    model.s_LC_pipel_in_ct = Set(
        model.s_CP,
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
            if p == pp and t == tp:
                elems.append((pi, p, c, t))
        return elems
    model.s_LP_truck_out_pt = Set(
        model.s_PP,
        model.s_T,
        dimen=4,
        initialize=s_LP_truck_out_pt_INIT,
        doc="Outbound producer trucking arcs from production pad pp at time t",
    )

    def s_LP_pipel_out_pt_INIT(model, pp, tp):
        elems = []
        for (pi, p, c, t) in model.s_LP_pipel.ordered_data():
            if p == pp and t == tp:
                elems.append((pi, p, c, t))
        return elems
    model.s_LP_pipel_out_pt = Set(
        model.s_PP,
        model.s_T,
        dimen=4,
        initialize=s_LP_pipel_out_pt_INIT,
        doc="Outbound producer pipeline arcs from production pad pp at time t",
    )

    def s_LC_truck_out_pt_INIT(model, pp, tp):
        elems = []
        for (ci, p, c, t) in model.s_LC_truck.ordered_data():
            if p == pp and t == tp:
                elems.append((ci, p, c, t))
        return elems
    model.s_LC_truck_out_pt = Set(
        model.s_PP,
        model.s_T,
        dimen=4,
        initialize=s_LC_truck_out_pt_INIT,
        doc="Outbound consumer trucking arcs from production pad pp at time t",
    )

    def s_LC_pipel_out_pt_INIT(model, pp, tp):
        elems = []
        for (ci, p, c, t) in model.s_LC_pipel.ordered_data():
            if p == pp and t == tp:
                elems.append((ci, p, c, t))
        return elems
    model.s_LC_pipel_out_pt = Set(
        model.s_PP,
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
            if Producer_Wellpad[pi] == p:
                pies.append(pi)
        return pies
    model.ProducerNodeMap = Set(
        model.s_PP,
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
        model.s_PP,
        model.s_T,
        dimen=1,
        initialize=ProducerNodeTimeMapINIT,
        doc="Mapping from producer node p in s_PP active at time t in s_T to pi in s_PI (one-to-many)",
    )

    # set for consumer-node mapping
    def ConsumerNodeMapINIT(model, c):
        cies = []
        for ci in model.s_CI:
            if Consumer_Wellpad[ci] == c:
                cies.append(ci)
        return cies

    model.ConsumerNodeMap = Set(
        model.s_CP,
        dimen=1,
        initialize=ConsumerNodeMapINIT,
        doc="Mapping from consumer node c in s_CP to ci in s_CI (one-to-many)",
    )

    # set for consumer-node-time mapping
    def ConsumerNodeTimeMapINIT(model, c, t):
        ceties = []
        for ci in model.s_CI:
            if Consumer_Wellpad[ci] == c and t in model.s_T_ci[ci]:
                ceties.append(ci)
        return ceties

    model.ConsumerNodeTimeMap = Set(
        model.s_CP,
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
    def p_FP_truck_UB_init(model, pi, p, c, t):
        if tp in model.s_T_pi[pi]:
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
    def p_FP_pipel_UB_init(model, pi, p, c, t):
        if tp in model.s_T_pi[pi]:
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
    def p_FC_truck_UB_init(model, ci, p, c, t):
        if tc in model.s_T_ci[ci]:
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
    def p_FC_pipel_UB_init(model, ci, p, c, t):
        if tc in model.s_T_ci[ci]:
            return model.p_ConsumerTransportCapacityPipel[pi]
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
                expr += sum(model.v_FP_truck[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_truck_in_ct[c, t])

            if bool(model.s_LP_pipel_in_ct[c, t]):
                expr += sum(model.v_FP_pipel[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_pipel_in_ct[c, t])

            if bool(model.s_LC_truck_in_ct[c, t]):
                expr += sum(model.v_FC_truck[ci, pa, ca, t] for (ci, pa, ca, t) in model.s_LC_truck_in_ct[c, t])

            if bool(model.s_LC_pipel_in_ct[c, t]):
                expr += sum(model.v_FC_pipel[ci, pa, ca, t] for (ci, pa, ca, t) in model.s_LC_pipel_in_ct[c, t])

            return expr == 0
        else:
            return Constraint.Skip

    model.ConsumerDemandBalance = Constraint(
        model.s_CP,
        model.s_T,
        rule=ConsumerDemandBalanceRule,
        doc="Consumer demand balance",
    )

    # supply balance
    def ProducerSupplyBalanceRule(model, p, t):
        if bool(model.ProducerNodeTimeMap[p, t]):
            expr = sum(model.v_Supply[pi, t] for pi in model.ProducerNodeTimeMap[p, t])

            if bool(model.s_LP_truck_out_pt[p, t]):
                expr += -sum(model.v_FP_truck[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_truck_out_pt[p, t])

            if bool(model.s_LP_pipel_out_pt[p, t]):
                expr += -sum(model.v_FP_pipel[pi, pa, ca, t] for (pi, pa, ca, t) in model.s_LP_pipel_out_pt[p, t])

            if bool(model.s_LC_truck_out_pt[p, t]):
                expr += -sum(model.v_FC_truck[ci, pa, ca, t] for (ci, pa, ca, t) in model.s_LC_truck_out_pt[p, t])

            if bool(model.s_LC_pipel_out_pt[p, t]):
                expr += -sum(model.v_FC_pipel[ci, pa, ca, t] for (ci, pa, ca, t) in model.s_LC_pipel_out_pt[p, t])

            return expr == 0
        else:
            return Constraint.Skip

    model.ProducerSupplyBalance = Constraint(
        model.s_PP,
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
        model.s_PP,
        model.s_CP,
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
        model.s_PP,
        model.s_CP,
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
        model.s_PP,
        model.s_CP,
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
        model.s_PP,
        model.s_CP,
        model.s_T,
        rule=ConsumerPipingMaxINIT,
        doc="Consumer piping maximum",
    )

    # -------------------- Define objectives ----------------------

    # Objective
    ClearingObjExpr = (
        sum(
            model.p_ConsumerDemandBid[ci] * model.v_Demand[ci, t]
            for ci in model.s_CI
            for t in model.s_T_ci[ci]
        )
        - sum(
            model.p_ProducerSupplyBid[pi] * model.v_Supply[pi, t]
            for pi in model.s_PI
            for t in model.s_T_pi[pi]
        )
        - sum(
            model.p_ProducerTransportBidTruck[pi] * model.v_FP_truck[pi, p, c, t]
            for (pi, p, c, t) in model.s_LP_truck
        )
        - sum(
            model.p_ProducerTransportBidPipel[pi] * model.v_FP_pipel[pi, p, c, t]
            for (pi, p, c, t) in model.s_LP_pipel
        )
        - sum(
            model.p_ConsumerTransportBidTruck[ci] * model.v_FC_truck[ci, p, c, t]
            for (ci, p, c, t) in model.s_LC_truck
        )
        - sum(
            model.p_ConsumerTransportBidPipel[ci] * model.v_FC_pipel[ci, p, c, t]
            for (ci, p, c, t) in model.s_LC_pipel
        )
    )

    model.objective = Objective(
        expr=ClearingObjExpr, sense=maximize, doc="Objective function"
    )

    print("Model setup complete")
    return model


# Create output dataframe
def jsonize_outputs(model, matches_dir):
    df_vP_Truck = pd.DataFrame(
        {
            "Carrier Index": key[0],
            "From wellpad": model.p_ProducerPad[key[0]],
            "To wellpad": key[2],
            "Date Index": key[3],
            "Date": model.d_T[key[3]],
            "Rate": value(model.v_FP_truck[key]),
        }
        for key in model.v_FP_truck
        if value(model.v_FP_truck[key]) > 0.01
    )
    df_vP_Truck["Distance"] = add_dataframe_distance(df_vP_Truck, model.df_distance)
    #'To index': key[1],
    #'From operator': model.p_ProducerOperator[key[0]],
    #'To operator': model.p_ConsumerOperator[key[2]],
    # Timestamps (i.e., Pandas datetime format) are not JSON serializable; convert to string
    if not df_vP_Truck.empty:
        df_vP_Truck["Price"] = [
            model.p_ConsumerNodalPrice[row["To wellpad"], row["Date Index"]].value
            - model.p_ProducerNodalPrice[row["From wellpad"], row["Date Index"]].value
            for ind, row in df_vP_Truck.iterrows()
        ]
        df_vP_Truck["Date"] = df_vP_Truck["Date"].dt.strftime("%Y/%m/%d")
        df_vP_Truck["From Longitude"] = [
            model.p_ProducerWellpadLon[pad]
            for pad in df_vP_Truck.loc[:, "From wellpad"]
        ]
        df_vP_Truck["From Latitude"] = [
            model.p_ProducerWellpadLat[pad]
            for pad in df_vP_Truck.loc[:, "From wellpad"]
        ]
        df_vP_Truck["To Longitude"] = [
            model.p_ConsumerWellpadLon[pad] for pad in df_vP_Truck.loc[:, "To wellpad"]
        ]
        df_vP_Truck["To Latitude"] = [
            model.p_ConsumerWellpadLat[pad] for pad in df_vP_Truck.loc[:, "To wellpad"]
        ]
        df_vP_Truck["Operator"] = [
            model.p_ProducerOperator[index] for index in df_vP_Truck["Carrier Index"]
        ]

    df_vM_Truck = pd.DataFrame(
        {
            "Midstream Index": key[0],
            "From wellpad": key[1],
            "To wellpad": key[2],
            "Departure Index": key[3],
            "Arrival Index": key[4],
            "Departure Date": model.d_T[key[3]],
            "Arrival Date": model.d_T[key[4]],
            "Rate": value(model.v_FM_Trucked[key]),
        }
        for key in model.v_FM_Trucked
        if value(model.v_FM_Trucked[key]) > 0.01
    )
    df_vM_Truck["Distance"] = add_dataframe_distance(df_vM_Truck, model.df_distance)
    if not df_vM_Truck.empty:
        df_vM_Truck["Price"] = [
            model.p_ConsumerNodalPrice[row["To wellpad"], row["Arrival Index"]].value
            - model.p_ProducerNodalPrice[
                row["From wellpad"], row["Departure Index"]
            ].value
            for ind, row in df_vM_Truck.iterrows()
        ]
        df_vM_Truck["Departure Date"] = df_vM_Truck["Departure Date"].dt.strftime(
            "%Y/%m/%d"
        )
        df_vM_Truck["Arrival Date"] = df_vM_Truck["Arrival Date"].dt.strftime(
            "%Y/%m/%d"
        )
        df_vM_Truck["From Longitude"] = [
            model.p_ProducerWellpadLon[pad]
            for pad in df_vM_Truck.loc[:, "From wellpad"]
        ]
        df_vM_Truck["From Latitude"] = [
            model.p_ProducerWellpadLat[pad]
            for pad in df_vM_Truck.loc[:, "From wellpad"]
        ]
        df_vM_Truck["To Longitude"] = [
            model.p_ConsumerWellpadLon[pad] for pad in df_vM_Truck.loc[:, "To wellpad"]
        ]
        df_vM_Truck["To Latitude"] = [
            model.p_ConsumerWellpadLat[pad] for pad in df_vM_Truck.loc[:, "To wellpad"]
        ]
        df_vM_Truck["Operator"] = [
            model.p_MidstreamOperator[index] for index in df_vM_Truck["Midstream Index"]
        ]

    df_v_Supply = pd.DataFrame(
        {
            "Supplier Index": key[0],
            "Supplier Wellpad": model.p_ProducerPad[key[0]],
            "Date Index": key[1],
            "Date": model.d_T[key[1]],
            "Rate": value(model.v_Supply[key]),
        }
        for key in model.v_Supply
        if value(model.v_Supply[key]) > 0
    )
    if not df_v_Supply.empty:
        df_v_Supply["Date"] = df_v_Supply["Date"].dt.strftime("%Y/%m/%d")
        df_v_Supply["Longitude"] = [
            model.p_ProducerWellpadLon[pad]
            for pad in df_v_Supply.loc[:, "Supplier Wellpad"]
        ]
        df_v_Supply["Latitude"] = [
            model.p_ProducerWellpadLat[pad]
            for pad in df_v_Supply.loc[:, "Supplier Wellpad"]
        ]
        df_v_Supply["Nodal Price"] = add_dataframe_prices(
            df_v_Supply, model.p_ProducerNodalPrice, "Supplier"
        )
        df_v_Supply["Operator"] = [
            model.p_ProducerOperator[index] for index in df_v_Supply["Supplier Index"]
        ]

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
    if not df_v_Demand.empty:
        df_v_Demand["Date"] = df_v_Demand["Date"].dt.strftime("%Y/%m/%d")
        df_v_Demand["Longitude"] = [
            model.p_ConsumerWellpadLon[pad]
            for pad in df_v_Demand.loc[:, "Consumer Wellpad"]
        ]
        df_v_Demand["Latitude"] = [
            model.p_ConsumerWellpadLat[pad]
            for pad in df_v_Demand.loc[:, "Consumer Wellpad"]
        ]
        df_v_Demand["Nodal Price"] = add_dataframe_prices(
            df_v_Demand, model.p_ConsumerNodalPrice, "Consumer"
        )
        df_v_Demand["Operator"] = [
            model.p_ConsumerOperator[index] for index in df_v_Demand["Consumer Index"]
        ]

    if df_v_Supply.empty and df_v_Demand.empty:
        print("***************************************")
        print("***************************************")
        print("*** No transactions; market is dry! ***")
        print("***************************************")
        print("***************************************")
        return None

    # convert dataframes to dictionaries for easy json output
    d_vP_Truck = df_vP_Truck.to_dict(orient="records")
    d_vM_Truck = df_vM_Truck.to_dict(orient="records")
    d_v_Supply = df_v_Supply.to_dict(orient="records")
    d_v_Demand = df_v_Demand.to_dict(orient="records")
    with open(matches_dir, "w") as data_file:
        # json.dump([d_v_Supply, d_v_Demand, d_v_Truck], data_file, indent=2)
        json.dump(
            {
                "Supply": d_v_Supply,
                "Demand": d_v_Demand,
                "Transport (Producer)": d_vP_Truck,
                "Transport (Midstream)": d_vM_Truck,
            },
            data_file,
            indent=2,
        )
    return None


# Create secondary dataframe for outputting profit
def jsonize_profits(model, profits_dir):
    # Create dataframes from paramter data
    df_ProducerProfit = pd.DataFrame(
        {
            "Producer Index": key,
            "Profit": model.p_ProducerProfit[key].value,
            "Operator": model.p_ProducerOperator[key],
        }
        for key in model.p_ProducerProfit
    )
    df_ConsumerProfit = pd.DataFrame(
        {
            "Consumer Index": key,
            "Profit": model.p_ConsumerProfit[key].value,
            "Operator": model.p_ConsumerOperator[key],
        }
        for key in model.p_ConsumerProfit
    )
    df_MidstreamProfit = pd.DataFrame(
        {
            "Midstream Index": key,
            "Profit": model.p_MidstreamTotalProfit[key].value,
            "Operator": model.p_MidstreamOperator[key],
        }
        for key in model.p_MidstreamTotalProfit
    )
    df_ProducerTransportProfit = pd.DataFrame(
        {
            "Producer Index": key,
            "Profit": model.p_ProducerTotalTransportProfit[key].value,
            "Operator": model.p_ProducerOperator[key],
        }
        for key in model.p_ProducerTotalTransportProfit
    )

    # convert to dictionaries for easy json output
    d_ProducerProfit = df_ProducerProfit.to_dict(orient="records")
    d_ConsumerProfit = df_ConsumerProfit.to_dict(orient="records")
    d_MidstreamProfit = df_MidstreamProfit.to_dict(orient="records")
    d_ProducerTransportProfit = df_ProducerTransportProfit.to_dict(orient="records")
    with open(profits_dir, "w") as data_file:
        json.dump(
            {
                "Supply": d_ProducerProfit,
                "Demand": d_ConsumerProfit,
                "Transport (Producer)": d_ProducerTransportProfit,
                "Transport (Midstream)": d_MidstreamProfit,
            },
            data_file,
            indent=2,
        )
    return None


# Post-solve calculations
def PostSolve(model):
    # Producer Nodal Price
    # ProducerNodalPriceINIT = dict.fromkeys([(p,t) for p in model.s_PP for t in model.s_T], None)
    # for index in model.ProducerSupplyBalance:
    #    ProducerNodalPriceINIT[index] = model.dual[model.ProducerSupplyBalance[index]]
    def ProducerNodalPriceINIT(model, p, t):
        try:
            # NOTE: Dual variables might be negative; may need to multiply by -1 to get meaningful results
            return -1 * model.dual[model.ProducerSupplyBalance[p, t]]
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
            return -1 * value(model.dual[model.ConsumerDemandBalance[c, t]])
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
            return value(
                -1 * model.dual[model.ConsumerDemandBalance[c, tc]]
                - (-1) * model.dual[model.ProducerSupplyBalance[p, tp]]
            )
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
        p = model.p_ProducerPadUnique[pi]
        try:
            return sum(
                (
                    model.p_ProducerNodalPrice[p, t].value
                    - model.p_ProducerSupplyBid[pi].value
                )
                * model.v_Supply[pi, t].value
                for t in model.s_T_pi[pi]
            )
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
        c = model.p_ConsumerPadUnique[ci]
        try:
            return sum(
                (
                    model.p_ConsumerDemandBid[ci].value
                    - model.p_ConsumerNodalPrice[c, t].value
                )
                * model.v_Demand[ci, t].value
                for t in model.s_T_ci[ci]
            )
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
                return (
                    model.p_TransportPrice[p, c, tp, tc].value
                    - model.p_MidstreamBid[mi].value
                ) * model.v_FM_Trucked[mi, p, c, tp, tc].value
                # return (model.p_TransportPrice[p,c,tp,tc].value - model.p_MidstreamBid[mi].value*model.p_ArcDistance[p,c].value)*model.v_FM_Trucked[mi,p,c,tp,tc].value
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
        return sum(
            model.p_MidstreamRouteProfit[mii, p, c, tp, tc]
            for (mii, p, c, tp, tc) in model.s_LLM
            if mii == mi
        )

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
                return (
                    model.p_TransportPrice[p, c, t, t].value
                    - model.p_ProducerTransportBidTruck[pi].value
                ) * model.v_FP_truck[pi, p, c, t].value
                # return (model.p_TransportPrice[p,c,t,t].value - model.p_ProducerTransportBidTruck[pi].value*model.p_ArcDistance[p,c].value)*model.v_FP_truck[pi,p,c,t].value
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
        return sum(
            model.p_ProducerRouteProfit[pii, p, c, t]
            for (pii, p, c, t) in model.s_LP_truck
            if pii == pi
        )

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

    # Midstream transport arcs
    fo.write("\n" + PrintSpacer)
    fo.write("\nMidstream Transportation Arcs")
    for (mi, p, c, tp, tc) in model.s_LLM:
        fo.write("\n" + ",".join((mi, p, c, tp, tc)))

    # Consumer set
    fo.write("\n" + PrintSpacer)
    fo.write("\nConsumer index mapping set")
    for c in model.s_CP:
        for t in model.s_T:
            fo.write(
                "\n" + c + "," + t + ": " + ",".join(model.ConsumerNodeTimeMap[c, t])
            )

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
    filename = os.path.join(save_dir, "_postsolve.txt")
    PrintSpacer = "#" * 50
    fo = open(filename, "w")

    # Objective value
    fo.write("\n" + PrintSpacer)
    fo.write("\nObjective")
    fo.write("\n" + str(value(model.objective)))

    # Producer Profits
    fo.write("\n" + PrintSpacer)
    fo.write("\nProducer Profits [currency]")
    for pi in model.s_PI:
        fo.write("\n" + pi + ": " + str(model.p_ProducerProfit[pi].value))

    # Consumer Profits
    fo.write("\n" + PrintSpacer)
    fo.write("\nConsumer Profits [currency]")
    for ci in model.s_CI:
        fo.write("\n" + ci + ": " + str(model.p_ConsumerProfit[ci].value))

    # Midstream per-route Profits
    fo.write("\n" + PrintSpacer)
    fo.write("\nMidstream Profits [currency]")
    for (mi, p, c, tp, tc) in model.s_LLM:
        fo.write(
            "\n"
            + ",".join((mi, p, c, tp, tc))
            + ": "
            + str(model.p_MidstreamRouteProfit[mi, p, c, tp, tc].value)
        )

    # Producer per-route Profits
    fo.write("\n" + PrintSpacer)
    fo.write("\nProducer Transportation Profits (per-route) [currency]")
    for (pi, p, c, t) in model.s_LP_truck:
        fo.write(
            "\n"
            + ",".join((pi, p, c, t))
            + ": "
            + str(model.p_ProducerRouteProfit[pi, p, c, t].value)
        )

    # Producer variable value
    fo.write("\n" + PrintSpacer)
    fo.write("\nProducer Supply Allocations [volume]")
    for pi in model.s_PI:
        for t in model.s_T_pi[pi]:
            fo.write(
                "\n"
                + pi
                + ","
                + model.p_ProducerPad[pi]
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
                + model.p_ConsumerPad[ci]
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

    # Producer nodal prices
    fo.write("\n" + PrintSpacer)
    fo.write("\nProducer Nodal Prices [currency_volume]")
    for (p, t) in model.s_PP * model.s_T:
        fo.write(
            "\n" + ",".join((p, t)) + ": " + str(model.p_ProducerNodalPrice[p, t].value)
        )

    # Consumer nodal prices
    fo.write("\n" + PrintSpacer)
    fo.write("\nConsumer Nodal Prices [currency_volume]")
    for (c, t) in model.s_CP * model.s_T:
        fo.write(
            "\n" + ",".join((c, t)) + ": " + str(model.p_ConsumerNodalPrice[c, t].value)
        )

    # Transportation prices
    fo.write("\n" + PrintSpacer)
    fo.write("\nTransportation Prices [currency_volume]")
    for (p, c, tp, tc) in model.s_PP * model.s_CP * model.s_T * model.s_T:
        fo.write(
            "\n"
            + ",".join((p, c, tp, tc))
            + ": "
            + str(model.p_TransportPrice[p, c, tp, tc].value)
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
