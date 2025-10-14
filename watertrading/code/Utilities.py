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

"""
A collection of utility functions that may be called from multiple modules, or which don't have
a more appropriate location.
"""
import os
import pandas as pd
import json

##################################################
def df_to_dict_helper(df):
    """
    A small function to help convert dataframes to dictionaries suitable to Pyomo's requirements
    Inputs: df - a dataframe
    Outputs: ind_dict - a dictionary
    """
    # Pull the lists of row and column labels from the dictionary
    row_labels = df.index
    col_labels = df.columns

    # Initiate dictionary
    ind_dict = {}

    # Iterate over the row and column labels
    for row_ind, row_label in enumerate(row_labels):
        for col_ind, col_label in enumerate(col_labels):
            value = df.iloc[row_ind, col_ind]
            ind_dict[row_label, col_label] = value
    return ind_dict


##################################################
def csv_2_json(input_dir, PRODUCER_CSV, CONSUMER_CSV, name="jsonized_data.json"):
    """
    A function that converts existing case study .csv data files into the newer JSON format
    Inputs:
        - producer csv data file
        - consumer csv data file
        - (optional) name: a name for the output file
    Outputs:
        - requests.JSON file
    """
    # Declare Pandas dataframes for producers and consumers
    df_producer = pd.DataFrame()
    df_consumer = pd.DataFrame()

    # Read in producer data; convert date data to datetime to enforce proper format, and then to string for JSON export (datetime is not compatible)
    df_producer = pd.read_csv(os.path.join(input_dir, PRODUCER_CSV))
    #df_producer["Start Date"] = pd.to_datetime(
    #    df_producer["Start Date"], format="%Y/%m/%d"
    #)
    #df_producer["End Date"] = pd.to_datetime(
    #    df_producer["End Date"], format="%Y/%m/%d"
    #)
    df_producer = excel_date_parser(df_producer)
    df_producer["Start Date"] = df_producer["Start Date"].dt.strftime("%Y/%m/%d")
    df_producer["End Date"] = df_producer["End Date"].dt.strftime("%Y/%m/%d")

    # Read in consumer data; convert date data to datetime to enforce proper format, and then to string for JSON export (datetime is not compatible)
    df_consumer = pd.read_csv(os.path.join(input_dir, CONSUMER_CSV))
    #df_consumer["Start Date"] = pd.to_datetime(
    #    df_consumer["Start Date"], format="%Y/%m/%d"
    #)
    #df_consumer["End Date"] = pd.to_datetime(
    #    df_consumer["End Date"], format="%Y/%m/%d"
    #)
    df_consumer = excel_date_parser(df_consumer)
    df_consumer["Start Date"] = df_consumer["Start Date"].dt.strftime("%Y/%m/%d")
    df_consumer["End Date"] = df_consumer["End Date"].dt.strftime("%Y/%m/%d")

    # Remove NaN entries due to missing quality (in case missing, that is) and replace with ""
    df_producer = df_producer.fillna("")
    df_consumer = df_consumer.fillna("")

    # Convert dataframes to dictionaries for export (we will use json.dump, not the pandas built-in function)
    d_producer = df_producer.to_dict(orient="records")
    d_consumer = df_consumer.to_dict(orient="records")
    d_restriction = (
        dict()
    )  # this is initialized so that it exists in the output; it will be managed by WordPress

    # open JSON file and dump data under named headers - this is the agreed upon format
    with open(os.path.join(input_dir, name), "w") as data_file:
        json.dump(
            {
                "Producers": d_producer,
                "Consumers": d_consumer,
                "Restrictions": d_restriction,
            },
            data_file,
            indent=2,
        )
    return None


##################################################
def excel_date_parser(df):
    """
    Excel dates are commonly saved to "%m/%d/%Y" even if the user chooses
    another format like "%Y-%m-%d". This function checks for the Excel format,
    then the expected format, and then converts the dates to correct input
    format for the jsonized data. Likely only called by csv_2_json().
    """
    # df_consumer["Start Date"] = pd.to_datetime(df_consumer["Start Date"], format="%Y/%m/%d")

    # Expected date formats
    formats = ["%m/%d/%Y", "%Y-%m-%d", "%Y/%m/%d"]

    # try each format; continue if one works
    for f in formats:
        try:
            df["Start Date"] = pd.to_datetime(df["Start Date"], format=f)
            df["End Date"] = pd.to_datetime(df["End Date"], format=f)
            break # so we don't test date formats in one works
        except ValueError:
            continue
    return df


##################################################
def fetch_case_data(requests_JSON):
    """
    NOTE: This is a duplicate of get_data(); a "light" version for fetching data needed for plots
    Inputs:
    > requests_JSON - directory to requests JSON file
    Outputs:
    > df_producer - producer request dataframe
    > df_consumer - consumer request dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
        print("Data directory not found")

    # Pull in JSON request data
    with open(requests_JSON, "r") as read_file:
        request_data = json.load(read_file)

    # Producer inputs
    df_producer = pd.DataFrame(data=request_data["Producers"])
    # convert time inputs to datetime
    df_producer["Start Date"] = pd.to_datetime(
        df_producer["Start Date"], format="%Y/%m/%d"
    )
    df_producer["End Date"] = pd.to_datetime(df_producer["End Date"], format="%Y/%m/%d")

    # Consumer inputs
    df_consumer = pd.DataFrame(data=request_data["Consumers"])
    # convert time inputs to datetime
    df_consumer["Start Date"] = pd.to_datetime(
        df_consumer["Start Date"], format="%Y/%m/%d"
    )
    df_consumer["End Date"] = pd.to_datetime(df_consumer["End Date"], format="%Y/%m/%d")

    # Midstream inputs
    df_midstream = pd.DataFrame(data=request_data["Midstreams"])
    df_midstream["Start Date"] = pd.to_datetime(
        df_midstream["Start Date"], format="%Y/%m/%d"
    )
    df_midstream["End Date"] = pd.to_datetime(
        df_midstream["End Date"], format="%Y/%m/%d"
    )

    # Matching restrictions
    df_restrictions = pd.DataFrame(data=request_data["Restrictions"])
    return df_producer, df_consumer, df_midstream, df_restrictions


##################################################
def fetch_distance_data(distance_JSON):
    """
    Inputs:
    > distance_JSON - directory to distance JSON file
    Outputs:
    > df_distance - distance dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
        print("Data directory not found")

    # Pull in JSON request data
    with open(distance_JSON, "r") as read_file:
        distance_data = json.load(read_file)

    # Producer inputs
    df_distance = pd.DataFrame(data=distance_data["DriveDistances"])

    return df_distance


##################################################
def fetch_match_data(matches_dir):
    """
    NOTE: This is a duplicate of get_data(); a "light" version for fetching data needed for plots
    Inputs:
    > matches_dir - directory to matches JSON file
    Outputs:
    > df_matches - matches dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
        print("Matches directory not found")

    # Pull in JSON request data
    with open(matches_dir, "r") as read_file:
        matches_data = json.load(read_file)

    # Producer inputs
    df_matches_supply = pd.DataFrame(data=matches_data["Supply"])
    df_matches_demand = pd.DataFrame(data=matches_data["Demand"])
    df_matches_producer_transport = pd.DataFrame(
        data=matches_data["Transport (Producer)"]
    )
    df_matches_midstream_transport = pd.DataFrame(
        data=matches_data["Transport (Midstream)"]
    )
    # convert time inputs to datetime
    df_matches_supply["Date"] = pd.to_datetime(
        df_matches_supply["Date"], format="%Y/%m/%d"
    )
    df_matches_demand["Date"] = pd.to_datetime(
        df_matches_demand["Date"], format="%Y/%m/%d"
    )
    if not df_matches_producer_transport.empty:
        df_matches_producer_transport["Date"] = pd.to_datetime(
            df_matches_producer_transport["Date"], format="%Y/%m/%d"
        )
    if not df_matches_midstream_transport.empty:
        df_matches_midstream_transport["Departure Date"] = pd.to_datetime(
            df_matches_midstream_transport["Departure Date"], format="%Y/%m/%d"
        )
        df_matches_midstream_transport["Arrival Date"] = pd.to_datetime(
            df_matches_midstream_transport["Arrival Date"], format="%Y/%m/%d"
        )

    return (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    )


##################################################
def fetch_profit_data(profits_dir):
    """
    Inputs:
    > profits_dir - directory to matches JSON file
    Outputs:
    > df_profits - matches dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
        print("Profits files directory not found")

    # Pull in JSON request data
    with open(profits_dir, "r") as read_file:
        profits_data = json.load(read_file)

    # Producer inputs
    df_profits_supply = pd.DataFrame(data=profits_data["Supply"])
    df_profits_demand = pd.DataFrame(data=profits_data["Demand"])
    df_profits_producer_transport = pd.DataFrame(
        data=profits_data["Transport (Producer)"]
    )
    df_profits_midstream_transport = pd.DataFrame(
        data=profits_data["Transport (Midstream)"]
    )

    return (
        df_profits_supply,
        df_profits_demand,
        df_profits_producer_transport,
        df_profits_midstream_transport,
    )


##################################################
def add_dataframe_distance(df, dist_matrix):
    """
    Inputs:
    > df - a dataframe including start and end destinations

    Outputs:
    > dist_matrix - a distance matrix

    FUNCTION DESCRIPTION:
    > Adds distances between start and end destinations in df to df as a new column
    """
    distances = []
    for index, row in df.iterrows():
        dist = dist_matrix.loc[row["From wellpad"], row["To wellpad"]]
        distances.append(dist)
    return distances


##################################################
def add_dataframe_prices(df, price_source, type):
    """
    Inputs:
    > df - a dataframe including locations and times
    > price_source - a parameter containing pricing data
    > type - a string indicating which type of price we are handling; for now, either "Supplier" or "Consumer"

    Outputs:
    > prices - a list of prices ordered to match the keys in df

    FUNCTION DESCRIPTION:
    > Returns a vector of price data aligned to the input dataframe, applying NaN to instances where prices are not defined
    """
    # Line of code inspiring this function: [model.p_ProducerNodalPrice[row["Supplier Wellpad"],row["Date"]] for ind,row in df_v_Supply.iterrows()]
    # Type dictionary allowing function reuse:
    type_dict = {"Supplier": "Supplier Wellpad", "Consumer": "Consumer Wellpad"}
    # Add prices to list in order, use NaN values if price not defined
    prices = []
    for ind, row in df.iterrows():
        try:
            prices.append(price_source[row[type_dict[type]], row["Date Index"]].value)
        except:
            prices.append(float("NaN"))
    return prices


##################################################
def GetUniqueDFData(df, key_col, val_col):
    """
    Returns a dictionary of unique dataframe entries keyed from one colum to another
    Inputs:
    - df: a datframe
    - key_col: a column to be used as keys
    - val_col: a column to be used as values
    Outputs:
    - a dictionary consisting of keys row_keys mapped to val_keys
    """
    uniques = df.drop_duplicates(key_col)
    return dict(zip(uniques.loc[:, key_col], uniques.loc[:, val_col]))


##################################################
def GenerateCombinationDataFrame(df1, df2, name, newname1, newname2):
    """
    Returns a DataFrame consisting of the combinations of the entities in df1 and df2
    NOTE: column DataFrames can be extracted using double-square brackets without conversion to sequence
    Inputs:
    - df1: an m1-by-1 DataFrame (column dataframe)
    - df2: an m2-by-1 DataFrame (column dataframe)
    Outputs:
    -df_cross: an m1xm2-by-2 DataFrame
    """
    # Will use pd.merge; reqiures some trickery; we will add dummy indices to get the dehavior we want
    pd.options.mode.chained_assignment = (
        None  # to stop pandas warning us about the behavior we are using
    )
    df1["temp"] = 1
    df2["temp"] = 1
    # rename columns for output (and to avoid merge)
    df1.rename(columns={name: newname1}, inplace=True)
    df2.rename(columns={name: newname2}, inplace=True)
    # merge: df_cross =
    return df1.merge(df2, on="temp", how="outer").drop(columns="temp")


##################################################
def SupplierBidType(text):
    """
    A simple function that returns (-1) if text=="Willing to pay" and (1) if text=="Want to be paid"
    Inputs:
    - text: the content of df_producer["Bid Type"]
    Outputs:
    - Positive or negative one, depending on text.
    """
    if text == "Willing to pay":
        return -1
    elif text == "Want to be paid":
        return 1
    else:
        raise Exception(
            "Please verify that the contents of df_producer['Bid Type'] are correct."
        )


##################################################
def ConsumerBidType(text):
    """
    A simple function that returns (1) if text=="Willing to pay" and (-1) if text=="Want to be paid"
    Inputs:
    - text: the content of df_consumer["Bid Type"]
    Outputs:
    - Positive or negative one, depending on text.
    """
    if text == "Willing to pay":
        return 1
    elif text == "Want to be paid":
        return -1
    else:
        raise Exception(
            "Please verify that the contents of df_consumer['Bid Type'] are correct."
        )


##################################################
def SupplierBalanceType(value):
    """
    A simple function that returns "I will pay" if the price value is negative and "To be paid to me" if it is positive; the opposite behavior of ConsumerBalanceType
    Inputs:
    - value: a price value
    Outputs:
    - a string, either "I will pay" or "To be paid to me" depending on value
    """
    if value < 0:
        return "I will pay"
    elif value >= 0:
        return "To be paid to me"
    else:
        raise Exception("Please ensure the value entered is a number.")


##################################################
def ConsumerBalanceType(value):
    """
    A simple function that returns "I will pay" if the price value is positive and "To be paid to me" if it is negative; the opposite behavior of SupplierBalanceType
    Inputs:
    - value: a price value
    Outputs:
    - a string, either "I will pay" or "To be paid to me" depending on value
    """
    if value < 0:
        return "I will pay"
    elif value >= 0:
        return "To be paid to me"
    else:
        raise Exception("Please ensure the value entered is a number.")


##################################################
def NetBalanceType(value):
    """
    A simple function that returns "I will pay" if the price value is positive and "To be paid to me" if it is negative; a measure of total match value
    Inputs:
    - value: a price value
    Outputs:
    - a string, either "I will pay" or "To be paid to me" depending on value
    """
    if value < 0:
        return "I will pay"
    elif value >= 0:
        return "To be paid to me"
    else:
        raise Exception("Please ensure the value entered is a number.")


##################################################
def DFRateSum(df):
    """
    A simple function that calculates the sum of a pivot table column if that pivot is not empty, else returns zero if it is
    Inputs:
    - df: a dataframe; it must either have a "Rate" column or be empty
    Outputs:
    - either df.Rate.sum() or 0, depending
    """
    if not df.empty:
        return df.Rate.sum()
    elif df.empty:
        return 0
    else:
        raise Exception("Please ensure that a dataframe has been input and has a \"Rate\" column.")


##################################################
def CreateEmptyOutputJSON(matches_dir):
    """
    Creates an empty JSON output file in the case there are no trades
    Inputs:
    - matches_dir: output file location and name
    Outputs:
    - None
    """
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
        "Supplier Bid (USD/bbl)": [],
        "Bid Type": [],
        "Trucks Accepted": [],
        "Pipes Accepted": [],
        "Truck Max Dist (mi)": [],
        "Trucking Capacity (bpd)": [],
        "Truck Transport Bid (USD/bbl)": [],
        "Pipe Max Dist (mi)": [],
        "Pipeline Capacity (bpd)": [],
        "Pipe Transport Bid (USD/bbl)": [],
        "Matches": [],
        "Match Total Volume (bbl)":[],
        "Match Total Value (USD)":[],
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
        "Consumer Bid (USD/bbl)": [],
        "Bid Type": [],
        "Trucks Accepted": [],
        "Pipes Accepted": [],
        "Truck Max Dist (mi)": [],
        "Trucking Capacity (bpd)": [],
        "Truck Transport Bid (USD/bbl)": [],
        "Pipe Max Dist (mi)": [],
        "Pipeline Capacity (bpd)": [],
        "Pipe Transport Bid (USD/bbl)": [],
        "Matches": [],
        "Match Total Volume (bbl)":[],
        "Match Total Value (USD)":[],
    }

    # convert dictionaries to dataframes
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


def dev_check_outputs(model):
    """
    A function for developers to check price and profit values; just prints a handful of useful values
    """
    # dev_check section:
    print("Producer Prices")
    for key, val in model.p_ProducerNodalPrice.items():
        if val.value != 0 and val.value is not None:
            print(f"{model.p_ProducerNodalPrice.name}[{key}] = {val.value}")
    print("Consumer Prices")
    for key, val in model.p_ConsumerNodalPrice.items():
        if val.value != 0 and val.value is not None:
            print(f"{model.p_ConsumerNodalPrice.name}[{key}] = {val.value}")
    #print("Transport Prices") # This prints a lot of lines. Use with caution.
    #for key, val in model.p_TransportPrice.items():
    #    if val.value != 0 and val.value is not None:
    #        print(f"{model.p_TransportPrice.name}[{key}] = {val.value}")
    print("Producer volume profit")
    for key, value in model.p_ProducerVolumeProfit.items():
        if value.value != 0:
            print(f"{model.p_ProducerVolumeProfit.name}[{key}] = {value.value}")
    print("Consumer volume profit")
    for key, value in model.p_ConsumerVolumeProfit.items():
        if value.value != 0:
            print(f"{model.p_ConsumerVolumeProfit.name}[{key}] = {value.value}")
    print("Producer trucking profit")
    for key, value in model.p_ProducerTruckProfit.items():
        if value.value != 0:
            print(f"{model.p_ProducerTruckProfit.name}[{key}] = {value.value}")
    print("Producer piping profit")
    for key, value in model.p_ProducerPipelProfit.items():
        if value.value != 0:
            print(f"{model.p_ProducerPipelProfit.name}[{key}] = {value.value}")
    print("Consumer trucking profit")
    for key, value in model.p_ConsumerTruckProfit.items():
        if value.value != 0:
            print(f"{model.p_ConsumerTruckProfit.name}[{key}] = {value.value}")
    print("Consumer piping profit")
    for key, value in model.p_ConsumerPipelProfit.items():
        if value.value != 0:
            print(f"{model.p_ConsumerPipelProfit.name}[{key}] = {value.value}")
    return None