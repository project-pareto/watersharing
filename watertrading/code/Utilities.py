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
def csv_2_json(
    input_dir, PRODUCER_CSV, CONSUMER_CSV, name="jsonized_data.json"
):
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
    df_producer["Start Date"] = pd.to_datetime(
        df_producer["Start Date"], format="%Y/%m/%d"
    )
    df_producer["End Date"] = pd.to_datetime(df_producer["End Date"], format="%Y/%m/%d")
    df_producer["Start Date"] = df_producer["Start Date"].dt.strftime("%Y/%m/%d")
    df_producer["End Date"] = df_producer["End Date"].dt.strftime("%Y/%m/%d")

    # Read in consumer data; convert date data to datetime to enforce proper format, and then to string for JSON export (datetime is not compatible)
    df_consumer = pd.read_csv(os.path.join(input_dir, CONSUMER_CSV))
    df_consumer["Start Date"] = pd.to_datetime(
        df_consumer["Start Date"], format="%Y/%m/%d"
    )
    df_consumer["End Date"] = pd.to_datetime(df_consumer["End Date"], format="%Y/%m/%d")
    df_consumer["Start Date"] = df_consumer["Start Date"].dt.strftime("%Y/%m/%d")
    df_consumer["End Date"] = df_consumer["End Date"].dt.strftime("%Y/%m/%d")

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
def add_dataframe_supply_coords(df, df_producer):
    """
    Inputs:
    > df - a dataframe including producer

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
def GenerateCombinationDataFrame(df1,df2,name,newname1,newname2):
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
    pd.options.mode.chained_assignment = None # to stop pandas warning us about the behavior we are using
    df1["temp"] = 1
    df2["temp"] = 1
    # rename columns for output (and to avoid merge)
    df1.rename(columns={name: newname1},inplace=True)
    df2.rename(columns={name: newname2},inplace=True)
    # merge: df_cross =
    return df1.merge(df2, on="temp", how="outer").drop(columns="temp")


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
    if value >= 0:
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
    if value >= 0:
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
    if value >= 0:
        return "To be paid to me"
    else:
        raise Exception("Please ensure the value entered is a number.")