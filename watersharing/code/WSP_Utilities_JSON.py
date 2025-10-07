"""
A collection of utility functions that may be called from multiple modules, or which don't have
a more appropriate location.
"""
import os
import pandas as pd
import json


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
def csv_2_json(input_dir,PRODUCER_CSV,CONSUMER_CSV,name="jsonized_data.json"):
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
    df_producer = pd.read_csv(os.path.join(input_dir,PRODUCER_CSV))
    df_producer['Start Date'] = pd.to_datetime(df_producer['Start Date'], format ="%Y/%m/%d")
    df_producer['End Date'] = pd.to_datetime(df_producer['End Date'], format ="%Y/%m/%d")
    df_producer['Start Date'] = df_producer['Start Date'].dt.strftime("%Y/%m/%d")
    df_producer['End Date']= df_producer['End Date'].dt.strftime("%Y/%m/%d")

    # Read in consumer data; convert date data to datetime to enforce proper format, and then to string for JSON export (datetime is not compatible)
    df_consumer = pd.read_csv(os.path.join(input_dir,CONSUMER_CSV))
    df_consumer['Start Date'] = pd.to_datetime(df_consumer['Start Date'], format ="%Y/%m/%d")
    df_consumer['End Date'] = pd.to_datetime(df_consumer['End Date'], format ="%Y/%m/%d")
    df_consumer['Start Date'] = df_consumer['Start Date'].dt.strftime("%Y/%m/%d")
    df_consumer['End Date']= df_consumer['End Date'].dt.strftime("%Y/%m/%d")

    # Convert dataframes to dictionaries for export (we will use json.dump, not the pandas built-in function)
    d_producer = df_producer.to_dict(orient='records')
    d_consumer = df_consumer.to_dict(orient='records')
    d_restriction = dict() # this is initialized so that it exists in the output; it will be managed by WordPress

    # open JSON file and dump data under named headers - this is the agreed upon format
    with open(os.path.join(input_dir,name), "w") as data_file:
        json.dump({"Producers":d_producer,"Consumers":d_consumer,"Restrictions":d_restriction}, data_file, indent=2)
    return None


##################################################
def fetch_case_data(REQUESTS_JSON):
    """
    NOTE: This is a duplicate of get_data(); a "light" version for fetching data needed for plots
    Inputs:
    > REQUESTS_JSON - directory to requests JSON file
    Outputs:
    > df_producer - producer request dataframe
    > df_consumer - consumer request dataframe
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
    return df_producer, df_consumer, df_restrictions


##################################################
def fetch_distance_data(DISTANCE_JSON):
    """
    Inputs:
    > DISTANCE_JSON - directory to distance JSON file
    Outputs:
    > df_distance - distance dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
                print("Data directory not found")

    # Pull in JSON request data
    with open(DISTANCE_JSON, "r") as read_file:
        distance_data = json.load(read_file)

    # Producer inputs
    df_distance = pd.DataFrame(data=distance_data["DriveDistances"])

    return df_distance


##################################################
def fetch_match_data(MATCHES_JSON):
    """
    NOTE: This is a duplicate of get_data(); a "light" version for fetching data needed for plots
    Inputs:
    > MATCHES_JSON - directory to matches JSON file
    Outputs:
    > df_matches - matches dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
                print("Matches directory not found")

    # Pull in JSON request data
    with open(MATCHES_JSON, "r") as read_file:
        matches_data = json.load(read_file)

    # Producer inputs
    df_matches = pd.DataFrame(data=matches_data)
    # convert time inputs to datetime
    df_matches['Date'] = pd.to_datetime(df_matches['Date'], format ="%Y/%m/%d")

    return df_matches


##################################################
def merge_match_data(MATCHES_JSON):
    """
    Inputs:
    > MATCHES_JSON - directory to matches JSON file

    Outputs:
    > MERGED_MATCHES_JSON - updated matches JSON file merged across dates
    
    FUNCTION DESCRIPTION:
    > Load request and original match data from JSON file into dataframes
    > Create a new data frame for merged matches
    > Process original matches:
        > for each match in matches:
            > find other matches with same "from index," "to index," and "volume"
              (the date may differ)
            > create a new match
    """