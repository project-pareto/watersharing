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
def csv_2_json(input_dir,PRODUCER_CSV,CONSUMER_CSV,MIDSTREAM_CSV,name="jsonized_data.json"):
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
    df_producer['Start Date'] = pd.to_datetime(df_producer['Start Date'], format ="%m/%d/%Y")
    df_producer['End Date'] = pd.to_datetime(df_producer['End Date'], format ="%m/%d/%Y")
    df_producer['Start Date'] = df_producer['Start Date'].dt.strftime("%m/%d/%Y")
    df_producer['End Date']= df_producer['End Date'].dt.strftime("%m/%d/%Y")

    # Read in consumer data; convert date data to datetime to enforce proper format, and then to string for JSON export (datetime is not compatible)
    df_consumer = pd.read_csv(os.path.join(input_dir,CONSUMER_CSV))
    df_consumer['Start Date'] = pd.to_datetime(df_consumer['Start Date'], format ="%m/%d/%Y")
    df_consumer['End Date'] = pd.to_datetime(df_consumer['End Date'], format ="%m/%d/%Y")
    df_consumer['Start Date'] = df_consumer['Start Date'].dt.strftime("%m/%d/%Y")
    df_consumer['End Date']= df_consumer['End Date'].dt.strftime("%m/%d/%Y")

    # Read in midstream data; convert date data to datetime to enforce proper format, and then to string for JSON export (datetime is not compatible)
    df_midstream = pd.read_csv(os.path.join(input_dir,MIDSTREAM_CSV))
    df_midstream['Start Date'] = pd.to_datetime(df_midstream['Start Date'], format ="%m/%d/%Y")
    df_midstream['End Date'] = pd.to_datetime(df_midstream['End Date'], format ="%m/%d/%Y")
    df_midstream['Start Date'] = df_midstream['Start Date'].dt.strftime("%m/%d/%Y")
    df_midstream['End Date']= df_midstream['End Date'].dt.strftime("%m/%d/%Y")

    # Convert dataframes to dictionaries for export (we will use json.dump, not the pandas built-in function)
    d_producer = df_producer.to_dict(orient='records')
    d_consumer = df_consumer.to_dict(orient='records')
    d_midstream = df_midstream.to_dict(orient='records')
    d_restriction = dict() # this is initialized so that it exists in the output; it will be managed by WordPress

    # open JSON file and dump data under named headers - this is the agreed upon format
    with open(os.path.join(input_dir,name), "w") as data_file:
        json.dump({"Producers":d_producer,"Consumers":d_consumer,"Midstreams":d_midstream,"Restrictions":d_restriction}, data_file, indent=2)
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
    df_producer['Start Date'] = pd.to_datetime(df_producer['Start Date'], format ="%m/%d/%Y")
    df_producer['End Date'] = pd.to_datetime(df_producer['End Date'], format ="%m/%d/%Y")

    # Consumer inputs
    df_consumer = pd.DataFrame(data=request_data["Consumers"])
    # convert time inputs to datetime
    df_consumer['Start Date'] = pd.to_datetime(df_consumer['Start Date'], format ="%m/%d/%Y")
    df_consumer['End Date'] = pd.to_datetime(df_consumer['End Date'], format ="%m/%d/%Y")

    # Midstream inputs
    df_midstream = pd.DataFrame(data=request_data["Midstreams"])
    df_midstream['Start Date'] = pd.to_datetime(df_midstream['Start Date'], format ="%m/%d/%Y")
    df_midstream['End Date'] = pd.to_datetime(df_midstream['End Date'], format ="%m/%d/%Y")

    # Matching restrictions
    df_restrictions = pd.DataFrame(data=request_data["Restrictions"])
    return df_producer, df_consumer, df_midstream, df_restrictions


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
    df_matches_supply = pd.DataFrame(data=matches_data["Supply"])
    df_matches_demand = pd.DataFrame(data=matches_data["Demand"])
    df_matches_producer_transport = pd.DataFrame(data=matches_data["Transport (Producer)"])
    df_matches_midstream_transport = pd.DataFrame(data=matches_data["Transport (Midstream)"])
    # convert time inputs to datetime
    df_matches_supply["Date"] = pd.to_datetime(df_matches_supply["Date"], format ="%m/%d/%Y")
    df_matches_demand["Date"] = pd.to_datetime(df_matches_demand["Date"], format ="%m/%d/%Y")
    if not df_matches_producer_transport.empty:
        df_matches_producer_transport["Date"] = pd.to_datetime(df_matches_producer_transport["Date"], format ="%m/%d/%Y")
    if not df_matches_midstream_transport.empty:
        df_matches_midstream_transport["Departure Date"] = pd.to_datetime(df_matches_midstream_transport["Departure Date"], format ="%m/%d/%Y")
        df_matches_midstream_transport["Arrival Date"] = pd.to_datetime(df_matches_midstream_transport["Arrival Date"], format ="%m/%d/%Y")

    return df_matches_supply, df_matches_demand, df_matches_producer_transport, df_matches_midstream_transport


##################################################
def fetch_profit_data(PROFITS):
    """
    Inputs:
    > PROFITS - directory to matches JSON file
    Outputs:
    > df_profits - matches dataframe
    """

    input_dir = os.path.dirname(__file__)
    if not os.path.exists(input_dir):
                print("Profits files directory not found")

    # Pull in JSON request data
    with open(PROFITS, "r") as read_file:
        profits_data = json.load(read_file)

    # Producer inputs
    df_profits_supply = pd.DataFrame(data=profits_data["Supply"])
    df_profits_demand = pd.DataFrame(data=profits_data["Demand"])
    df_profits_producer_transport = pd.DataFrame(data=profits_data["Transport (Producer)"])
    df_profits_midstream_transport = pd.DataFrame(data=profits_data["Transport (Midstream)"])

    return df_profits_supply, df_profits_demand, df_profits_producer_transport, df_profits_midstream_transport


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
        dist = dist_matrix.loc[row["From wellpad"],row["To wellpad"]]
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
    #Line of code inspiring this function: [model.p_ProducerNodalPrice[row["Supplier Wellpad"],row["Date"]] for ind,row in df_v_Supply.iterrows()]
    # Type dictionary allowing function reuse:
    type_dict = {"Supplier":"Supplier Wellpad", "Consumer":"Consumer Wellpad"}
    # Add prices to list in order, use NaN values if price not defined
    prices = []
    for ind, row in df.iterrows():
        try:
            prices.append(price_source[row[type_dict[type]],row["Date Index"]].value)
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
        dist = dist_matrix.loc[row["From wellpad"],row["To wellpad"]]
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
        return dict(zip(uniques.loc[:,key_col],uniques.loc[:,val_col]))


##################################################
def GetDataFromGSheet():
    """
    Fetches input data from Google Sheets and returns input dataframes properly formatted
    """
    # Google Sheet IDs
    # Old Spreadsheet addresses
    #ConsumerGSheetID = "https://docs.google.com/spreadsheets/d/10ZlgdiiBP-KfjlTkleg1fRMJf3cpKqy4HowIioscGlA/export?format=csv&gid=944349706"
    #ProducerGSheetID = "https://docs.google.com/spreadsheets/d/1cAmxfLx3vjSX-roBGy2T7kaH7YzRxzKcNFrIPmgNn-k/export?format=csv&gid=1315829334"
    #MidstreamGSheetID = "https://docs.google.com/spreadsheets/d/1O2LHgQ1uqCdaNHB2K2EUKxicjGbazj6FH23lHeocZN4/export?format=csv&gid=18554279"
    # New Spreadsheet addresses (for No Negatives Update)
    ConsumerGSheetID = "https://docs.google.com/spreadsheets/d/1afO-P5kx5E4WmU8A741peLtKyfqkKRv173VrGJ-pjcY/export?format=csv&gid=1832445375"
    ProducerGSheetID = "https://docs.google.com/spreadsheets/d/1ZfPIMFDO-_o5SZV_456vpUInKrm2LAUf7mSTIQPeCGI/export?format=csv&gid=1508677222"
    MidstreamGSheetID = "https://docs.google.com/spreadsheets/d/1FEkkEb93gM73JxvuWAt3rmtyb_rcBADfhHB6MvGWf0A/export?format=csv&gid=339065539"

    # Download, read, and output data to dataframe all in one step with Pandas
    df_consumer_raw = pd.read_csv(ConsumerGSheetID)
    df_producer_raw = pd.read_csv(ProducerGSheetID)
    df_midstream_raw = pd.read_csv(MidstreamGSheetID)

    # Print the dataframes to make sure it worked (for testing purposes)
    #print(df_consumer_raw)
    #print(df_producer_raw)
    #print(df_midstream_raw)

    # Change column names to match the expected names in the original code
    #df2 = df2.rename(columns={'E': 'A', 'F': 'B', 'G': 'C', 'H': 'D'})
    df_consumer = df_consumer_raw.rename(columns={"Timestamp":"Index",
        "Organization Name":"Operator",
        "Wellpad ID":"Wellpad",
        "Wellpad Longitude":"Longitude",
        "Wellpad Latitude":"Latitude",
        "Demand start date":"Start Date",
        "Demand end date":"End Date",
        "Estimated demand rate (bpd)":"Demand Rate (bpd)",
        "How would you like to bid?":"Bid Coefficient",
        "Please enter the value of your bid (in USD/bbl)":"Consumer Bid (USD/bbl)"
        })
    
    df_producer = df_producer_raw.rename(columns={"Timestamp":"Index",
        "Organization Name":"Operator",
        "Wellpad ID":"Wellpad",
        "Wellpad Longitude":"Longitude",
        "Wellpad Latitude":"Latitude",
        "Production start date":"Start Date",
        "Production end date":"End Date",
        "Estimated production rate (bpd)":"Supply Rate (bpd)",
        "If you are willing to transport your water, how far? (Please enter a value in miles) If not, enter zero (0).":"Max Transport (bbl)",
        "How would you like to bid?":"Bid Coefficient",
        "Water selling price bid (USD/bbl)":"Supplier Bid (USD/bbl)",
        "Water transportation price bid (in USD/bbl, an entry of zero means you will provide this service for free)":"Transport Bid (USD/bbl)"
        })
    
    df_midstream = df_midstream_raw.rename(columns={"Timestamp":"Index",
        "Organization Name":"Operator",
        "Transport availability start date":"Start Date",
        "Transport availability end date":"End Date",
        "Total daily capacity (bpd)":"Total Capacity (bbl)",
        "Maximum single drive distance (miles)":"Max Transport (bbl)",
        "I want to be paid at least this much (in USD/bbl)":"Transport Bid (USD/bbl)",
        "Maximum transport delay (in days; a whole number)":"Lag (days)"
        })
    
    # Replace bid coefficient terms with +/-1 as appropriate, then 
    df_producer.replace(to_replace="I am willing to pay (up to)", value=-1, inplace=True)
    df_producer.replace(to_replace="I want to be paid (at least)", value=1, inplace=True)
    df_producer["Supplier Bid (USD/bbl)"] = df_producer.apply(lambda row: row["Supplier Bid (USD/bbl)"]*row["Bid Coefficient"], axis=1)

    df_consumer.replace(to_replace="I am willing to pay (up to)", value=1, inplace=True)
    df_consumer.replace(to_replace="I want to be paid (at least)", value=-1, inplace=True)
    df_consumer["Consumer Bid (USD/bbl)"] = df_consumer.apply(lambda row: row["Consumer Bid (USD/bbl)"]*row["Bid Coefficient"], axis=1)
    
    return df_consumer, df_producer, df_midstream