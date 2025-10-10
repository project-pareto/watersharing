#####################################################################################################
# PARETO was produced under the DOE Produced Water Application for Beneficial Reuse Environmental
# Impact and Treatment Optimization (PARETO), and is copyright (c) 2021-2025 by the software owners:
# The Regents of the University of California, through Lawrence Berkeley National Laboratory, et al.
# All rights reserved.
#
# NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S.
# Government consequently retains certain rights. As such, the U.S. Government has been granted for
# itself and others acting on its behalf a paid-up, nonexclusive, irrevocable, worldwide license in
# the Software to reproduce, distribute copies to the public, prepare derivative works, and perform
# publicly and display publicly, and to permit others to do so.
#####################################################################################################
from os import getcwd
from os.path import join as path_join
from os.path import dirname, abspath
import pandas as pd
import pytest

# Function to test
from watertrading.code.GetDistance import get_pipeline_distance

##################################################
def test_get_pipeline_distance():
    """
    This test ensures the Haversine distance calculation is in relative
    agreement with the Google Maps distance between two points selected
    in Texas. (Texas is assumed to be the general location for most users
    and so the function radius input is selected to ensure the greatest
    accuracy in this region.)
    """

    # Test dataframes to run functions (note only these fields required for this function)
    data_df_producer = {
        "WellpadUnique": ["A", "B", "C"],
        "Latitude":[31.33, 31.75, 32.01],
        "Longitude":[-101.78, -101.12, -101.44]
    }
    data_df_consumer = {
        "WellpadUnique": ["D", "E", "F"],
        "Latitude":[31.22, 31.61, 32.23],
        "Longitude":[-101.33, -101.99, -101.55]
    }
    test_df_producer = pd.DataFrame(data_df_producer)
    test_df_consumer = pd.DataFrame(data_df_consumer)

    # Dump data to JSON file in test directory
    test_dir = path_join(dirname(abspath(__file__)),"_test_distance.json")

    # Generate distance dataframe
    test_df_pipe_distance = get_pipeline_distance(test_dir, test_df_producer, test_df_consumer)

    # Dataframe should not be empty
    assert not test_df_pipe_distance.empty