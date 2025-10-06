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
"""
Test Trading Main Script
"""
import pytest

# Imports
import os
from watertrading.code import MainScript as MS

##################################################
def test_trading_code():
    """
    This test demonstrates the use of the water exchange process (trading)
    and provides a test confirming that the code (in its entirety) runs.
    Note that both this and the sharing test use the same source data.
    """
    # Designate data file names
    DISTANCE_JSON = "_distance.json"
    REQUESTS_JSON = "_requests.json"
    MATCHES_JSON_TRADING = "_matches_trading.json"
    PROFITS_JSON_TRADING = "_profits_trading.json"

    # Designate directories
    data_dir = os.getcwd()
    output_dir = os.path.join(data_dir,"trading","match-detail")
    file_names = {
        "requests":REQUESTS_JSON,
        "distance":DISTANCE_JSON,
        "matches":MATCHES_JSON_TRADING,
        "profits":PROFITS_JSON_TRADING
    }

    # Run main script; this test should cover most of the code
    MS.run_optimization_models(
        data_dir=data_dir,
        output_dir=output_dir,
        file_names=file_names,
        Update_distance_matrix=True
    )

    # Tests
    assert os.path.exists(MATCHES_JSON_TRADING)