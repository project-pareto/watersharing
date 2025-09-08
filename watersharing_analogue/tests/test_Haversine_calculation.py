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
Test utilities
"""
import pytest

# Function to test
from watertrading.code.GetDistance import Haversine_calcualtion

##################################################
def test_Haversine_calculation():
    """
    This test ensures the Haversine distance calculation is in relative
    agreement with the Google Maps distance between two points selected
    in Texas. (Texas is assumed to be the general location for most users
    and so the function radius input is selected to ensure the greatest
    accuracy in this region.)
    """
    # FIrst point: ~Odessa
    lat1 = 31.846260265163785
    lon1 = -102.3651954674979

    # Second point: ~Stanton
    lat2 = 32.1385652166999
    lon2 = -101.78910082956584

    #NOTE: the Google Maps distance between these two points is 39.34 miles; assume =/-5% accuracy
    measured_dist = 39.34
    UB = 1.05
    LB = 0.95

    # Tests
    test_dist = Haversine_calcualtion(lat1, lon1, lat2, lon2)
    assert test_dist <= UB*measured_dist and test_dist >= LB*measured_dist