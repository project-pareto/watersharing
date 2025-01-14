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
Functions related to water quality management are kept here for convenient grouping, separated
comveniently from the primary code base.
"""

def quality_overlap_check(producer_qualities,consumer_qualities):
    """
    Checks whether any of the specified quality overlap between the two stakeholders
    Inputs:
    - producer_qualities: a list of quality specifications, containing some non-"" entries
    - consumer_qualities: a list of quality specifications, containing some non-"" entries
    Outputs:
    - True if overlap, False if no overlap
    """
    # If any are aligned, return True, else False
    for i in range(len(producer_qualities)):
        if producer_qualities[i] != 0 and consumer_qualities[i] != 0:
            return True
    # if it gets here, there is no overlap
    return False


def quality_entry_check(quality_list):
    """
    A preliminary check to determine whether quality evluation is required
    Inputs:
    - quality_list: a list of quality specifications, possible containing some or all "0" entries
    Outputs:
    - A boolean; True if any quality specified, False if not
    Quality metric values are keyed to:
        "TSS"
        "TDS"
        "Chloride"
        "Barium"
        "Calcium carbonates"
        "Iron"
        "Boron"
        "Hydrogen Sulfide"
        "NORM"
    Quality constraint types are keyed to:
        "TSS Constraint"
        "TDS Constraint"
        "Chloride Constraint"
        "Barium Constraint"
        "Calcium carbonates Constraint"
        "Iron Constraint"
        "Boron Constraint"
        "Hydrogen Sulfide Constraint"
        "NORM Constraint"
    """
    if all(i == 0 for i in quality_list):
        # No quality disclosures were made
        return False
    else:
        # there is at least one quality disclosure
        return True


def match_quality_check(pi,ci,df_producer,df_consumer,match_qual_dict):
    """
    A function to check whether two requests have compatible quality data, if any.
    Possible values are "All specifications met", "Specifications met with mismatches"
    and "False"
    Inputs:
    - pi: a producer request index to match
    - ci: a consumer request index to match
    - df_producer: producer dataframe with quality information
    - df_consumer: consumer dataframe with quality information
    Outputs:
    - A quality match flag, either: "Best match did not include quality details" "No quality violations" or "False"
    Quality metric values are keyed to:
        "TSS"
        "TDS"
        "Chloride"
        "Barium"
        "Calcium carbonates"
        "Iron"
        "Boron"
        "Hydrogen Sulfide"
        "NORM"
    Quality constraint types are keyed to:
        "TSS Constraint"
        "TDS Constraint"
        "Chloride Constraint"
        "Barium Constraint"
        "Calcium carbonates Constraint"
        "Iron Constraint"
        "Boron Constraint"
        "Hydrogen Sulfide Constraint"
        "NORM Constraint"
    """
    # Set up quality information for each request pi and ci
    producer_qualities = [
        df_producer.loc[pi, "TSS"],
        df_producer.loc[pi, "TDS"],
        df_producer.loc[pi, "Chloride"],
        df_producer.loc[pi, "Barium"],
        df_producer.loc[pi, "Calcium carbonates"],
        df_producer.loc[pi, "Iron"],
        df_producer.loc[pi, "Boron"],
        df_producer.loc[pi, "Hydrogen Sulfide"],
        df_producer.loc[pi, "NORM"]
    ]
    consumer_qualities = [
        df_consumer.loc[ci, "TSS"],
        df_consumer.loc[ci, "TDS"],
        df_consumer.loc[ci, "Chloride"],
        df_consumer.loc[ci, "Barium"],
        df_consumer.loc[ci, "Calcium carbonates"],
        df_consumer.loc[ci, "Iron"],
        df_consumer.loc[ci, "Boron"],
        df_consumer.loc[ci, "Hydrogen Sulfide"],
        df_consumer.loc[ci, "NORM"]
    ]
    # Cast numerical entries to floats, unless using the "" value for no entry
    producer_qualities = [float(producer_qualities[i]) if producer_qualities[i] != "" else "" for i in range(len(producer_qualities))]
    consumer_qualities = [float(consumer_qualities[i]) if consumer_qualities[i] != "" else "" for i in range(len(consumer_qualities))]
    producer_quality_constraints = [
        df_producer.loc[pi, "TSS Constraint"],
        df_producer.loc[pi, "TDS Constraint"],
        df_producer.loc[pi, "Chloride Constraint"],
        df_producer.loc[pi, "Barium Constraint"],
        df_producer.loc[pi, "Calcium carbonates Constraint"],
        df_producer.loc[pi, "Iron Constraint"],
        df_producer.loc[pi, "Boron Constraint"],
        df_producer.loc[pi, "Hydrogen Sulfide Constraint"],
        df_producer.loc[pi, "NORM Constraint"]
    ]
    consumer_quality_constraints = [
        df_consumer.loc[ci, "TSS Constraint"],
        df_consumer.loc[ci, "TDS Constraint"],
        df_consumer.loc[ci, "Chloride Constraint"],
        df_consumer.loc[ci, "Barium Constraint"],
        df_consumer.loc[ci, "Calcium carbonates Constraint"],
        df_consumer.loc[ci, "Iron Constraint"],
        df_consumer.loc[ci, "Boron Constraint"],
        df_consumer.loc[ci, "Hydrogen Sulfide Constraint"],
        df_consumer.loc[ci, "NORM Constraint"]
    ]
    # Check for presence of quality entries
    producer_quality_check = quality_entry_check(producer_qualities)
    consumer_quality_check = quality_entry_check(consumer_qualities)
    # Begin quality conditional check
    if (not producer_quality_check) or (not consumer_quality_check):
        # One, the other, or both did not speficy quality
        match_qual_dict[pi,ci,"supply side"] = "Best match did not include quality details"
        match_qual_dict[pi,ci,"demand side"] = "Best match did not include quality details"
        return True
    else: #equivalent to: elif producer_quality_check and consumer_quality_check:
        # Both included quality data; now check whether quality specs overlap, then evaluate match potential
        if not quality_overlap_check(producer_qualities,consumer_qualities):
            # If there is no quality overlap, allow match but note issue
            match_qual_dict[pi,ci,"supply side"] = "Best match did not include quality details for specified disclosures"
            match_qual_dict[pi,ci,"demand side"] = "Best match did not include quality details for specified disclosures"
            return True
        else: # there is a quality overlap, we need a full check
            # start quality loop
            for i in range(len(producer_qualities)):
                if producer_qualities[i] != "" and consumer_qualities[i] != "":
                    # check this instance for potential conflict
                    if consumer_quality_constraints[i] == "lt":
                        if (producer_qualities[i] > consumer_qualities[i]):
                            # producer spec violates consumer spec
                            return False
                        #else, continue
                    elif consumer_quality_constraints[i] == "gt":
                        if (producer_qualities[i] < consumer_qualities[i]):
                            # producer spec violates consumer spec
                            return False
                        #else, continue
            # If we complete the loop, then the match is allowed
            match_qual_dict[pi,ci,"supply side"] = "No quality violations"
            match_qual_dict[pi,ci,"demand side"] = "No quality violations"
            return True
