"""
A collection of functions for plotting & visualization
"""
import matplotlib
matplotlib.use('agg')  # Set the backend before importing any other parts of matplotlib

import os
import pandas as pd
import json
from WSP_Utilities_JSON import fetch_case_data,fetch_match_data,fetch_distance_data
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from matplotlib.dates import date2num
from numpy import linspace

##################################################
def SummaryPlot(input_dir,MATCHES_JSON,REQUESTS_JSON,DISTANCE_JSON,fig_name="plot.png",annotate_plot=True):
    """
    A function that converts existing case study .csv data files into the newer JSON format
    Inputs:
        - producer csv data file
        - consumer csv data file
        - (optional) name: a name for the output file
    Outputs:
        - requests.JSON file
    """

    # Create arrays to add location data to matches
    #df_match_prod_lon = []
    #df_match_prod_lat = []
    #df_match_con_lon = []
    #df_match_con_lat = []

    # Reload problem data
    df_producer, df_consumer, df_restrictions = fetch_case_data(REQUESTS_JSON)
    df_match = fetch_match_data(MATCHES_JSON)
    df_distance = fetch_distance_data(DISTANCE_JSON)

    """
    # Generate distance column and add it to matches the dataframe
    #print(df_distance)
    distances = []
    for index, row in df_matches.iterrows():
        dist = df_distance.loc[row["From wellpad"],row["To wellpad"]]
        distances.append(dist)
    df_matches["Distance"] = distances
    

    # use Pandas merge() to add latitude and longitude data to the matches dataframe, then use that to plot lines colored by date.
    df_merge = pd.merge(df_matches,df_producer[["Index","Longitude","Latitude"]],left_on="From index", right_on="Index", how="left")
    df_merge = pd.merge(df_merge,df_consumer[["Index","Longitude","Latitude"]],left_on="To index",right_on="Index",how="left")
    #df_merge.rename(columns={"Longitude_x": "From Longitude", "Latitude_x": "From Latitude"})
    #df_merge.rename(columns={"Longitude_y": "To Longitude", "Latitude_y": "To Latitude"})
    """
    
    # add deltas for annotations
    dx = 0.07 # 5% of current axis range, manually calculated
    dy = 0.04 # 5% of current axis range, manually calculated

    # plot producer locations
    ax = df_producer.plot.scatter(x = 'Longitude', y = 'Latitude', s = 100, c = "black", marker="^")
    for i in range(len(df_producer.index)):
        ax.annotate(df_producer["Wellpad"][i], (df_producer["Longitude"][i]+dx, df_producer["Latitude"][i]+dy))
        ax.annotate(df_producer["Rate"][i], (df_producer["Longitude"][i]+dx, df_producer["Latitude"][i]))

    # add consumer locations to plot
    df_consumer.plot.scatter(x = 'Longitude', y = 'Latitude', s = 100, color = "black", marker="v", ax=ax)
    if annotate_plot:
        for i in range(len(df_consumer.index)):
            ax.annotate(df_consumer["Wellpad"][i], (df_consumer["Longitude"][i]+dx, df_consumer["Latitude"][i]+dy))
            ax.annotate(df_consumer["Rate"][i], (df_consumer["Longitude"][i]+dx, df_consumer["Latitude"][i]))

    ax.invert_xaxis()
    ax.get_figure().savefig(os.path.join(input_dir,"plot_well_locs.png"))

    # add matches as lines
    #df_merge.plot.scatter(x=["Longitude_x","Longitude_y"],y=["Latitude_x","Latitude_y"], c="black", style="-", ax=ax)
    #df_merge.plot.line(x=["Longitude_x","Longitude_y"],y=["Latitude_x","Latitude_y"], c="black", style="-", ax=ax)
    date_nums_range = date2num(df_match["Date"])
    vmin = min(date_nums_range)
    vmax = max(date_nums_range)
    #vmin_norm = (vmin - vmin)/(vmax - vmin)
    #vmax_norm = (vmax - vmin)/(vmax - vmin)
    colors = pl.cm.coolwarm(linspace(0,1,100))
    """
    print(vmin)
    print(vmax)
    for idx, row in df_match.iterrows():
        _x = [row["Longitude_x"], row["Longitude_y"]]
        _y = [row["Latitude_x"], row["Latitude_y"]]
        c = date2num(row["Date"])
        ax.plot(_x, _y, linestyle="-", linewidth=1, c=c, cmap=plt.cm.coolwarm,vmin=vmin,vmax=vmax)
    """
    lines_x = []
    lines_y = []
    date_nums = []
    for idx, row in df_match.iterrows():
        _x = [row["From Longitude"], row["To Longitude"]]
        _y = [row["From Latitude"], row["To Latitude"]]
        #lines_x.append(_x[0])
        #lines_x.append(_x[1])
        #date_nums.append(date2num(row["Date"].to_pydatetime()))
        #lines_y.append(_y[0])
        #lines_y.append(_y[1])
        #date_nums.append(date2num(row["Date"].to_pydatetime()))
        #lines_x.append(None) # to separate lines, or they will be connected
        #lines_y.append(None) # for each adjacent row
        #date_nums.append(vmin)
        #ax.plot(lines_x, lines_y, linestyle="-", linewidth=1, c=date_nums, cmap=plt.cm.coolwarm,vmin=vmin,vmax=vmax)
        #ax.plot(_x, _y, linestyle="-", linewidth=1, c=(date2num(row["Date"]) - vmin)/(vmax - vmin), cmap="coolwarm",vmin=vmin_norm,vmax=vmax_norm)
        ax.plot(_x, _y, linestyle="-", linewidth=1, c=colors[int(100*(date2num(row["Date"])-vmin)/(vmax-vmin) - 1)])
    #print(len(lines_x))
    #print(len(date_nums))
    #ax.plot(lines_x, lines_y, linestyle="-", linewidth=1, color="black")
    #"""

    #cmap = mpl.cm.coolwarm
    #norm = mpl.colors.Normalize(vmin=0, vmax=1)
    #plot1.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
    #         cax=ax, orientation='vertical', label='Date')

    # Save final figure
    plot1 = ax.get_figure()
    plot1.savefig(os.path.join(input_dir,fig_name))

    # Print some useful stats
    match_total = df_match["Rate"].sum(axis=0, skipna=True, numeric_only=False, min_count=0)
    print("Total matched volume: ",match_total)
    
    match_volume_dist = sum(df_match["Rate"]*df_match["Distance"])
    print("Total volume-distance in bbl-mi: ",match_volume_dist)

    avg_dist = match_volume_dist/match_total
    print("Average distance per barrel: ",avg_dist)

    print("Maximum network distance: ",df_distance.max(numeric_only=True).max())

    # Export merged match file
    #print("Matches:")
    #print(df_matches)
    #print("Merged:")
    #print(df_match)
    #df_match.to_excel("merged_matches.xlsx") # save to excel for convenience
    df_match['Date'] = df_match['Date'].dt.strftime("%Y/%m/%d") # convert date to because DateTime isn't compatible with JSON
    d_match = df_match.to_dict(orient='records') # convert dataframe to dict; we'll use JSON.py, not the pandas built-in JSON export (more formatting control)
    MATCHES_PLUS_JSON = MATCHES_JSON.replace('.json', '_plus.json') # add "plus" to the file name so we don't overwrite the original
    with open(MATCHES_PLUS_JSON, "w") as data_file:
        json.dump(d_match, data_file, indent=2)
