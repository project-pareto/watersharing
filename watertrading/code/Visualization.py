"""
A collection of functions for plotting & visualization
"""
# for basic plots
import os
import pandas as pd
import json
from Utilities import (
    fetch_case_data,
    fetch_match_data,
    fetch_distance_data,
    fetch_profit_data,
)
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
import numpy as np
from matplotlib.dates import date2num
from scipy.interpolate import RBFInterpolator

# for interactive Jupyter notebook plots
import plotly.graph_objects as go
from ipywidgets import Layout, HTML, SelectionSlider, interact
from ipyleaflet import (
    Map,
    basemaps,
    basemap_to_tiles,
    CircleMarker,
    Marker,
    MarkerCluster,
    AwesomeIcon,
    Popup,
    Polyline,
    LayerGroup,
    LegendControl,
)

# import ipywidgets as widgets


##################################################
def SummaryPlot(
    input_dir, MATCHES, requests, distance, fig_name="plot.png", annotate_plot=True
):
    """
    Creates a basic summary plot illustrating stakeholder positioning and connectivity
    """

    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)
    df_distance = fetch_distance_data(distance)

    # add deltas for annotations
    dx = 0.04  # 5% of current axis range, manually calculated
    dy = 0.04  # 5% of current axis range, manually calculated

    # plot producer locations
    ax = df_producer.plot.scatter(
        x="Longitude", y="Latitude", s=100, c="red", marker="^"
    )
    if annotate_plot:
        for i in range(len(df_producer.index)):
            ax.annotate(
                df_producer["Wellpad"][i],
                (df_producer["Longitude"][i] + dx, df_producer["Latitude"][i] + dy),
            )
            ax.annotate(
                df_producer["Supply Rate (bpd)"][i],
                (df_producer["Longitude"][i] + dx, df_producer["Latitude"][i]),
            )

    # add consumer locations to plot
    df_consumer.plot.scatter(
        x="Longitude", y="Latitude", s=100, color="blue", marker="v", ax=ax
    )
    if annotate_plot:
        for i in range(len(df_consumer.index)):
            ax.annotate(
                df_consumer["Wellpad"][i],
                (df_consumer["Longitude"][i] + dx, df_consumer["Latitude"][i] + dy),
            )
            ax.annotate(
                df_consumer["Demand Rate (bpd)"][i],
                (df_consumer["Longitude"][i] + dx, df_consumer["Latitude"][i]),
            )

    # ax.invert_xaxis()
    ax.get_figure().savefig(os.path.join(input_dir, "plot_well_locs.png"))

    # add matches as lines
    colors = pl.cm.coolwarm(np.linspace(0, 1, 100))
    lines_x = []
    lines_y = []
    date_nums = []
    for idx, row in df_matches_midstream_transport.iterrows():
        _x = [row["From Longitude"], row["To Longitude"]]
        _y = [row["From Latitude"], row["To Latitude"]]
        ax.plot(_x, _y, linestyle="-", linewidth=1, c="gold")

    # Save final figure
    plot1 = ax.get_figure()
    plot1.savefig(os.path.join(input_dir, fig_name))

    # Print some useful stats
    """
    match_total = df_merge["value"].sum(axis=0, skipna=True, numeric_only=False, min_count=0)
    print("Total matched volume: ",match_total)
    
    match_volume_dist = sum(df_merge["value"]*df_merge["Distance"])
    print("Total volume-distance in bbl-mi: ",match_volume_dist)

    avg_dist = match_volume_dist/match_total
    print("Average distance per barrel: ",avg_dist)

    print("Maximum network distance: ",df_distance.max(numeric_only=True).max())
    

    # Export merged match file
    #print("Matches:")
    #print(df_matches)
    #print("Merged:")
    print(df_merge)
    df_merge.to_excel("merged_matches.xlsx") # save to excel for convenience
    df_merge['Date'] = df_merge['Date'].dt.strftime("%Y/%m/%d") # convert date to because DateTime isn't compatible with JSON
    d_merge = df_merge.to_dict(orient='records') # convert dataframe to dict; we'll use JSON.py, not the pandas built-in JSON export (more formatting control)
    MATCHES_PLUS_JSON = MATCHES_JSON.replace('.json', '_plus.json') # add "plus" to the file name so we don't overwrite the original
    with open(MATCHES_PLUS_JSON, "w") as data_file:
        json.dump(d_merge, data_file, indent=2)
    """
    return plot1


def SupplyPlot(input_dir, MATCHES, requests, fig_name="plot_supply.png"):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    SupplyPlotBar(input_dir, MATCHES, requests, ax)
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def SupplyPlotBar(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of supplier allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    BaseRates = dict(zip(df_producer["Index"], df_producer["Supply Rate (bpd)"]))

    # generate lists of producer allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    for ind, loc in df_matches_supply.iterrows():
        x_coords.append(ind + 1)
        back_heights.append(BaseRates[loc["Supplier Index"]])
        fore_heights.append(loc["Rate"])
        labels.append(
            loc["Supplier Index"] + " (" + loc["Date"].strftime("%Y/%m/%d") + ")"
        )

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="red")
    ax.bar(labels, height=fore_heights, color="red", edgecolor="red")
    plt.xticks(rotation=75)
    plt.xlabel("Producer ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))
    return None


##################################################
def SupplyPlotBarByOperator(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of supplier allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    df_producer["Operator-Wellpad"] = (
        df_producer["Operator"] + "-" + df_producer["Wellpad"]
    )
    df_matches_supply["Operator-Wellpad"] = (
        df_matches_supply["Operator"] + "-" + df_matches_supply["Supplier Wellpad"]
    )
    BaseRates = pd.pivot_table(
        df_producer, values="Supply Rate (bpd)", index="Operator-Wellpad", aggfunc="sum"
    )
    AllocationRates = pd.pivot_table(
        df_matches_supply, values="Rate", index="Operator-Wellpad", aggfunc="sum"
    )

    # generate lists of producer allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    bar_coord = 0
    for ind, loc in AllocationRates.iterrows():
        bar_coord += 1
        x_coords.append(bar_coord)
        back_heights.append(BaseRates.loc[ind]["Supply Rate (bpd)"])
        fore_heights.append(AllocationRates.loc[ind]["Rate"])
        labels.append(ind)
        # labels.append(ind +" ("+loc["Date"].strftime('%Y/%m/%d')+")")

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="red")
    ax.bar(labels, height=fore_heights, color="red", edgecolor="red")
    ymin, ymax = GetLims(back_heights)
    ax.set_ylim([None, ymax])
    ax.tick_params(axis="x", labelrotation=45)
    plt.xlabel("Producer Wellpad ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))
    return None


##################################################
def DemandPlot(input_dir, MATCHES, requests, fig_name="plot_demand.png"):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    DemandPlotBar(input_dir, MATCHES, requests, ax)
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def DemandPlotBar(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of supplier allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    BaseRates = dict(zip(df_consumer["Index"], df_consumer["Demand Rate (bpd)"]))

    # generate lists of producer allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    for ind, loc in df_matches_demand.iterrows():
        x_coords.append(ind + 1)
        back_heights.append(BaseRates[loc["Consumer Index"]])
        fore_heights.append(loc["Rate"])
        labels.append(
            loc["Consumer Index"] + " (" + loc["Date"].strftime("%Y/%m/%d") + ")"
        )

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="blue")
    ax.bar(labels, height=fore_heights, color="blue", edgecolor="blue")
    plt.xticks(rotation=75)
    plt.xlabel("Consumer ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))
    return None


##################################################
def DemandPlotBarByOperator(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of supplier allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    df_consumer["Operator-Wellpad"] = (
        df_consumer["Operator"] + "-" + df_consumer["Wellpad"]
    )
    df_matches_demand["Operator-Wellpad"] = (
        df_matches_demand["Operator"] + "-" + df_matches_demand["Consumer Wellpad"]
    )
    BaseRates = pd.pivot_table(
        df_consumer, values="Demand Rate (bpd)", index="Operator-Wellpad", aggfunc="sum"
    )
    AllocationRates = pd.pivot_table(
        df_matches_demand, values="Rate", index="Operator-Wellpad", aggfunc="sum"
    )

    # generate lists of consumer allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    bar_coord = 0
    for ind, loc in AllocationRates.iterrows():
        bar_coord += 1
        x_coords.append(bar_coord)
        back_heights.append(BaseRates.loc[ind]["Demand Rate (bpd)"])
        fore_heights.append(AllocationRates.loc[ind]["Rate"])
        labels.append(ind)
        # labels.append(ind +" ("+loc["Date"].strftime('%Y/%m/%d')+")")

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="blue")
    ax.bar(labels, height=fore_heights, color="blue", edgecolor="blue")
    ymin, ymax = GetLims(back_heights)
    ax.set_ylim([None, ymax])
    ax.tick_params(axis="x", labelrotation=45)
    plt.xlabel("Consumer ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))
    return None


##################################################
def MidstreamPlot(input_dir, MATCHES, requests, fig_name="plot_midstream.png"):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    MidstreamPlotBar(input_dir, MATCHES, requests, ax)
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def MidstreamPlotBar(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of supplier allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    BaseRates = dict(zip(df_midstream["Index"], df_midstream["Total Capacity (bbl)"]))

    # generate lists of midstream allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    pivot = pd.pivot_table(
        df_matches_midstream_transport[["Midstream Index", "Rate"]],
        index="Midstream Index",
        values="Rate",
        aggfunc="sum",
    )
    x = 0
    for ind, loc in pivot.iterrows():
        x += 1
        x_coords.append(x)
        back_heights.append(BaseRates[ind])
        fore_heights.append(loc["Rate"])
        labels.append(ind)

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="gold")
    ax.bar(labels, height=fore_heights, color="gold", edgecolor="gold")
    plt.xticks(rotation=75)
    plt.xlabel("Midstream ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))
    return None


##################################################
def MidstreamPlotBarByOperator(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of supplier allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    BaseRates = dict(
        zip(df_midstream["Operator"], df_midstream["Total Capacity (bbl)"])
    )

    # generate lists of midstream allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    try:
        pivot = pd.pivot_table(
            df_matches_midstream_transport[["Operator", "Rate"]],
            index="Operator",
            values="Rate",
            aggfunc="sum",
        )
        pivot.reset_index(inplace=True)
    except:
        pivot = pd.DataFrame(
            {"Operator": key, "Rate": 0} for key in df_midstream["Operator"]
        )

    x = 0
    for ind, loc in pivot.iterrows():
        x += 1
        x_coords.append(x)
        back_heights.append(BaseRates[loc["Operator"]])
        fore_heights.append(loc["Rate"])
        labels.append(loc["Operator"])

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="gold")
    ax.bar(labels, height=fore_heights, color="gold", edgecolor="gold")
    ymin, ymax = GetLims(back_heights)
    ax.set_ylim([None, ymax])
    ax.tick_params(axis="x", labelrotation=45)
    plt.xlabel("Midstream ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))
    return None


##################################################
def ProducerTransportPlot(
    input_dir, MATCHES, requests, fig_name="plot_producer_transport.png"
):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ProducerTransportPlotBar(input_dir, MATCHES, requests, ax)
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def ProducerTransportPlotBar(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of producer transport allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    BaseRates = dict(zip(df_producer["Index"], df_producer["Supply Rate (bpd)"]))

    # generate lists of midstream allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    try:
        pivot = pd.pivot_table(
            df_matches_producer_transport[["Carrier Index", "Rate"]],
            index="Operator",
            values="Rate",
            aggfunc="sum",
        )
        pivot = pivot.rename_axis(None, axis=1).reset_index()
    except:
        pivot = pd.DataFrame(
            {"Operator": key, "Rate": 0} for key in df_producer["Index"]
        )

    x = 0
    for ind, loc in pivot.iterrows():
        x += 1
        x_coords.append(x)
        back_heights.append(BaseRates[loc["Carrier Index"]])
        fore_heights.append(loc["Rate"])
        labels.append(loc["Carrier Index"])

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="orange")
    ax.bar(labels, height=fore_heights, color="orange", edgecolor="orange")
    ymin, ymax = GetLims(back_heights)
    ax.set_ylim([None, ymax])
    plt.xticks(rotation=75)
    plt.xlabel("Producer ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    # get plot, save a copy, and return handle
    return None

    ##################################################


def ProducerTransportPlotBarByOperator(input_dir, MATCHES, requests, ax):
    """
    Creates a basic bar plot of producer transport allocations
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # create lookup dictionary
    BaseRates = dict(zip(df_producer["Index"], df_producer["Supply Rate (bpd)"]))

    # generate lists of midstream allocations
    x_coords = []
    back_heights = []
    fore_heights = []
    labels = []
    try:
        pivot = pd.pivot_table(
            df_matches_producer_transport[["Carrier Index", "Rate"]],
            index="Carrier Index",
            values="Rate",
            aggfunc="sum",
        )
        pivot.reset_index(inplace=True)
    except:
        pivot = pd.DataFrame(
            {"Carrier Index": key, "Rate": 0} for key in df_producer["Index"]
        )

    x = 0
    for ind, loc in pivot.iterrows():
        x += 1
        x_coords.append(x)
        back_heights.append(BaseRates[loc["Carrier Index"]])
        fore_heights.append(loc["Rate"])
        labels.append(df_producer.loc[ind]["Operator"])

    # plot bars
    ax.bar(labels, height=back_heights, color="white", edgecolor="orange")
    ax.bar(labels, height=fore_heights, color="orange", edgecolor="orange")
    ymin, ymax = GetLims(back_heights)
    ax.set_ylim([None, ymax])
    ax.tick_params(axis="x", labelrotation=45)
    plt.xlabel("Producer ID")
    plt.ylabel("Volume [bbl]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    return None


##################################################
def AllocationPlot(input_dir, MATCHES, requests, fig_name="Allocations_AllInOne"):
    """
    Plots allocations on a series of subplots
    """

    # Create figure and axes handles
    fig = plt.figure(figsize=(8, 8))  # W,H
    ax1 = plt.subplot(4, 1, 1)
    ax2 = plt.subplot(4, 1, 2)
    ax3 = plt.subplot(4, 1, 3)
    ax4 = plt.subplot(4, 1, 4)
    SupplyPlotBar(input_dir, MATCHES, requests, ax1)
    DemandPlotBar(input_dir, MATCHES, requests, ax2)
    MidstreamPlotBar(input_dir, MATCHES, requests, ax3)
    ProducerTransportPlotBar(input_dir, MATCHES, requests, ax4)
    ax1.set(xlabel="Producer", ylabel="Volume Supplied [bbl]")
    ax2.set(xlabel="Consumer", ylabel="Volume Consumed [bbl]")
    ax3.set(xlabel="Midstream", ylabel="Volume Transported [bbl]")
    ax4.set(xlabel="Producer Transport")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    # save figure and return plot handle
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def AllocationPlotByOperator(
    input_dir, MATCHES, requests, fig_name="Allocations_AllInOne"
):
    """
    Plots allocations on a series of subplots
    """

    # Create figure and axes handles
    fig = plt.figure(figsize=(8, 8))  # W,H
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 2, 5)
    ax4 = plt.subplot(3, 2, 6)
    SupplyPlotBarByOperator(input_dir, MATCHES, requests, ax1)
    DemandPlotBarByOperator(input_dir, MATCHES, requests, ax2)
    MidstreamPlotBarByOperator(input_dir, MATCHES, requests, ax3)
    ProducerTransportPlotBarByOperator(input_dir, MATCHES, requests, ax4)
    ax1.set(xlabel="Producer", ylabel="Volume Supplied [bbl]")
    ax2.set(xlabel="Consumer", ylabel="Volume Consumed [bbl]")
    ax3.set(xlabel="Midstream", ylabel="Volume Transported [bbl]")
    ax4.set(xlabel="Producer Transport")
    ax1.set_title(f"PW Supplied", ha="left", x=-0)
    ax2.set_title(f"PW Consumed", ha="left", x=-0)
    ax3.set_title(f"PW Transported", ha="left", x=-0)
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    # save figure and return plot handle
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def InteractiveMapPlot(input_dir, MATCHES, requests):
    """
    Creates an interactive plot for presentation in a Jupyter notebook
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # get center point for map based on all producer/consumer coordinates
    all_coords = pd.concat(
        [
            df_producer[["Longitude", "Latitude"]],
            df_consumer[["Longitude", "Latitude"]],
        ],
        ignore_index=True,
    )
    mean_lon = all_coords["Longitude"].mean()
    mean_lat = all_coords["Latitude"].mean()

    # Set up base map
    m = Map(
        basemap=basemap_to_tiles(basemaps.OpenStreetMap.Mapnik),
        center=(mean_lat, mean_lon),
        zoom=11,
        max_zoom=14,
        min_zoom=7,
        scroll_wheel_zoom=True,
        layout=Layout(width="100%", height="700px"),
    )

    # Create icons
    prod_icon = AwesomeIcon(
        name="chevron-circle-up", marker_color="red", icon_color="white"
    )
    cons_icon = AwesomeIcon(
        name="chevron-circle-down", marker_color="blue", icon_color="white"
    )

    # Add lines to map
    for ind, loc in df_matches_midstream_transport.iterrows():
        polyline = Polyline(
            locations=[
                (loc["From Latitude"], loc["From Longitude"]),
                (loc["To Latitude"], loc["To Longitude"]),
            ],
            color="Gold",
            weight=4,
            text=loc["Operator"],
            textposition="center",
        )
        polyline.popup = HTML(
            loc["Operator"]
            + ": "
            + loc["From wellpad"]
            + " to "
            + loc["To wellpad"]
            + " ("
            + str(loc["Rate"])
            + ")"
        )
        m.add_layer(polyline)
    for ind, loc in df_matches_producer_transport.iterrows():
        polyline = Polyline(
            locations=[
                (loc["From Latitude"], loc["From Longitude"]),
                (loc["To Latitude"], loc["To Longitude"]),
            ],
            color="Orange",
            weight=4,
            text=loc["Operator"],
            textposition="center",
        )
        polyline.popup = HTML(
            loc["Operator"]
            + ": "
            + loc["From wellpad"]
            + " to "
            + loc["To wellpad"]
            + " ("
            + str(loc["Rate"])
            + ")"
        )
        m.add_layer(polyline)

    lat_offset = 0.025
    # Add icons to map
    for ind, loc in df_producer.iterrows():
        # marker = Marker(icon=prod_icon,
        #    mode="icons+text",
        #    location=[loc["Latitude"],loc["Longitude"]],
        #    title = loc["Wellpad"])
        # marker.text = HTML(loc["Operator"]+"-"+loc["Wellpad"])
        # m.add_layer(marker)
        message = HTML()
        message.value = loc["Operator"] + "-" + loc["Wellpad"]
        popup = Popup(
            location=[loc["Latitude"] + lat_offset, loc["Longitude"]],
            child=message,
            close_button=False,
            auto_close=False,
            close_on_escape_key=False,
        )
        m.add_layer(popup)
        m.add_layer(
            Marker(
                icon=prod_icon,
                mode="markers+text",
                location=[loc["Latitude"], loc["Longitude"]],
                title=loc["Wellpad"],
                text=loc["Operator"] + "-" + loc["Wellpad"],
                textfont={"color": "black", "size": 16},
                textposition="bottom right",
                draggable=False,
            )
        )
    for ind, loc in df_consumer.iterrows():
        message = HTML()
        message.value = loc["Operator"] + "-" + loc["Wellpad"]
        popup = Popup(
            location=[loc["Latitude"] + lat_offset, loc["Longitude"]],
            child=message,
            close_button=False,
            auto_close=False,
            close_on_escape_key=False,
        )
        m.add_layer(popup)
        m.add_layer(
            Marker(
                icon=cons_icon,
                mode="markers+text",
                location=[loc["Latitude"], loc["Longitude"]],
                title=loc["Wellpad"],
                text=loc["Operator"] + "-" + loc["Wellpad"],
                textposition="right",
                draggable=False,
            )
        )

    legend = LegendControl(
        {
            "Supplier": "red",
            "Consumer": "blue",
            "Midstream": "gold",
            "Supplier-Transport": "orange",
        },
        title="Legend",
        position="bottomright",
    )
    m.add(legend)

    # return plot handle
    return m


##################################################
def NodalPricePlot(input_dir, MATCHES, requests, fig_name="plot_nodal_price"):
    """
    Creates an interactive plot for presentation in a Jupyter notebook
    """
    # Reload problem data
    df_producer, df_consumer, df_midstream, df_restrictions = fetch_case_data(requests)
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # Create numpy array of coordinate data:
    all_coords = pd.concat(
        [
            df_matches_supply[["Longitude", "Latitude"]],
            df_matches_demand[["Longitude", "Latitude"]],
        ],
        ignore_index=True,
    )
    min_lon = all_coords["Longitude"].min()
    max_lon = all_coords["Longitude"].max()
    min_lat = all_coords["Latitude"].min()
    max_lat = all_coords["Latitude"].max()
    coord_grid = np.mgrid[min_lon:max_lon:50j, min_lat:max_lat:50j]
    coord_flat = coord_grid.reshape(2, -1).T

    # Create numpy array of price data
    obs_coords = all_coords.to_numpy()
    obs_prices_df = pd.concat(
        [df_matches_supply[["Nodal Price"]], df_matches_demand[["Nodal Price"]]],
        ignore_index=True,
    )
    obs_prices = obs_prices_df.to_numpy()

    # generate RBF interpolation of observed data
    try:
        interp_flat = RBFInterpolator(
            obs_coords,
            obs_prices,
            kernel="thin_plate_spline",
            epsilon=1.5,
            smoothing=0.0025,
        )(coord_flat)
        interp_grid = interp_flat.reshape(
            50, 50
        )  # NOTE: must be the same as the grid spacing used for coord_grid
    except:
        print("Singular matrix - cannot interpolate data")
        return None

    # plot interpolated data
    vmin = obs_prices_df.min()
    vmax = obs_prices_df.max()
    fig, ax = plt.subplots(1, 1)
    fig.set_figheight(10)
    fig.set_figwidth(15)
    ax.pcolormesh(
        *coord_grid,
        interp_grid,
        vmin=vmin,
        vmax=vmax,
        shading="gouraud",
        cmap=mpl.colormaps["Spectral"],
    )
    p = ax.scatter(
        *obs_coords.T,
        c=obs_prices,
        s=50,
        ec="k",
        vmin=vmin,
        vmax=vmax,
        cmap=mpl.colormaps["Spectral"],
    )
    cbar = fig.colorbar(p)
    cbar.ax.set_xlabel("Price (USD/bbl)", rotation=0, labelpad=25)

    # get plot, save a copy, and return handle
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def ProfitPlot(input_dir, PROFITS, fig_name="plot_profit"):
    """
    Creates a plot of stakeholder profits
    """
    # write subplots for each of these:
    # model.p_ProducerProfit [pi]
    # model.p_ConsumerProfit [ci]
    # model.p_MidstreamTotalProfit [mi]
    # model.p_ProducerTotalTransportProfit [pi]

    (
        df_profits_supply,
        df_profits_demand,
        df_profits_producer_transport,
        df_profits_midstream_transport,
    ) = fetch_profit_data(PROFITS)

    # Create figure and axes handles
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(4, 1, 1)
    ax2 = plt.subplot(4, 1, 2)
    ax3 = plt.subplot(4, 1, 3)
    ax4 = plt.subplot(4, 1, 4)
    ax1.bar(
        x=df_profits_supply["Producer Index"].tolist(),
        height=df_profits_supply["Profit"].tolist(),
        color="red",
    )
    ax2.bar(
        x=df_profits_demand["Consumer Index"].tolist(),
        height=df_profits_demand["Profit"].tolist(),
        color="blue",
    )
    ax3.bar(
        x=df_profits_midstream_transport["Midstream Index"].tolist(),
        height=df_profits_midstream_transport["Profit"].tolist(),
        color="gold",
    )
    ax4.bar(
        x=df_profits_producer_transport["Producer Index"].tolist(),
        height=df_profits_producer_transport["Profit"].tolist(),
        color="orange",
    )
    ax1.set(xlabel="Producer ID", ylabel="Surplus [USD]")
    ax2.set(xlabel="Consumer ID", ylabel="Surplus [USD]")
    ax3.set(xlabel="Midstream ID", ylabel="Surplus [USD]")
    ax4.set(xlabel="Producer ID", ylabel="Surplus [USD]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    # save figure and return plot handle
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def ProfitPlotByOperator(input_dir, PROFITS, fig_name="plot_profit"):
    """
    Creates a plot of stakeholder profits
    """
    # write subplots for each of these:
    # model.p_ProducerProfit [pi]
    # model.p_ConsumerProfit [ci]
    # model.p_MidstreamTotalProfit [mi]
    # model.p_ProducerTotalTransportProfit [pi]

    (
        df_profits_supply,
        df_profits_demand,
        df_profits_producer_transport,
        df_profits_midstream_transport,
    ) = fetch_profit_data(PROFITS)

    # pivot to get totals
    supply_pivot = pd.pivot_table(
        df_profits_supply[["Operator", "Profit"]],
        index="Operator",
        values="Profit",
        aggfunc="sum",
    )
    demand_pivot = pd.pivot_table(
        df_profits_demand[["Operator", "Profit"]],
        index="Operator",
        values="Profit",
        aggfunc="sum",
    )
    midstream_pivot = pd.pivot_table(
        df_profits_midstream_transport[["Operator", "Profit"]],
        index="Operator",
        values="Profit",
        aggfunc="sum",
    )
    p_transport_pivot = pd.pivot_table(
        df_profits_producer_transport[["Operator", "Profit"]],
        index="Operator",
        values="Profit",
        aggfunc="sum",
    )

    # Create figure and axes handles
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 2, 5)
    ax4 = plt.subplot(3, 2, 6)
    ax1.bar(
        x=[ind for ind, row in supply_pivot.iterrows()],
        height=supply_pivot["Profit"].to_list(),
        color="red",
    )
    ax1.set_ylim(GetLims(supply_pivot["Profit"].to_list()))
    ax1.tick_params(axis="x", labelrotation=45)
    ax2.bar(
        x=[ind for ind, row in demand_pivot.iterrows()],
        height=demand_pivot["Profit"].to_list(),
        color="blue",
    )
    ax2.set_ylim(GetLims(demand_pivot["Profit"].to_list()))
    ax2.tick_params(axis="x", labelrotation=45)
    ax3.bar(
        x=[ind for ind, row in midstream_pivot.iterrows()],
        height=midstream_pivot["Profit"].tolist(),
        color="gold",
    )
    ax3.set_ylim(GetLims(midstream_pivot["Profit"].to_list()))
    ax3.tick_params(axis="x", labelrotation=45)
    ax4.bar(
        x=[ind for ind, row in p_transport_pivot.iterrows()],
        height=p_transport_pivot["Profit"].tolist(),
        color="orange",
    )
    ax4.set_ylim(GetLims(p_transport_pivot["Profit"].to_list()))
    ax4.tick_params(axis="x", labelrotation=45)
    ax1.set(xlabel="Producer ID", ylabel="Surplus [USD]")
    ax2.set(xlabel="Consumer ID", ylabel="Surplus [USD]")
    ax3.set(xlabel="Midstream ID", ylabel="Surplus [USD]")
    ax4.set(xlabel="Producer ID", ylabel="Surplus [USD]")
    ax1.set_title(f"Supplier Surplus", ha="left", x=-0)
    ax2.set_title(f"Consumer Profit", ha="left", x=-0)
    ax3.set_title(f"Midstream Profit", ha="left", x=-0)
    ax4.set_title(f"Supplier Transportation Profit", ha="left", x=-0)
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    # save figure and return plot handle
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def RevenuePlot(input_dir, MATCHES, fig_name="plot_revenue"):
    """
    Creates a plot of stakeholder revenue streams
    """
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # Create figure and axes handles
    fig = plt.figure(figsize=(6.4, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 2, 5)
    ax4 = plt.subplot(3, 2, 6)
    # NOTE: I am using the -1 in the below so that the resulting graph gives the expected visual interpretation of revenue, instead of the economic one
    ax1.bar(
        x=df_matches_supply["Supplier Index"].tolist(),
        height=(df_matches_supply["Nodal Price"] * df_matches_supply["Rate"]).to_list(),
        color="red",
    )
    ax2.bar(
        x=df_matches_demand["Consumer Index"].tolist(),
        height=(
            -1 * df_matches_demand["Nodal Price"] * df_matches_demand["Rate"]
        ).to_list(),
        color="blue",
    )
    if not df_matches_midstream_transport.empty:
        ax3.bar(
            x=df_matches_midstream_transport["Midstream Index"].tolist(),
            height=(
                df_matches_midstream_transport["Price"]
                * df_matches_midstream_transport["Rate"]
            ).tolist(),
            color="gold",
        )
    if not df_matches_producer_transport.empty:
        ax4.bar(
            x=df_matches_producer_transport["Carrier Index"].tolist(),
            height=(
                df_matches_producer_transport["Price"]
                * df_matches_producer_transport["Rate"]
            ).tolist(),
            color="orange",
        )
    ax1.set(xlabel="Producer ID", ylabel="Revenue [USD]")
    ax2.set(xlabel="Consumer ID", ylabel="Revenue [USD]")
    ax3.set(xlabel="Midstream ID", ylabel="Revenue [USD]")
    ax4.set(xlabel="Producer ID", ylabel="Revenue [USD]")
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    # save figure and return plot handle
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def RevenuePlotByOperator(input_dir, MATCHES, fig_name="plot_revenue"):
    """
    Creates a plot of stakeholder revenue streams
    """
    (
        df_matches_supply,
        df_matches_demand,
        df_matches_producer_transport,
        df_matches_midstream_transport,
    ) = fetch_match_data(MATCHES)

    # Add revenue to dataframes, then pivot to get totals
    df_matches_supply["Revenue"] = (
        df_matches_supply["Nodal Price"] * df_matches_supply["Rate"]
    )
    supply_pivot = pd.pivot_table(
        df_matches_supply[["Operator", "Revenue"]],
        index="Operator",
        values="Revenue",
        aggfunc="sum",
        fill_value=0,
        dropna=False,
    )

    df_matches_demand["Revenue"] = (
        df_matches_demand["Nodal Price"] * df_matches_demand["Rate"] * (-1)
    )  # NOTE: the negative one value here gives the expencted interpretation of revenue flow
    demand_pivot = pd.pivot_table(
        df_matches_demand[["Operator", "Revenue"]],
        index="Operator",
        values="Revenue",
        aggfunc="sum",
        fill_value=0,
        dropna=False,
    )

    if not df_matches_midstream_transport.empty:
        df_matches_midstream_transport["Revenue"] = (
            df_matches_midstream_transport["Price"]
            * df_matches_midstream_transport["Rate"]
        )
        midstream_pivot = pd.pivot_table(
            df_matches_midstream_transport[["Operator", "Revenue"]],
            index="Operator",
            values="Revenue",
            aggfunc="sum",
        )

    if not df_matches_producer_transport.empty:
        df_matches_producer_transport["Revenue"] = (
            df_matches_producer_transport["Price"]
            * df_matches_producer_transport["Rate"]
        )
        p_transport_pivot = pd.pivot_table(
            df_matches_producer_transport[["Operator", "Revenue"]],
            index="Operator",
            values="Revenue",
            aggfunc="sum",
        )

    # Create figure and axes handles
    fig = plt.figure(figsize=(8, 8))
    ax1 = plt.subplot(3, 1, 1)
    ax2 = plt.subplot(3, 1, 2)
    ax3 = plt.subplot(3, 2, 5)
    ax4 = plt.subplot(3, 2, 6)
    ax1.bar(
        x=[ind for ind, row in supply_pivot.iterrows()],
        height=supply_pivot["Revenue"].to_list(),
        color="red",
    )
    ax1.set_ylim(GetLims(supply_pivot["Revenue"].to_list()))
    ax1.tick_params(axis="x", labelrotation=45)
    ax2.bar(
        x=[ind for ind, row in demand_pivot.iterrows()],
        height=demand_pivot["Revenue"].to_list(),
        color="blue",
    )
    ax2.set_ylim(GetLims(demand_pivot["Revenue"].to_list()))
    ax2.tick_params(axis="x", labelrotation=45)
    if not df_matches_midstream_transport.empty:
        ax3.bar(
            x=[ind for ind, row in midstream_pivot.iterrows()],
            height=midstream_pivot["Revenue"].tolist(),
            color="gold",
        )
        ax3.set_ylim(GetLims(midstream_pivot["Revenue"].to_list()))
        ax3.tick_params(axis="x", labelrotation=45)
    if not df_matches_producer_transport.empty:
        ax4.bar(
            x=[ind for ind, row in p_transport_pivot.iterrows()],
            height=p_transport_pivot["Revenue"].tolist(),
            color="orange",
        )
        ax4.set_ylim(GetLims(p_transport_pivot["Revenue"].to_list()))
        ax4.tick_params(axis="x", labelrotation=45)
    ax1.set(xlabel="Producer ID", ylabel="Revenue [USD]")
    ax2.set(xlabel="Consumer ID", ylabel="Revenue [USD]")
    ax3.set(xlabel="Midstream ID", ylabel="Revenue [USD]")
    ax4.set(xlabel="Producer ID", ylabel="Revenue [USD]")
    ax1.set_title(f"Supplier Revenues", ha="left", x=-0)
    ax2.set_title(f"Consumer Revenues", ha="left", x=-0)
    ax3.set_title(f"Midstream Revenues", ha="left", x=-0)
    ax4.set_title(f"Supplier Transportation Revenues", ha="left", x=-0)
    plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1))

    # save figure and return plot handle
    plt.savefig(os.path.join(input_dir, fig_name))
    return fig


##################################################
def GetLims(data):
    # ax.set_ylim([ymin, ymax])
    ymin_data = min(data)
    ymax_data = max(data)
    if ymax_data >= 1:
        ymax = 1.05 * ymax_data
    else:
        ymax = 1.0
    if ymin_data <= -1:
        ymin = 1.05 * ymin_data
    else:
        ymin = -1.0
    return ymin, ymax
