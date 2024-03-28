# %%
from __future__ import annotations
from typing import SupportsFloat as Numeric, Tuple

import pandas as pd
import geopandas as gpd
import shapely
from shapely import Point
from shapely.geometry import Polygon

import numpy as np
import matplotlib.pyplot as plt

import openmeteo_requests
import requests_cache
from retry_requests import retry
from xarray import Dataset

def init_open_meteo():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    return openmeteo

def calculate_forecast(openmeteo, coordinates: Tuple[Numeric, Numeric], location: str = '') -> Dataset:
    """## Calculate weather forecast for a location using Open-Meteo API.

    ### Args:
        - `coordinates (Tuple[Numeric, Numeric])`: Latitude and longitude of the location (in decimal degrees)
        - `location (str, optional)`: Name of the location. Defaults to ''.

    ### Returns:
        - `Dataset`: Weather forecast data for the location.
    """
    # print(f"Location: {location}")
    # print(f"Coordinates: {coordinates}")

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
    print(f"Calculating forecast for Lat: {coordinates[0]}, Lon: {coordinates[1]}")
    params = {
        "latitude": coordinates[0],
        "longitude": coordinates[1],
        "hourly": ["precipitation", "cloud_cover", "cloud_cover_low", "cloud_cover_mid", "cloud_cover_high", "visibility"],
        "wind_speed_unit": "ms",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
        "start_date": "2024-04-08",
        "end_date": "2024-04-08"
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    # print(f"LOCATION: {location}")
    # print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    # print(f"Elevation {response.Elevation()} m asl")
    # print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    # print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover_mid = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(4).ValuesAsNumpy()
    hourly_visibility = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {}
    hourly_data["precipitation"] = (
        ('date',), hourly_precipitation, {'units': 'inch'})
    hourly_data["cloud_cover"] = (
        ('date',), hourly_cloud_cover, {'units': 'percent'})
    hourly_data["cloud_cover_low"] = (('date',), hourly_cloud_cover_low, {
                                      'units': 'percent', 'long_name': 'Low cloud cover'})
    hourly_data["cloud_cover_mid"] = (('date',), hourly_cloud_cover_mid, {
                                      'units': 'percent', 'long_name': 'Mid cloud cover'})
    hourly_data["cloud_cover_high"] = (('date',), hourly_cloud_cover_high, {
                                       'units': 'percent', 'long_name': 'High cloud cover'})
    hourly_data["visibility"] = (
        ('date',), hourly_visibility, {'units': 'miles'})
    # hourly_data['location'] = location
    # hourly_data['latitude'] = coordinates[0]
    # hourly_data['longitude'] = coordinates[1]

    # hourly_dataframe = pd.DataFrame(data=hourly_data)
    dates = pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive='left'
    )
    dates = list(map(lambda x: x.to_pydatetime(), dates))
    ds = Dataset(data_vars=hourly_data, coords={'date': (('date', ), dates)})
    ds.attrs['location'] = location
    ds.attrs['latitude'] = coordinates[0]
    ds.attrs['longitude'] = coordinates[1]

    return ds

#%%
def create_grid(gdf=None, bounds=None, n_cells=10, overlap=False, crs="EPSG:4326"):
    if bounds != None:
        xmin, ymin, xmax, ymax= bounds
    else:
        xmin, ymin, xmax, ymax= gdf.total_bounds

    centerpoints = []

    # get cell size
    cell_size = (xmax-xmin)/n_cells
    # create the cells in a loop
    grid_cells = []
    for x0 in np.arange(xmin, xmax+cell_size, cell_size ):
        for y0 in np.arange(ymin, ymax+cell_size, cell_size):
            x1 = x0-cell_size
            y1 = y0+cell_size
            poly = shapely.geometry.box(x0, y0, x1, y1)
            grid_cells.append( poly )

    cells = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
    
    if overlap == True:
        cols = ['grid_id','geometry','grid_area']
        cells = cells.sjoin(gdf, how='inner').drop_duplicates('geometry')

    # centerpoints = [[cell.centroid.x, cell.centroid.y] for cell in cells['geometry']]

    print('Number of cells:', len(cells))
    # return cells, centerpoints
    return cells

def run_model(HH, n_cells: int = 50):
    # TODO: Uncomment.
    openmeteo = init_open_meteo()

    # %%
    data = gpd.read_file('eclipse2024/center.shp')
    # %%
    center_coords = data['geometry'].get_coordinates().to_numpy()

    data = gpd.read_file('eclipse2024/upath_hi.shp')
    upper_coords = data['geometry'].get_coordinates().to_numpy()

    data = gpd.read_file('eclipse2024/upath_lo.shp')
    lower_coords = data['geometry'].get_coordinates().to_numpy()

    #%%
    path = "./maps/tl_2023_us_state.shp"
    df = gpd.read_file(path)
    df = df.to_crs("EPSG:4326")

    ne_states = ['MA','NH', 'VT', 'RI', 'CT', 'NY', 'PA', 'OH', 'ME']
    ne = df[df.STUSPS.isin(ne_states)]

    fig, ax = plt.subplots(figsize=(16, 10))

    plt.xlim([-86, -66])
    plt.ylim([37.5, 49])

    worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    worldmap.plot(color='grey', alpha=0.5, ax=ax)

    ne.plot(color='tab:blue', ax=ax)
    ne.boundary.plot(color='black', ax=ax)

    center_coords = [c for c in center_coords if ((c[0] > -85) and (c[0] < -67))]

    plt.plot([c[0] for c in center_coords], [c[1] for c in center_coords], color='tab:red')
    plt.plot([c[0] for c in upper_coords], [c[1] for c in upper_coords], color='tab:orange')
    plt.plot([c[0] for c in lower_coords], [c[1] for c in lower_coords], color='tab:green')

    eclipse_geom = Polygon(upper_coords)
    eclipse = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[eclipse_geom]) 

    ne_eclipse = gpd.overlay(ne, eclipse, how='intersection')

    grid = create_grid(gdf=ne_eclipse, n_cells=n_cells, overlap=True)

    if (len(grid) > 500):
        print('Too many data points! Exiting...')
        exit(1)

    lat_lon_datapoints = [[cell.centroid.y, cell.centroid.x] for cell in grid['geometry']]
    
    # for lat_lon in lat_lon_datapoints:
    #     print(lat_lon)

    forecast_datapoints = [calculate_forecast(openmeteo, lat_lon) for lat_lon in lat_lon_datapoints]
    overcast_datapoints = [forecast_datapoint['cloud_cover'].values[HH] for forecast_datapoint in forecast_datapoints]
    grid['value'] = overcast_datapoints
    # grid['value'] = grid.apply(lambda x: np.random.normal(50, 15),1)

    # print(forecast_datapoints)
    # print(overcast_datapoints)
    # print([forecast_datapoint['date'].values[HH] for forecast_datapoint in forecast_datapoints])

    grid.plot(fc="none", ec='black',ax=ax)
    grid.plot(column='value', ec='none', lw=0.2, legend=True, cmap='Reds', alpha=0.9, ax=ax)

    plt.scatter([c[1] for c in lat_lon_datapoints], [c[0] for c in lat_lon_datapoints], color='black', s=3, marker='x')
    # print(centerpoints)

    plt.show()

if __name__ == '__main__':
    hour_of_interest = 15
    run_model(hour_of_interest, n_cells=50)