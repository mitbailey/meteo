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
import matplotlib.colors as colors

import openmeteo_requests
import requests_cache
from retry_requests import retry
from xarray import Dataset

# Many things are currently hard-coded which shouldn't be. This is a work in progress.

def init_open_meteo():
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    creation_time = str(cache_session.get('https://api.open-meteo.com/v1/forecast').created_at)[:10] + 'T' + str(cache_session.get('https://api.open-meteo.com/v1/forecast').created_at)[11:19]
    # print(creation_time)
    # exit(1)

    return openmeteo, creation_time

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

def fetch_forecast(n_cells: int = 50):
    # TODO: Uncomment.
    openmeteo, creation_time = init_open_meteo()

    # %%
    # data = gpd.read_file('eclipse2024/center.shp')
    # %%
    # center_coords = data['geometry'].get_coordinates().to_numpy()

    data = gpd.read_file('eclipse2024/upath_hi.shp')
    upper_coords = data['geometry'].get_coordinates().to_numpy()

    # data = gpd.read_file('eclipse2024/upath_lo.shp')
    # lower_coords = data['geometry'].get_coordinates().to_numpy()

    # duration_data = gpd.read_file('eclipse2024/duration.shp')
    # duration_coords = data['geometry'].get_coordinates().to_numpy()


    #%%
    path = "./maps/tl_2023_us_state.shp"
    df = gpd.read_file(path)
    df = df.to_crs("EPSG:4326")

    ne_states = ['MA','NH', 'VT', 'RI', 'CT', 'NY', 'ME']
    ne = df[df.STUSPS.isin(ne_states)]

    # fig, ax = plt.subplots(figsize=(16, 10))

    # MIN_LAT = 40
    # MAX_LAT = 48
    # MIN_LON = -80
    # MAX_LON = -66

    # plt.xlim([MIN_LON, MAX_LON])
    # plt.ylim([MIN_LAT, MAX_LAT])

    AOI_MIN_LAT = 41
    AOI_MAX_LAT = 45.75
    AOI_MIN_LON = -77
    AOI_MAX_LON = -69.5

    AOI = gpd.GeoDataFrame(geometry=[Polygon([(AOI_MIN_LON, AOI_MIN_LAT), (AOI_MIN_LON, AOI_MAX_LAT), (AOI_MAX_LON, AOI_MAX_LAT), (AOI_MAX_LON, AOI_MIN_LAT)])], crs='EPSG:4326')

    # worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    # worldmap.plot(color='grey', alpha=0.5, ax=ax)

    # ne.plot(color='tab:blue', ax=ax)
    # ne.boundary.plot(color='black', ax=ax)

    # center_coords = [c for c in center_coords if ((c[0] > -85) and (c[0] < -67))]

    # plt.plot([c[0] for c in center_coords], [c[1] for c in center_coords], color='black', linewidth=2)
    # plt.plot([c[0] for c in center_coords], [c[1] for c in center_coords], color='goldenrod', linewidth=2, linestyle='--')
    # plt.plot([c[0] for c in upper_coords], [c[1] for c in upper_coords], color='tab:orange')
    # plt.plot([c[0] for c in lower_coords], [c[1] for c in lower_coords], color='tab:green')


    # duration_data.loc[[7], 'geometry'].plot(ax=ax, color='lightgrey', alpha=0.8)
    # duration_data.loc[[2], 'geometry'].plot(ax=ax, color='lightgrey', alpha=0.8)
    # duration_data.loc[[1], 'geometry'].plot(ax=ax, color='lightgrey', alpha=0.8)
    # # duration_data.loc[[7], 'geometry'].plot(ax=ax, color='olive')
    # # duration_data.loc[[2], 'geometry'].plot(ax=ax, color='darkkhaki')
    # # duration_data.loc[[1], 'geometry'].plot(ax=ax, color='gold')

    # duration_data.boundary.plot(ax=ax, color='white', linewidth=1.5)
    # duration_data.boundary.plot(ax=ax, color='k', linewidth=1, linestyle='--')

    # x_offset = 0
    # y_offset = 0.49
    # text = ax.annotate('210 s', weight='bold', xy=(-78.5+x_offset, 42.2+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    # text.set_rotation(30)
    # text = ax.annotate('180 s', weight='bold', xy=(-78.5+x_offset, 42.0+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    # text.set_rotation(30)
    # text = ax.annotate('150 s', weight='bold', xy=(-78.5+x_offset, 41.85+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    # text.set_rotation(30)
    # text = ax.annotate('120 s', weight='bold', xy=(-78.5+x_offset, 41.72+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    # text.set_rotation(30)

    # # ax.annotate('180s', xy=(-79.68, 42.04), xytext=(3, 3), textcoords="offset points", fontsize=20)
    # # ax.annotate('<=150s', xy=(-79.68, 41.75), xytext=(3, 3), textcoords="offset points", fontsize=20)
    # ax.annotate('White boundary indicates 30 second duration decrease.', xy=(-79.68, 40.2), xytext=(3, 3), textcoords="offset points", fontsize=8)

    # for i in range(7):
        # ax.annotate(i, xy=(-78, 41), xytext=(3, 3), textcoords="offset points", fontsize=24)

    # exit(1)

    # for idx, row in duration_data.iterrows():
    #     # duration_data.plot(color=duration_data['Duration'], ax=ax)
    #     X,Y = row['geometry'].exterior.coords.xy
    #     # print( list(X),  list(Y) )
    #     plt.plot(X, Y, color='white')
    #     if (idx > 3):
    #         break

    # for POLYGON in duration_data:
    #     duration_coords = POLYGON['geometry'].exterior.coords.xy
    #     plt.plot([c[0] for c in duration_coords], [c[1] for c in duration_coords], color='white')

    eclipse_geom = Polygon(upper_coords)
    eclipse = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[eclipse_geom]) 

    ne_eclipse = gpd.overlay(ne, eclipse, how='intersection')
    ne_eclipse = gpd.overlay(ne_eclipse, AOI, how='intersection')

    grid = create_grid(gdf=ne_eclipse, n_cells=n_cells, overlap=True)

    if (len(grid) > 500):
        print(f'Large number of datapoints ({len(grid)}). Proceed?')
        input()

    lat_lon_datapoints = [[cell.centroid.y, cell.centroid.x] for cell in grid['geometry']]
    
    # for lat_lon in lat_lon_datapoints:
    #     print(lat_lon)

    forecast_datapoints = [calculate_forecast(openmeteo, lat_lon) for lat_lon in lat_lon_datapoints]

    return forecast_datapoints, grid, ne, creation_time

def plot_forecast(forecast_datapoints, grid, ne, creation_time, HH=[15,'']):
    fig, ax = plt.subplots(figsize=(16, 10))
    
    MIN_LAT = 40
    MAX_LAT = 48
    MIN_LON = -80
    MAX_LON = -66

    plt.xlim([MIN_LON, MAX_LON])
    plt.ylim([MIN_LAT, MAX_LAT])

    data = gpd.read_file('eclipse2024/center.shp')
    center_coords = data['geometry'].get_coordinates().to_numpy()
    
    data = gpd.read_file('eclipse2024/upath_hi.shp')
    upper_coords = data['geometry'].get_coordinates().to_numpy()
    
    data = gpd.read_file('eclipse2024/upath_lo.shp')
    lower_coords = data['geometry'].get_coordinates().to_numpy()
    
    duration_data = gpd.read_file('eclipse2024/duration.shp')

    worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    worldmap.plot(color='grey', alpha=0.5, ax=ax)

    ne.plot(color='tab:blue', ax=ax)
    ne.boundary.plot(color='black', ax=ax)

    plt.plot([c[0] for c in center_coords], [c[1] for c in center_coords], color='black', linewidth=2)
    plt.plot([c[0] for c in center_coords], [c[1] for c in center_coords], color='goldenrod', linewidth=2, linestyle='--')
    plt.plot([c[0] for c in upper_coords], [c[1] for c in upper_coords], color='tab:orange')
    plt.plot([c[0] for c in lower_coords], [c[1] for c in lower_coords], color='tab:green')

    duration_data.loc[[7], 'geometry'].plot(ax=ax, color='lightgrey', alpha=0.8)
    duration_data.loc[[2], 'geometry'].plot(ax=ax, color='lightgrey', alpha=0.8)
    duration_data.loc[[1], 'geometry'].plot(ax=ax, color='lightgrey', alpha=0.8)
    
    duration_data.boundary.plot(ax=ax, color='white', linewidth=1.5)
    duration_data.boundary.plot(ax=ax, color='k', linewidth=1, linestyle='--')
    
    x_offset = 0
    y_offset = 0.49
    text = ax.annotate('210 s', weight='bold', xy=(-78.5+x_offset, 42.2+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    text.set_rotation(30)
    text = ax.annotate('180 s', weight='bold', xy=(-78.5+x_offset, 42.0+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    text.set_rotation(30)
    text = ax.annotate('150 s', weight='bold', xy=(-78.5+x_offset, 41.85+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    text.set_rotation(30)
    text = ax.annotate('120 s', weight='bold', xy=(-78.5+x_offset, 41.72+y_offset), xytext=(3, 3), textcoords="offset points", fontsize=8)
    text.set_rotation(30)
    
    ax.annotate('White boundary indicates 30 second duration decrease.', xy=(-79.68, 40.2), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax.annotate(f'{HH[1]}', xy=(-79, 47.2), xytext=(3, 3), textcoords="offset points", fontsize=16)

    overcast_datapoints = [forecast_datapoint['cloud_cover'].values[HH[0]] for forecast_datapoint in forecast_datapoints]
    grid['value'] = overcast_datapoints
    datetimes = [forecast_datapoint['date'].values[HH[0]] for forecast_datapoint in forecast_datapoints]

    for i, _ in enumerate(overcast_datapoints):
        print(f'{datetimes[i]}: {overcast_datapoints[i]}%')

    grid.plot(fc="none", ec='black',ax=ax)
    norm=plt.Normalize(vmin=0, vmax=100)
    cbar=plt.cm.ScalarMappable(norm=norm, cmap='Reds')
    grid.plot(column='value', ec='none', lw=0.2, legend=False, cmap='Reds', alpha=0.9, ax=ax, vmin=0, vmax=100)

    plt.suptitle(f'Cloud Cover Forecast for {HH[0]}:00-{HH[0]}:59 EDT ({HH[1]}) on 2024-04-08', fontsize=16)
    ax.set_title(f'Forecast fetch date: {creation_time}Z')
    ax.set_xlabel('Longitude [degrees]')
    ax.set_ylabel('Latitude [degrees]')

    ax_cbar = fig.colorbar(cbar, ax=ax)
    ax_cbar.set_label('Total Cloud Cover [%]')

    # Save fig
    creation_time = creation_time.replace(':', '')
    creation_time = creation_time.replace('-', '')
    plt.savefig(f'frames/cloud_cover_{creation_time}Z_{HH[0]}_{HH[1]}.png')

if __name__ == '__main__':
    hours_of_interest = [[13,'Pre-Eclipse'], [14,'Partial Begins'], [15,'Totality'], [16,'Partial Ends'], [17,'Post-Eclipse']]
    # hours_of_interest = range(6,18)

    forecast_datapoints, grid, ne, creation_time = fetch_forecast(n_cells=50)
    for hour in hours_of_interest:
        plot_forecast(forecast_datapoints, grid, ne, creation_time, HH=hour)
    plt.show()

    # data = gpd.read_file('eclipse2024/ppath.shp')
    # print(data)