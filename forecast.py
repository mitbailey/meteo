import openmeteo_requests

import requests_cache
import pandas as pd
from retry_requests import retry

locations = {
    # All east to west.
    # Along path middle.
    "fondaVT": [44.8714, -73.0976],
    "plattsburghNY": [44.6995, -73.4529],
    "watertownNY": [43.9748, -75.9108],
    "rochesterNY": [43.1548, -77.6156],
    "clevelandOH": [41.4995, -81.6954],
    "wapakonetaOH": [40.5675, -84.1936],

    # South of path.
    "burlingtonVT": [44.4759, -73.2121],
    "northhudsonNY": [43.9526, -73.7285],
    "lyonsfallNY": [43.6253, -75.3671],
    "syracuseNY": [43.0481, -76.1474],
    "jamestownNY": [42.097, -79.2353],
    "akronOH": [41.0814, -81.519],

    # North of path.
    "champlainNY": [44.9864, -73.4465],
    "potsdamNY": [44.6698, -74.9813],
    "toledoNY": [41.6639, -83.5552],
}

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

for location, coordinates in locations.items():
    # print(f"Location: {location}")
    # print(f"Coordinates: {coordinates}")


    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/forecast"
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
    print(f"LOCATION: {location}")
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()
    hourly_precipitation = hourly.Variables(0).ValuesAsNumpy()
    hourly_cloud_cover = hourly.Variables(1).ValuesAsNumpy()
    hourly_cloud_cover_low = hourly.Variables(2).ValuesAsNumpy()
    hourly_cloud_cover_mid = hourly.Variables(3).ValuesAsNumpy()
    hourly_cloud_cover_high = hourly.Variables(4).ValuesAsNumpy()
    hourly_visibility = hourly.Variables(5).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s", utc = True),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s", utc = True),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["precipitation"] = hourly_precipitation
    hourly_data["cloud_cover"] = hourly_cloud_cover
    hourly_data["cloud_cover_low"] = hourly_cloud_cover_low
    hourly_data["cloud_cover_mid"] = hourly_cloud_cover_mid
    hourly_data["cloud_cover_high"] = hourly_cloud_cover_high
    hourly_data["visibility"] = hourly_visibility

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    print(hourly_dataframe)


