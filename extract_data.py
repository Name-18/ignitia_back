import ee
import pandas as pd
from datetime import datetime

# Initialize Earth Engine
ee.Initialize(project='sanguine-anthem-465209-d9')

def extract_point_data(latitude, longitude, year):
    point = ee.Geometry.Point([longitude, latitude])
    start_date = f'{year}-01-01'
    end_date = f'{year}-12-31'

    # Elevation, Terrain, Land Cover
    elevation = ee.Image('USGS/SRTMGL1_003').select('elevation')
    terrain = ee.Algorithms.Terrain(elevation)
    land_cover = ee.Image('ESA/WorldCover/v100/2020').select('Map')

    # NDVI
    ndvi = ee.ImageCollection('MODIS/061/MOD13Q1') \
        .filterDate(start_date, end_date).filterBounds(point) \
        .select('NDVI').mean().multiply(0.0001).rename('ndvi')

    # LST
    lst = ee.ImageCollection('MODIS/061/MOD11A1') \
        .filterDate(start_date, end_date).filterBounds(point) \
        .select('LST_Day_1km').mean().multiply(0.02).subtract(273.15).rename('lst')

    # ERA5 (Humidity & Wind)
    era5 = ee.ImageCollection('ECMWF/ERA5_LAND/DAILY_AGGR') \
        .filterDate(start_date, end_date).filterBounds(point).mean()

    dew = era5.select('dewpoint_temperature_2m').subtract(273.15)
    temp = era5.select('temperature_2m').subtract(273.15)

    # RH computation
    rh = dew.subtract(temp).multiply(17.625).divide(temp.add(243.04)) \
        .exp().multiply(6.112).multiply(100).divide(
            temp.multiply(17.625).divide(temp.add(243.04)).exp().multiply(6.112)
        ).rename('relative_humidity')

    # Wind speed
    wind = era5.select('u_component_of_wind_10m').pow(2).add(
        era5.select('v_component_of_wind_10m').pow(2)
    ).sqrt().rename('wind_speed')

    # Merge all images into one
    combined = elevation.addBands(terrain) \
        .addBands(land_cover) \
        .addBands(ndvi) \
        .addBands(lst) \
        .addBands(rh) \
        .addBands(wind)

    # Perform a single reduceRegion call
    stats = combined.reduceRegion(
        reducer=ee.Reducer.first(),
        geometry=point,
        scale=250,  # Use a moderate scale for speed/accuracy balance
        maxPixels=1e9
    ).getInfo()

    # Build result
    result = {
        'lat': latitude,
        'lon': longitude,
        'elevation': stats.get('elevation'),
        'slope': stats.get('slope'),
        'aspect': stats.get('aspect'),
        'land_cover_type': stats.get('Map'),
        'ndvi': stats.get('ndvi'),
        'lst': stats.get('lst'),
        'relative_humidity': round(max(0, min(100, stats.get('relative_humidity', 0))), 2) if stats.get('relative_humidity') is not None else None,
        'wind_speed': stats.get('wind_speed'),
        'month': datetime.now().month
    }

    return result



# === RUN AND SAVE ===

    # Saves in current dir

    # print("✅ CSV saved as point_data_output.csv in current directory.")
import sys

# Take args from command line
if __name__ == "__main__":
    lat = float(sys.argv[1])
    lon = float(sys.argv[2])
    year = int(sys.argv[3])

    data = extract_point_data(lat, lon, year)

    # Print as JSON string to return to JS
    # data = extract_point_data(lat, lon, year)

    df = pd.DataFrame([data])
    df.to_csv("point_data_output.csv", index=False)  
    import json
    print(json.dumps(data))
