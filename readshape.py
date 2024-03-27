# %%
import geopandas as gpd
# %%
fp = 'eclipse2024/center.shp'

data = gpd.read_file(fp)
# %%
data = gpd.read_file('eclipse2024/duration.shp')
# %%
