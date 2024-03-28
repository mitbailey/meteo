# %%
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import shapely
from shapely import Point
from shapely.geometry import Polygon

# %%
data = gpd.read_file('eclipse2024/center.shp')
# %%
print(data['geometry'])
print(type(data))
print(data['geometry'].get_coordinates().to_numpy())
# print([list(data['geometry'].coords) for row_id in range(len(data['geometry']))])
center_coords = data['geometry'].get_coordinates().to_numpy()

data = gpd.read_file('eclipse2024/upath_hi.shp')
upper_coords = data['geometry'].get_coordinates().to_numpy()

data = gpd.read_file('eclipse2024/upath_lo.shp')
lower_coords = data['geometry'].get_coordinates().to_numpy()

print(gpd.datasets.available)

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

# r = 100
# count = 0

# CCc = [c for c in center_coords[::r] if Point(c[0], c[1]).within(ne.unary_union)]
# count += len(CCc)
# plt.scatter([c[0] for c in CCc], [c[1] for c in CCc], color='black', s=3)

# n = 3
# for y in range(1, n, 1):
#     dy = y/n

#     CCp = [[c[0], c[1]+dy] for c in center_coords[::r]]
#     CCm = [[c[0], c[1]-dy] for c in center_coords[::r]]

#     count += len(CCp) + len(CCm)

#     plt.scatter([c[0] for c in CCp], [c[1] for c in CCp], color='tab:red', s=2)
#     plt.scatter([c[0] for c in CCm], [c[1] for c in CCm], color='purple', s=2)

# point1 = Point(-75, 42)
# print(point1.within(ne.unary_union))

# print('Total number of points:', count)

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

    centerpoints = [[cell.centroid.x, cell.centroid.y] for cell in cells['geometry']]

    print('Number of cells:', len(cells))
    return cells, centerpoints

# e_max = 0.9
# center_coords_shifted_up = [[c[0], c[1]+e_max] for c in center_coords]
# center_coords_shifted_down = [[c[0], c[1]-e_max] for c in center_coords]
# poly_center_coords = center_coords_shifted_up + center_coords_shifted_down[::-1]

# eclipse_geom = Polygon(poly_center_coords)
eclipse_geom = Polygon(upper_coords)
eclipse = gpd.GeoDataFrame(index=[0], crs='EPSG:4326', geometry=[eclipse_geom]) 

ne_eclipse = gpd.overlay(ne, eclipse, how='intersection')

grid, centerpoints = create_grid(gdf=ne_eclipse, n_cells=50, overlap=True)
# plot centers

grid['value'] = grid.apply(lambda x: np.random.normal(10),1)
grid.plot(fc="none", ec='black',ax=ax)
grid.plot(column='value', ec='none', lw=0.2, legend=True, cmap='Reds', alpha=0.9, ax=ax)

plt.scatter([c[0] for c in centerpoints], [c[1] for c in centerpoints], color='black', s=3, marker='x')

plt.show()