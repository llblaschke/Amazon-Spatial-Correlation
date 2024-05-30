import cartopy.feature as cf
import cartopy.crs as ccrs
from matplotlib.colors import from_levels_and_colors, ListedColormap, LinearSegmentedColormap
from cycler import cycler
import string

from model_params_functions import *

# COLORS
color_land    = cf.COLORS['land']
cmap_white, _ = from_levels_and_colors([0,1], ["oldlace"])
cmap_nan, _   = from_levels_and_colors([0,1], ["grey"])
cmap_trends   = "PRGn_r"

c_dark_green   = '#00441b'
c_light_green  = '#a5da9f'
c_light_violet = '#c1a4ce'
c_dark_violet  = '#40004b'
c_orange = '#fd9804'
c_white  = '#bcbcbc'


cmap_eblf, _ = from_levels_and_colors([0,1], [c_dark_green])
cmap_hlu,  _ = from_levels_and_colors([0,1], ["wheat"])
cmap_defo, _ = from_levels_and_colors([0,1], [c_orange])
cmap_dh, _   = from_levels_and_colors([0,1], [c_dark_violet])

cmap_sum = ListedColormap([c_light_green, "w", c_orange, c_dark_violet])

# MAPS
map_proj       = ccrs.LambertConformal(central_longitude=-60, central_latitude=-10)
map_extent     = [-80, -45, -18, 9]
transform_data = ccrs.PlateCarree()
aspect         = 140/107
subplot_kws = {'projection':map_proj}

bbox_props = dict(boxstyle='round', facecolor='wheat', alpha=0.9)


# SIMULATION



# colors for plots of models
clist = ["darkslateblue", "#a29b98", c_light_green]
cmap = LinearSegmentedColormap.from_list("yellow_blue", clist)
cell_color_list = [c_dark_violet] + [c for c in cmap(np.linspace(0,1,ncells-2))] + [c_dark_green]
cell_color_cmap = LinearSegmentedColormap.from_list("yellow_blue", clist)
cell_color_cycler = cycler(color=cell_color_list)



def label_axes2(axes, labels=None, loc=None, **kwargs):
    if labels is None:
        labels = string.ascii_lowercase

    # re-use labels rather than stop labeling
    if loc is None:
        loc = (.9, .9)
    for ax, lab in zip(axes, labels):
        ax.annotate(lab+")", xy=loc,
                    xycoords='axes fraction',
                    weight='bold',
                    **kwargs)