# Geospatial

A place to learn about geospatial analysis.

This repository is a working notebook for exploring how to represent, analyze, and
visualize data that has a location attached to it. Expect notes, small experiments,
and worked examples rather than a polished library.

## What geospatial analysis is

Geospatial analysis is the study of data tied to positions on the Earth — points,
lines, and areas — and of the relationships between them. Typical questions it
answers:

- What is near what? (proximity, nearest neighbors)
- What is inside what? (spatial joins, point-in-polygon)
- How does a value change across space? (interpolation, density, hot spots)
- How does a place change over time? (change detection, time series by region)

## Topics to cover

- **Data models** — vector (points, lines, polygons) vs. raster (grids, imagery)
- **Coordinate reference systems** — latitude/longitude, projections, and why the
  choice of CRS changes your distance and area measurements
- **File formats** — GeoJSON, Shapefile, GeoPackage, GeoTIFF, Parquet/GeoParquet
- **Core operations** — buffers, intersections, spatial joins, aggregation
- **Analysis** — spatial statistics, clustering, routing, terrain
- **Visualization** — static maps, web maps, and choropleths

## Tools worth knowing

| Tool | Use |
| --- | --- |
| [GeoPandas](https://geopandas.org/) | Vector data in Python, pandas-style |
| [Shapely](https://shapely.readthedocs.io/) | Geometry operations |
| [Rasterio](https://rasterio.readthedocs.io/) | Reading and writing raster data |
| [PostGIS](https://postgis.net/) | Spatial queries in PostgreSQL |
| [QGIS](https://qgis.org/) | Desktop GIS for inspecting and editing data |
| [GDAL/OGR](https://gdal.org/) | Format conversion and the engine behind most of the above |

## Launch the labs

[![Open Alpha Earth embeddings demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webpug/geospatial/blob/main/alpha_earth_embeddings_demo.ipynb)

The badge opens the **public Alpha Earth embeddings demo template**. Before entering personal names or
coordinates, choose **File → Save a copy in Drive**. In that copy, open **Share**,
set **General access → Restricted**, and **check the collaborators**, removing
anyone who should not have access. A launch link cannot enforce private permissions.
**Never save personal data back to GitHub**, including source, outputs, or widget
state. Do not use **Save a copy in GitHub** for your personal copy.

Run the setup cells in order. Enter your own Earth Engine project ID in the blank
`PROJECT_ID` field; empty or whitespace-only values are rejected before authentication.
The initial PEEK point, Step 4 query, and first public place example use downtown
Seattle (47.6062, -122.3321). In Step 3, run the places editor and enter a name,
latitude, and longitude, then click **Add / update**. Reusing a name updates that
place. Select a place to edit it or use **Remove selected**; **Reset defaults**
restores the eight public examples. Run **which places are twins?** after editing
to compare the current list (at least two places). Re-running either cell preserves
your edits; the comparison does not reset the list.

In Step 4, run the query editor, enter latitude and longitude, and click **Apply
query**. Then run the map and histogram cells. Rerunning the editor or map preserves
the applied point; editing a field alone does not apply it or call Earth Engine.

The optional Google Maps search cell accepts a city, address, business, or natural
language description. Enable billing and **Places API (New)** in your Google Cloud
project, restrict the key to that API, and add `GOOGLE_MAPS_API_KEY` through Colab's
**Secrets (key icon)** with **notebook access** enabled. Only **Search** reads the
secret and sends the query to Google; requests may incur charges. Select a result
and click **Add selected to PLACES** or **Use for Step4**, then run the relevant
analysis cells. Manual entry works without a key. Search state stays in the kernel;
the controls do not write files or Drive data.

Values stay in the Python kernel until you restart the runtime (or disconnect and
delete it). Removing entries or resetting defaults does not erase earlier results.
Outputs, including widget displays and charts, can contain personal data. The
template uses `omit_outputs=true` and has no saved outputs or widget state, but
this is not a privacy guarantee. Clear outputs and restart when finished, and
inspect the notebook before sharing. The editor makes no Earth Engine requests;
running the comparison sends coordinates and place names to Google Earth Engine.

Other notebooks:

- [Portal lab — open in Colab](https://colab.research.google.com/github/webpug/geospatial/blob/main/portal_lab.ipynb)
- [Original standout lab — open in Colab](https://colab.research.google.com/github/webpug/geospatial/blob/main/standout_systems_labelling_lab.ipynb)

Use the same Drive-copy and sharing checks before entering personal data in any lab.

Offline checks: with `ipywidgets`, `nbformat`, `ipython`, `numpy`, `matplotlib`, and
`requests` installed, run `python tests/check_notebooks.py`. The checks compile all
notebook cells and check project validation, startup defaults, editors, and mocked
Places responses without live Places or Earth Engine requests.
