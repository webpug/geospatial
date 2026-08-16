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

## Layout

Nothing yet — this is the first commit. Notes and examples will be added as
directories per topic.
