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
project and restrict the key to that API. **Preferred:** add `GOOGLE_MAPS_API_KEY`
through Colab's **Secrets (key icon)** in the left sidebar with **notebook access**
enabled. Alternatively, run **Optional: set a runtime Google Maps API key**
immediately before the search cell and paste the key into its masked prompt.
Keep the key out of notebook code. Search prefers the runtime key, then Secrets.
Rerun key setup and submit a **blank** prompt to clear the runtime key and use
Secrets again. Cancellation also clears it. If hidden input is unavailable, setup
stops before any visible-input fallback; use Secrets instead.

The runtime key stays in Python memory until reset through that cell, runtime
restart, or disconnecting **and deleting** the runtime; closing the browser alone
does not clear it. Setup does not print the key or store it in a widget, file, or
notebook source. Only **Search** reads Secrets (when no runtime key is set) and
sends the key and query to Google; requests may incur charges. Select a result
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
- [Original standout lab — open in Colab](https://colab.research.google.com/github/webpug/geospatial/blob/main/external_inspirations/standout_systems_labelling_lab.ipynb)

External inspiration: **Teodora Szasz, PhD** authored the Standout Systems notebook
**“You Are Hand-Labelling Data Google Already Described for Free”**, preserved in
[external_inspirations/](external_inspirations/standout_systems_labelling_lab.ipynb).
Our Alpha Earth demo is adapted from her work, with added interactive controls,
Seattle defaults, and Google Maps search. Read her work on
[Substack](https://teodoracoach.substack.com/) and [teodora.coach](https://teodora.coach/).

Use the same Drive-copy and sharing checks before entering personal data in any lab.

Offline checks: with `ipywidgets`, `nbformat`, `ipython`, `numpy`, `matplotlib`, and
`requests` installed, run `python tests/check_notebooks.py`. The checks compile all
notebook cells and check project validation, startup defaults, editors, and mocked
Places responses without live Places or Earth Engine requests.

## OLMoEarth and comparison labs

[![Open OLMoEarth demo in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webpug/geospatial/blob/main/olmo_earth_embeddings_demo.ipynb)
[![Open OLMoEarth vs AlphaEarth in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webpug/geospatial/blob/main/olmo_vs_alpha_earth_comparison.ipynb)

- **[OLMoEarth embeddings demo](olmo_earth_embeddings_demo.ipynb):** acquire real quarterly
  Sentinel-2 imagery, load pretrained V1 Nano (or Tiny/Base), and explore RGB,
  embedding PCA, unsupervised groups, interactive spatial cosine queries and editable
  neighborhood rankings.
- **[OLMoEarth vs AlphaEarth](olmo_vs_alpha_earth_comparison.ipynb):** compare independent
  PCA panels, cosine-to-the-same-query maps, place matrices, neighbor rankings and
  guarded Spearman agreement on matching spatial support.

Both notebooks are standalone public templates. **Save a copy in Drive, restrict
sharing, and inspect collaborators before personal use**, as described above.
Outputs are cleared, `omit_outputs=true` is set, and no widget state is committed.
Those settings cannot guarantee privacy. Clear outputs and delete the runtime when
finished; never save personal entries or credentials back to GitHub.

Requirements: Python **3.11+**, PyTorch **2.7+** (normally supplied by Colab), internet
for package/weight downloads, an Earth Engine account, and **your own project ID**
in the initially blank field. GPU is recommended; CPU is supported and slower.
The minimal inference package is pinned to
[`96eeae43032d50dfc637d1200819b3966994392b`](https://github.com/allenai/olmoearth_pretrain_minimal/tree/96eeae43032d50dfc637d1200819b3966994392b).
V1 Nano is a lightweight stable V1 choice, **not the newest family or an accuracy
ceiling**. Downloads must succeed; there is no random-weight or fake-data fallback.

Run in order: install → embedded helpers → controls → authenticate → load weights →
acquire chip → panels/query → acquire PLACES → rankings. Four public defaults cover
Seattle urban, Discovery Park, Seattle port and Tiger Mountain forest. Add/update,
remove or reset up to eight places; choose 32, 64 (default) or 128 input pixels.
The default year is 2024. Optional Google Maps search uses the same masked runtime
key / Colab Secrets pattern as the AlphaEarth demo. Search can populate PLACES or
the chip center, while the spatial query later selects a cell *within* that chip.
Editing controls never contacts Earth Engine. Explicit acquisition runs sequential
requests; failed acquisition clears current results, and changed inputs require a
new acquisition. Old displayed plots remain snapshots. Imagery is not cached; only
selected model weights are reused in memory.

The data path uses explicit north-up 10 m UTM grids and `computePixels` structured
NumPy downloads with accompanying validity bands. Sentinel-2 L2A raw DN enters the
official normalizer without dividing by 10000. SCL classes 4, 5, 6 and 7 are retained
for four quarterly medians. Dates on the 15th of February, May, August and November
are **representative composite dates, not scene dates**. Missing inputs use the
minimal package's 12-band MISSING mask. Output OLMo cells require every input pixel,
band and quarter to be valid. Printed provenance includes coverage, collections,
windows, timestamps, band order, CRS/affine transform, model ID, code SHA,
aggregation and measured runtime. No cloud bucket is required.

The comparison matches AOI, year and exact grid, then averages AlphaEarth's valid
10 m vectors over each 4×4 footprint and normalizes them. Both models use the
intersection of valid 40 m cells; Alpha block coverage fractions remain inspectable.
OLMo's contextual receptive field extends beyond the nominal token footprint.
**OLMo receives four quarterly Sentinel-2-only composites; AlphaEarth is an annual
multimodal product.** Inputs, model size and temporal aggregation differ. PCA colors
are fitted independently and cannot be compared across models; all cosine scores
stay within one model. The matrix difference is a diagnostic, not accuracy.
Rank agreement has too-few/constant guards. There is no dimension mapping,
cross-model vector cosine, ground-truth evaluation or universal winner claim.
Runtime is labeled **end-to-end pipeline**, with model loading reported separately;
it is not an apples-to-apples inference or training benchmark.

Credit: the **Ai2 OLMoEarth team** ([official project](https://github.com/allenai/olmoearth_pretrain),
[minimal package](https://github.com/allenai/olmoearth_pretrain_minimal),
[quickstart](https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Inference-Quickstart.md));
**Ivan Zvonkov**, whose [official generation notebook](https://github.com/allenai/olmoearth_pretrain/blob/main/scripts/tools/generate_embeddings.ipynb)
inspired this workflow; **Google / Google DeepMind** for
[AlphaEarth Satellite Embedding V1](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL);
and **Teodora Szasz** for the adapted lab experience and original notebook linked
above. OLMoEarth weights are governed by the
[OLMoEarth Artifact License](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE),
not a blanket MIT license. Consult the linked dataset catalogs for data terms.

### Maintaining and checking the new notebooks

Edit [notebook_sources/](notebook_sources/) and
[scripts/build_earth_notebooks.py](scripts/build_earth_notebooks.py), then run:

```bash
python scripts/build_earth_notebooks.py
python scripts/build_earth_notebooks.py --check
python tests/check_earth_notebooks.py
python tests/check_notebooks.py
```

The generator embeds identical helper source into both notebooks; notebooks never
fetch these helpers or import another notebook at runtime. The new offline suite
requires the minimal package and notebook dependencies. It executes the actual
normalizer and embedded helpers, checks shapes, timestamps, masks, band order, grid
orientation, valid-only aggregation, shared support, within-model comparisons,
privacy controls and stale-result handling. Explicitly labeled test doubles check
Earth Engine requests and encoder contracts without fabricating notebook outputs.
The existing AlphaEarth, portal and original notebooks and their behavior are
preserved. **Live authenticated Earth Engine and hosted Colab execution have not
been verified here**; offline tests do not establish live dataset availability or
successful pretrained weight downloads.
