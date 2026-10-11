"""Regenerate standalone notebooks; helper source is embedded, never fetched at runtime."""
from pathlib import Path
import argparse
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SHA = '96eeae43032d50dfc637d1200819b3966994392b'


def markdown(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip(), execution_count=None, outputs=[])


def embedded(filename):
    return code(f'# Embedded source: notebook_sources/{filename}\n' + (ROOT/'notebook_sources'/filename).read_text())


def build(comparison=False):
    filename = 'olmo_vs_alpha_earth_comparison.ipynb' if comparison else 'olmo_earth_embeddings_demo.ipynb'
    title = 'OLMoEarth × AlphaEarth · same places, two representations' if comparison else 'OLMoEarth · a neighborhood in learned features'
    intro = ('Compare representations on the **same AOI, calendar year and exact UTM grid**. '
             'Inspect independent PCA views, query the same 40 m cell, and compare neighborhood rankings.\n\n'
             '**This is not an accuracy contest.** OLMoEarth sees four quarterly **Sentinel-2-only** composites here; '
             'Google AlphaEarth supplies an **annual multimodal** product. Inputs, model sizes and temporal aggregation differ. '
             'Matching space and year does not make the experiments equivalent. There is no ground truth, universal winner, '
             'cross-model vector cosine, or direct dimension mapping in this lab.' if comparison else
             'Turn real Sentinel-2 observations into spatial features with an openly available pretrained model. '
             'You will read a neighborhood in RGB, explore its embedding colors and unsupervised groups, '
             'ask which cells resemble a query, and compare your own named places. '
             '**Features are not labels:** similarity suggests a pattern to investigate, not a semantic conclusion.')
    cells = [markdown(f'''[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/webpug/geospatial/blob/main/{filename})

# {title}

{intro}

**Route:** private copy → setup → controls → connect → pretrained weights → real imagery → explore → compare places.

Start with the four public Seattle-area examples, **2024**, **64 × 64 pixels** (640 m square), and **V1 Nano**. A GPU is recommended; CPU works more slowly. V1 Nano is a lightweight, stable V1 default, **not the newest family or a maximum-accuracy choice**. Tiny and Base are optional and use more memory.'''), markdown('''## Your private workspace

This is a **public template**. Before entering personal names or coordinates, choose **File → Save a copy in Drive**. Open **Share**, set **General access → Restricted**, and inspect/remove collaborators. A launch link cannot enforce privacy. Never save personal entries, outputs, credentials or widget state back to GitHub; do not use **Save a copy in GitHub** for personal work.

Inputs and results stay in runtime memory. Earth Engine receives coordinates when you explicitly run acquisition; the optional Maps Search sends a query and key to Google only when clicked. Model weights download from Hugging Face and may be cached by its client. We do not write your places or imagery to Drive, a bucket or a repository. Results and charts can still expose your entries. `omit_outputs=true` and cleared template outputs are conveniences, not privacy guarantees: clear outputs, restart/delete the runtime, and inspect the file before sharing. Closing a browser alone does not clear runtime memory.

Edits invalidate in-memory results. Existing plots remain labeled snapshots until rerun. There are no substitute datasets, random-weight fallbacks or precomputed demonstration outputs.'''), markdown('''## 1 · Install the small inference package

Requires **Python ≥3.11**, PyTorch ≥2.7, an Earth Engine account and your own Cloud project with Earth Engine enabled. Colab normally supplies PyTorch; the installer retains a compatible installation. Choose **Runtime → Change runtime type → GPU** if available. If an already-imported dependency changes, restart the runtime and run from here again.

The minimal package is pinned to the reviewed commit below. Its dependencies resolve at installation time. First weight loading needs internet; a failed download stops the lab.'''), code(f'''#@title Install dependencies (run once per fresh runtime)
import sys
if sys.version_info < (3, 11):
    raise RuntimeError("Use a Python 3.11+ Colab runtime.")
%pip install -q "olmoearth-pretrain-minimal @ git+https://github.com/allenai/olmoearth_pretrain_minimal.git@{SHA}" earthengine-api pyproj numpy pandas scipy scikit-learn matplotlib ipywidgets requests
from google.colab import output
output.enable_custom_widget_manager()'''), markdown('''## 2 · Load the self-contained lab code

These embedded helpers are part of this notebook: no live repository helper or second notebook is imported. They implement explicit grids, validity masks, pretrained inference and local controls. Run all four folded cells once.'''), embedded('earth_helpers.py'), embedded('lab_ui.py'), embedded('maps_key.py'), embedded('maps_search.py'), code(f'''#@title Open controls · edits make no Earth Engine requests
COMPARISON = {comparison!r}
if 'PLACES' not in globals():
    PLACES = DEFAULT_PLACES.copy()
if 'lab' not in globals():
    lab = LabControls(PLACES, comparison=COMPARISON)
places_editor = lab.editor
display(lab.view)'''), markdown('''### Optional · Google Maps coordinate search

Manual entry needs no Maps key. For search, enable billing and **Places API (New)** and restrict your key to that API. Requests may incur charges. Preferred: add `GOOGLE_MAPS_API_KEY` to **Colab Secrets (key icon)** and enable notebook access. Alternatively run the masked prompt below; it never falls back to visible input. A blank entry or cancellation clears the runtime key; otherwise it takes precedence over Secrets until cleared or the runtime is deleted.

Click **Search**, select a candidate, then **Add selected to PLACES** or **Use as chip center**. These transfer buttons only edit controls; run acquisition explicitly afterward. Google Maps attribution appears with search results. This search chooses a chip center; the later spatial query stays inside the acquired chip.'''), code('''#@title Optional: set / clear a runtime Maps key (skip when using Secrets)
set_runtime_maps_key()'''), code('''#@title Optional: open Maps search (no request until Search)
if 'places_search' in globals():
    places_search.close()
places_search = PlacesSearch()
display(places_search.view)'''), markdown('''## 3 · Connect and load real pretrained weights

Enter your own project in the blank control above, then run both cells. Blank projects fail before authentication. Changing the model requires rerunning model loading. The loader always requests pretrained weights; errors propagate rather than switching to random features.'''), code('''#@title Authenticate Earth Engine with your project
import ee
lab.authenticate(ee)'''), code('''#@title Load the selected real V1 pretrained model
# Nano uses load_model_from_id(ModelID.OLMOEARTH_V1_NANO, load_weights=True).
# Tiny/Base use the matching ModelID with the same mandatory load_weights=True.
lab.load_model()'''), markdown('''## 4 · Acquire the selected neighborhood

Each quarter of the selected year contributes a **median Sentinel-2 SR_HARMONIZED composite**. SCL classes **4, 5, 6, 7** retain vegetation, bare surfaces, water and unclassified surfaces; clouds, shadows, snow and other classes are excluded. SCL screening is imperfect. February/May/August/November **15th** are explicitly **representative dates**, not actual scene dates. Timestamps use day, **zero-based month**, year.

The chip is centered on your coordinates in its local UTM zone, with a fixed 10 m north-up affine transform. Native 20/60 m bands are sampled by nearest neighbor; this does not create 10 m detail. Earth Engine returns a structured NumPy array plus explicit per-band validity on that exact grid. Four sequential small downloads avoid bucket exports.

Raw L2A **DN** values go directly to `Normalizer(std_multiplier=2)`—no division by 10000. The band order is **B02 B03 B04 B08 B05 B06 B07 B8A B11 B12 B01 B09**. Model inputs and masks are **B,H,W,T,12**; mask 0 means online encoder, mask 3 means missing. Transport fill values are never counted as valid data.

Encoder output is **B,H/4,W/4,T,S,D** at patch size 4 and input resolution 10 m. We average time and band-set dimensions and normalize the final vectors. A plotted OLMo token conservatively requires every input pixel, band and quarter to be valid. Its nominal footprint is **40 × 40 m**, but the contextual receptive field extends beyond that footprint. An empty collection or no valid output stops with an error.''')]
    if comparison:
        cells.append(markdown('''### Match the AlphaEarth grid

Load **GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL**, all **A00…A63** bands, for the same year and AOI. Availability is checked live; unsupported years and missing coverage fail clearly. AlphaEarth's 10 m vectors are downloaded on the **identical affine grid**, averaged over valid pixels in each **4 × 4 block**, then normalized. A block with at least one valid Alpha vector may participate; per-block valid fractions remain available as `lab.require_chip()['alpha_block_coverage']`.

Both models' panels, query maps and place summaries use the **intersection of valid 40 m output cells**. OLMo uses the stricter all-input rule. Every place uses its own centered grid, shared exactly by the two models. The annual Alpha product may omit open water or other locations; no imagery is substituted.'''))
    cells.extend([code('''#@title Acquire the chip · explicit network + inference step
# Results are published only after all required steps succeed.
lab.acquire_chip(ee)
print(json.dumps(lab.require_chip()['provenance'], indent=2))'''), markdown('''### Read the provenance before reading the colors

Coverage fractions describe retained input values, each quarter, output tokens and (in the comparison) shared support. The printed record includes collection, year, windows, representative timestamps, band order, affine transform, code SHA, model ID and pooling rule.

Timing is **end-to-end pipeline runtime**, not a training or inference speed contest. OLMo's pipeline includes Earth Engine download, preparation and local inference; Alpha's includes download and aggregation of an already-computed product. Model loading is reported separately. Hardware, network and service caches affect the numbers. Imagery is downloaded again on every acquisition; there is no data cache to hide changed inputs.'''), markdown('''## 5 · See the neighborhood

All panels share the same spatial footprint and north-up orientation. RGB is a stretched **third-quarter** reference composite. PCA compresses features into three display colors fitted on valid cells; these are **not RGB, land-cover labels or absolute feature meanings**. Missing data is gray/transparent. PCA needs at least three valid cells.

''' + ('''Each model fits its **own PCA**: colors across the two models are **not comparable**, even if they happen to look alike. Compare spatial patterns and masks, not matching hues.''' if comparison else '''K-means groups similar embedding vectors without labels. Group numbers and colors have no fixed semantics, and changing the chip or number of clusters can change them. Check patterns against RGB and local knowledge before forming a hypothesis.''')), code('''#@title Draw aligned panels from the current chip
paint_chip(lab.require_chip())'''), markdown('''## 6 · Ask a spatial question

Use a **zero-based token row/column** (0…15 for a 64 px chip), or enter latitude/longitude inside the chip. Click **Show similarity** to draw a cosine heatmap. The star marks the query footprint; gray cells are excluded. The center cell may be missing—choose a colored valid cell instead.

A high score means the model represents two cells similarly, not that they share a known label. These controls run locally on acquired features and never contact Earth Engine.''' + (''' The same query footprint is used for both models. Each cosine is computed **within its own model's feature space**. The heatmap comparison does not imply calibrated scores or compatible dimensions.''' if comparison else '')), code('''#@title Interactive local query
query_view = spatial_query_controls(lab)'''), markdown('''## 7 · Which neighborhoods resemble one another?

Edit **PLACES** above: add/update, select/remove, or reset four public examples (urban, park, port, forest). Select a place and click **Use selected as chip** to explore it in the earlier cells. Use **2–8 places** for this comparison; requests run sequentially. Each place is a neighborhood mean over its valid output cells, normalized after averaging—not a single point measurement.

Run acquisition after editing, then choose a named ranking query. This downloads a new chip per place. If any place fails, no partial result or previous ranking is reused. Coverage differences can affect neighborhood summaries.'''), code('''#@title Acquire all PLACES · sequential network + inference step
lab.acquire_places(ee)
# Full provenance per place is retained; printed records can contain your coordinates.
for name, result in lab.require_places().items():
    print(name)
    print(json.dumps(result['provenance'], indent=2))'''), code('''#@title Compare place similarities and choose a ranking query locally
ranking_view = ranking_controls(lab)'''), markdown('''### Interpret the comparison carefully

''' + ('''The matrices contain within-model place cosine. The **OLMo − Alpha** difference is a diagnostic of different geometry, **not accuracy**. The neighbor table compares matching place identities, excluding the query itself. Spearman measures agreement in neighbor ordering, with average ranks for ties. It is undefined for fewer than three neighbors, constant scores or invalid values.

Agreement can be interesting even when scores differ; disagreement can reflect modalities, seasonal aggregation, training, model size, coverage or spatial context. Four places are a small exploratory sample. Neither rankings nor PCA establish a universal winner without a labeled evaluation.''' if comparison else '''Matrix entries are within-model cosine between pooled neighborhood vectors. The ranking excludes the query itself. Urban, park, port and forest are names for public example locations, not training labels or predictions. Similarity may be affected by season, cloud filtering, water coverage, neighborhood composition and the model's spatial context.''') + '''

**Try next:** change one coordinate or the year, rerun acquisition and inspect both coverage and patterns. Keep the inputs and aggregation fixed when comparing runs. If you see no data, try another place/year; if you run out of memory, choose Nano and 32 px. If authentication fails, check project registration/access and rerun the connection cell. This lab has offline contract tests; live authenticated Earth Engine and hosted Colab execution must still be verified in your runtime.'''), markdown('''## Credits, sources and licenses

- **Ai2 OLMoEarth team**: [official project](https://github.com/allenai/olmoearth_pretrain), [platform](https://olmoearth.allenai.org/), [inference quickstart](https://github.com/allenai/olmoearth_pretrain/blob/main/docs/Inference-Quickstart.md), [minimal package](https://github.com/allenai/olmoearth_pretrain_minimal), and [V1 Nano pretrained weights](https://huggingface.co/allenai/OlmoEarth-v1-Nano). Model artifacts use the [OLMoEarth Artifact License](https://github.com/allenai/olmoearth_pretrain/blob/main/LICENSE); this is **not a blanket MIT license for weights**.
- **Ivan Zvonkov**: inspiration from the official [Earth Engine embedding generation notebook](https://github.com/allenai/olmoearth_pretrain/blob/main/scripts/tools/generate_embeddings.ipynb). This small-chip lab uses the minimal package and direct pixel requests instead of that notebook's bucket workflow.
- **Google / Google DeepMind AlphaEarth**: [Satellite Embedding V1 dataset, usage and terms](https://developers.google.com/earth-engine/datasets/catalog/GOOGLE_SATELLITE_EMBEDDING_V1_ANNUAL). Sentinel-2: [harmonized L2A catalog and terms](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_SR_HARMONIZED); pixel downloads follow [computePixels](https://developers.google.com/earth-engine/apidocs/ee-data-computepixels).
- **Teodora Szasz, PhD**: the adapted lab experience builds on her original **“You Are Hand-Labelling Data Google Already Described for Free”**, preserved in [this repository](https://github.com/webpug/geospatial/blob/main/external_inspirations/standout_systems_labelling_lab.ipynb). [Her work](https://teodora.coach/) inspired our [AlphaEarth playground](https://colab.research.google.com/github/webpug/geospatial/blob/main/alpha_earth_embeddings_demo.ipynb) and its place-editor and privacy patterns; the optional Maps helpers here are adapted from that demo.

[All labs and requirements](https://github.com/webpug/geospatial#launch-the-labs) · [OLMoEarth demo](https://colab.research.google.com/github/webpug/geospatial/blob/main/olmo_earth_embeddings_demo.ipynb) · [OLMoEarth / AlphaEarth comparison](https://colab.research.google.com/github/webpug/geospatial/blob/main/olmo_vs_alpha_earth_comparison.ipynb)
''')])
    notebook = nbf.v4.new_notebook(cells=cells, metadata=dict(
        colab=dict(name=filename, omit_outputs=True),
        kernelspec=dict(display_name='Python 3', language='python', name='python3'),
        language_info=dict(name='python', version='3.11')))
    for i, cell in enumerate(notebook.cells):
        cell.id = f'earth-lab-{i:03d}'
        if cell.cell_type == 'code' and cell.source.startswith('# Embedded source:'):
            cell.metadata['cellView'] = 'form'
    return filename, notebook


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for comparison in (False, True):
        filename, notebook = build(comparison)
        rendered = nbf.writes(notebook) + '\n'
        path = ROOT/filename
        if args.check:
            assert path.read_text() == rendered, f'Regenerate {filename}'
        else:
            path.write_text(rendered)
