"""Shared implementation embedded verbatim in both standalone public notebooks.

No downloads or authentication occur on import. See scripts/build_earth_notebooks.py.
"""
import json
import math
import time
from dataclasses import dataclass, asdict
from datetime import date
import numpy as np
from pyproj import Transformer

CODE_SHA = '96eeae43032d50dfc637d1200819b3966994392b'
S2_COLLECTION = 'COPERNICUS/S2_SR_HARMONIZED'
ALPHA_COLLECTION = 'GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL'
MODEL_BANDS = ('B02', 'B03', 'B04', 'B08', 'B05', 'B06', 'B07', 'B8A', 'B11', 'B12', 'B01', 'B09')
EE_BANDS = ('B2', 'B3', 'B4', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12', 'B1', 'B9')
ALPHA_BANDS = tuple(f'A{i:02d}' for i in range(64))
PATCH = 4
SCALE = 10
MAX_PLACES = 8
DEFAULT_PLACES = {
    'Seattle urban': (47.6062, -122.3321),
    'Discovery Park': (47.6597, -122.4165),
    'Seattle port': (47.5806, -122.3509),
    'Tiger Mountain forest': (47.4881, -121.9460),
}


def query_coordinates(latitude, longitude):
    if isinstance(latitude, bool) or isinstance(longitude, bool):
        raise ValueError('Coordinates must be numbers, not booleans.')
    try:
        lat, lon = float(latitude), float(longitude)
    except (TypeError, ValueError, OverflowError):
        raise ValueError('Enter numeric latitude and longitude.') from None
    if not (math.isfinite(lat) and math.isfinite(lon)):
        raise ValueError('Coordinates must be finite.')
    if not (-80 <= lat <= 84 and -180 <= lon <= 180):
        raise ValueError('UTM needs latitude -80 to 84 and longitude -180 to 180.')
    return lat, lon


def validate_places(places, minimum=2):
    if not minimum <= len(places) <= MAX_PLACES:
        raise ValueError(f'Use {minimum} to {MAX_PLACES} places.')
    for name, coords in places.items():
        if not isinstance(name, str) or not name.strip() or len(name) > 100:
            raise ValueError('Place names need 1–100 characters.')
        query_coordinates(*coords)
    return tuple((name, *query_coordinates(*coords)) for name, coords in places.items())


@dataclass(frozen=True)
class Settings:
    project: str
    year: int = 2024
    size: int = 64
    model: str = 'OLMOEARTH_V1_NANO'

    def __post_init__(self):
        if not isinstance(self.project, str) or not self.project.strip():
            raise ValueError('Enter your own Earth Engine project ID; rerun authentication.')
        object.__setattr__(self, 'project', self.project.strip())
        if type(self.year) is not int or not 2017 <= self.year <= 2100:
            raise ValueError('Use a calendar year from 2017 to 2100; coverage is checked live.')
        if type(self.size) is not int or self.size not in (32, 64, 128):
            raise ValueError('Chip size must be 32, 64 or 128 pixels.')
        if self.model not in ('OLMOEARTH_V1_NANO', 'OLMOEARTH_V1_TINY', 'OLMOEARTH_V1_BASE'):
            raise ValueError('Choose a supported pretrained V1 model.')


def quarters(year):
    # Representative composite dates, deliberately NOT claimed as scene dates.
    return [dict(start=f'{year}-{month:02d}-01',
                 end=f'{year + 1}-01-01' if month == 10 else f'{year}-{month + 3:02d}-01',
                 representative=f'{year}-{month + 1:02d}-15')
            for month in (1, 4, 7, 10)]


def timestamps(year):
    dates = [date.fromisoformat(q['representative']) for q in quarters(year)]
    return np.array([[[d.day, d.month - 1, d.year] for d in dates]], dtype=np.int64)


def make_grid(lat, lon, size):
    lat, lon = query_coordinates(lat, lon)
    if type(size) is not int or size not in (32, 64, 128):
        raise ValueError('Chip size must be 32, 64 or 128.')
    zone = min(60, int((lon + 180) // 6) + 1)
    # Standard UTM exceptions for Norway / Svalbard.
    if 56 <= lat < 64 and 3 <= lon < 12:
        zone = 32
    if 72 <= lat < 84 and 0 <= lon < 42:
        zone = 31 if lon < 9 else 33 if lon < 21 else 35 if lon < 33 else 37
    crs = f'EPSG:{(32600 if lat >= 0 else 32700) + zone}'
    x, y = Transformer.from_crs('EPSG:4326', crs, always_xy=True).transform(lon, lat)
    return dict(crsCode=crs, dimensions=dict(width=size, height=size),
                affineTransform=dict(scaleX=SCALE, shearX=0, translateX=x-size*SCALE/2,
                                     shearY=0, scaleY=-SCALE, translateY=y+size*SCALE/2))


def affine_list(grid):
    a = grid['affineTransform']
    return [a[k] for k in ('scaleX', 'shearX', 'translateX', 'shearY', 'scaleY', 'translateY')]


def grid_region(ee, grid):
    a = grid['affineTransform']
    x, y = a['translateX'], a['translateY']
    return ee.Geometry.Rectangle([x, y-grid['dimensions']['height']*SCALE,
                                  x+grid['dimensions']['width']*SCALE, y],
                                 proj=grid['crsCode'], geodesic=False)


def read_pixels(ee, image, bands, grid):
    """Export validity explicitly; filling transports pixels, never creates validity."""
    bands = list(bands)
    valid_names = ['valid_' + b for b in bands]
    selected = image.select(bands)
    validity = selected.mask().rename(valid_names).unmask(0, sameFootprint=False)
    packed = selected.unmask(0, sameFootprint=False).addBands(validity).toFloat()
    response = ee.data.computePixels(dict(expression=packed, fileFormat='NUMPY_NDARRAY',
                                          bandIds=bands + valid_names, grid=grid))
    expected = (grid['dimensions']['height'], grid['dimensions']['width'])
    if response.shape != expected or not set(bands + valid_names) <= set(response.dtype.names or ()):
        raise ValueError('Earth Engine returned an unexpected pixel grid or band schema.')
    values = np.stack([response[b] for b in bands], axis=-1).astype(np.float32)
    valid = np.stack([response[b] > 0 for b in valid_names], axis=-1) & np.isfinite(values)
    return np.where(valid, values, np.nan), valid


def acquire_s2(ee, settings, grid):
    region = grid_region(ee, grid)
    values, masks, counts = [], [], []
    for window in quarters(settings.year):  # Sequential, four bounded requests.
        collection = (ee.ImageCollection(S2_COLLECTION).filterBounds(region)
                      .filterDate(window['start'], window['end']))
        count = collection.size().getInfo()
        if not count:
            raise ValueError(f"No Sentinel-2 scenes in {window['start']} to {window['end']}.")
        def prepare(image):
            scl = image.select('SCL')
            good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6)).Or(scl.eq(7))
            # Nearest-neighbor resampling to the common grid before the median.
            # Native 20/60 m bands remain lower-information observations.
            return (image.select(list(EE_BANDS)).updateMask(good)
                    .reproject(crs=grid['crsCode'], crsTransform=affine_list(grid)))
        composite = collection.map(prepare).median()
        data, valid = read_pixels(ee, composite, EE_BANDS, grid)
        if not valid.any():
            raise ValueError(f"No valid Sentinel-2 pixels in quarter starting {window['start']}.")
        values.append(data)
        masks.append(valid)
        counts.append(int(count))
    return np.stack(values, axis=2), np.stack(masks, axis=2), counts


def acquire_alpha(ee, settings, grid):
    collection = (ee.ImageCollection(ALPHA_COLLECTION).filterBounds(grid_region(ee, grid))
                  .filterDate(f'{settings.year}-01-01', f'{settings.year + 1}-01-01'))
    if not collection.size().getInfo():
        raise ValueError(f'No AlphaEarth coverage for this AOI in {settings.year}. Try 2024 or another AOI.')
    data, valid = read_pixels(ee, collection.sort('system:index').mosaic(), ALPHA_BANDS, grid)
    if not valid.all(axis=-1).any():
        raise ValueError('AlphaEarth has no valid vectors on this exact grid.')
    return data, valid.all(axis=-1)


def block_view(array, patch=PATCH):
    h, w = array.shape[:2]
    if h % patch or w % patch:
        raise ValueError('Spatial dimensions must divide by patch size.')
    return array.reshape(h//patch, patch, w//patch, patch, *array.shape[2:])


def conservative_tokens(valid):
    if valid.ndim != 4 or valid.shape[2:] != (4, 12):
        raise ValueError('Expected H,W,4,12 validity.')
    return block_view(valid.all(axis=(2, 3))).all(axis=(1, 3))


def unit_vectors(values, valid):
    values = np.asarray(values, dtype=np.float32)
    norm = np.linalg.norm(np.where(np.isfinite(values), values, 0), axis=-1)
    good = np.asarray(valid, bool) & np.isfinite(values).all(axis=-1) & (norm > 1e-12)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    result[good] = values[good] / norm[good, None]
    return result, good


def aggregate_alpha(values, valid):
    valid = valid & np.isfinite(values).all(axis=-1)
    counts = block_view(valid).sum(axis=(1, 3))
    sums = block_view(np.where(valid[..., None], values, 0)).sum(axis=(1, 3))
    means = sums / np.maximum(counts[..., None], 1)
    vectors, good = unit_vectors(means, counts > 0)
    return vectors, good, counts / (PATCH * PATCH)


def prepare_sample(dn, valid, year, device='cpu'):
    import torch
    from olmoearth_pretrain_minimal import Normalizer
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
    from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskedOlmoEarthSample, MaskValue
    if dn.ndim != 4 or dn.shape[2:] != (4, 12) or dn.shape != valid.shape:
        raise ValueError('Expected matching H,W,4,12 DN and validity arrays.')
    if tuple(Modality.SENTINEL2_L2A.band_order) != MODEL_BANDS:
        raise ValueError('Installed model band order differs from this pinned lab.')
    valid = valid & np.isfinite(dn)
    # Feed raw L2A DN to the official normalizer. Do NOT divide by 10000.
    normalized = Normalizer(std_multiplier=2).normalize(Modality.SENTINEL2_L2A,
                                                        np.where(valid, dn, 0)[None])
    normalized = np.where(valid[None], normalized, 0).astype(np.float32)
    mask = np.where(valid[None], MaskValue.ONLINE_ENCODER.value, MaskValue.MISSING.value)
    return MaskedOlmoEarthSample(
        sentinel2_l2a=torch.as_tensor(normalized, device=device),
        sentinel2_l2a_mask=torch.as_tensor(mask, dtype=torch.long, device=device),
        timestamps=torch.as_tensor(timestamps(year), dtype=torch.long, device=device))


def infer_olmo(model, dn, valid, year, device):
    import torch
    good = conservative_tokens(valid & np.isfinite(dn))
    if not good.any():
        raise ValueError('No fully valid 40 m tokens across all bands and quarters. Try another AOI/year.')
    sample = prepare_sample(dn, valid, year, device)
    with torch.inference_mode():
        features = model.encoder(sample, patch_size=4, input_res=10, fast_pass=True)['tokens_and_masks'].sentinel2_l2a
        if features.ndim != 6 or tuple(features.shape[:4]) != (1, *good.shape, 4):
            raise ValueError('Unexpected encoder output; expected B,H/4,W/4,T,S,D.')
        pooled = features.mean(dim=(3, 4))[0].float().cpu().numpy()
    vectors, good = unit_vectors(pooled, good)
    if not good.any():
        raise ValueError('The encoder produced no finite, nonzero valid output vectors.')
    return vectors, good


class Pipeline:
    """Only model weights are reused in memory. Imagery/results are never cached."""
    def __init__(self):
        self.model = None
        self.model_key = None
        self.device = None
        self.load_seconds = None

    def load(self, settings):
        import torch
        from olmoearth_pretrain_minimal import ModelID, load_model_from_id
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        key = (CODE_SHA, settings.model, device)
        if self.model_key == key and self.model is not None:
            return
        self.model = self.model_key = None  # A failed replacement cannot expose old weights.
        start = time.perf_counter()
        model_id = getattr(ModelID, settings.model)
        model = load_model_from_id(model_id, load_weights=True)
        model = model.to(device).eval()
        self.model, self.model_key, self.device = model, key, device
        self.load_seconds = time.perf_counter() - start

    def run(self, ee, settings, lat, lon, comparison=False):
        if self.model is None or self.model_key[:2] != (CODE_SHA, settings.model):
            raise ValueError('Run the pretrained model load cell for the selected model first.')
        grid = make_grid(lat, lon, settings.size)
        start = time.perf_counter()
        dn, valid, counts = acquire_s2(ee, settings, grid)
        download_s2 = time.perf_counter() - start
        vectors, good = infer_olmo(self.model, dn, valid, settings.year, self.device)
        olmo_seconds = time.perf_counter() - start
        result = dict(grid=grid, dn=dn, input_valid=valid, olmo=vectors, olmo_valid=good,
                      shared_valid=good.copy(), settings=asdict(settings))
        runtime = dict(olmo_pipeline_seconds=olmo_seconds, sentinel2_download_seconds=download_s2,
                       olmo_model_load_seconds=self.load_seconds, device=self.device)
        coverage = dict(sentinel2_band_time_fraction=float(valid.mean()),
                        sentinel2_quarter_fractions=valid.mean(axis=(0, 1, 3)).tolist(),
                        olmo_token_fraction=float(good.mean()))
        if comparison:
            start = time.perf_counter()
            alpha, alpha_valid = acquire_alpha(ee, settings, grid)
            download_alpha = time.perf_counter() - start
            alpha, alpha_good, fraction = aggregate_alpha(alpha, alpha_valid)
            shared = good & alpha_good
            if not shared.any():
                raise ValueError('No common valid output cells for both models.')
            result.update(alpha=alpha, alpha_valid=alpha_good, shared_valid=shared,
                          alpha_block_coverage=fraction)
            runtime.update(alpha_pipeline_seconds=time.perf_counter()-start,
                           alpha_download_seconds=download_alpha)
            coverage.update(alpha_10m_fraction=float(alpha_valid.mean()),
                            alpha_token_fraction=float(alpha_good.mean()),
                            alpha_mean_block_fraction=float(fraction.mean()),
                            shared_token_fraction=float(shared.mean()))
        result['provenance'] = dict(
            sentinel2_collection=S2_COLLECTION, alpha_collection=ALPHA_COLLECTION if comparison else None,
            year=settings.year, quarterly_windows=quarters(settings.year), quarter_scene_counts=counts,
            timestamp_meaning='Representative composite dates, NOT actual scene acquisition dates',
            timestamps_day_zero_based_month_year=timestamps(settings.year).tolist(),
            grid=grid, scale_metres=SCALE, ee_bands=EE_BANDS, model_bands=MODEL_BANDS,
            alpha_bands=ALPHA_BANDS if comparison else None, model_id=settings.model, code_sha=CODE_SHA,
            normalization='Official Normalizer(std_multiplier=2) on raw L2A DN',
            masking='SCL 4,5,6,7; output OLMo requires every pixel/band/quarter valid',
            resampling='Nearest neighbor to explicit 10 m UTM grid before quarterly median',
            aggregation='OLMo time/bandset mean; Alpha valid-only 4x4 mean; L2 normalize final vectors',
            coverage=coverage, runtime=runtime)
        return result


def spatial_query(grid, latitude=None, longitude=None, row=None, column=None):
    if row is None or column is None:
        lat, lon = query_coordinates(latitude, longitude)
        x, y = Transformer.from_crs('EPSG:4326', grid['crsCode'], always_xy=True).transform(lon, lat)
        a = grid['affineTransform']
        column = math.floor((x-a['translateX']) / (SCALE*PATCH))
        row = math.floor((a['translateY']-y) / (SCALE*PATCH))
    if type(row) is not int or type(column) is not int:
        raise ValueError('Token row/column must be integers.')
    if not (0 <= row < grid['dimensions']['height']//PATCH and
            0 <= column < grid['dimensions']['width']//PATCH):
        raise ValueError('Query is outside this chip. Change the query or acquire a new chip.')
    return row, column


def similarity_to_query(vectors, valid, row, column):
    if not valid[row, column]:
        raise ValueError('Query token is invalid. Choose a colored cell on the coverage/PCA panel.')
    scores = np.full(valid.shape, np.nan)
    # Both operands are from ONE model. No cross-model cosine API exists here.
    scores[valid] = np.clip(vectors[valid] @ vectors[row, column], -1, 1)
    return scores


def place_vector(result, model):
    valid = result['shared_valid']
    vector, good = unit_vectors(result[model][valid].mean(axis=0)[None], np.array([True]))
    if not good[0]:
        raise ValueError('Place mean has zero norm; cannot compare it.')
    return vector[0]


def pairwise(vectors):
    # A single homogeneous model matrix; differing feature counts fail at stacking.
    values = np.stack(vectors)
    if values.ndim != 2:
        raise ValueError('Expected a single model matrix of place vectors.')
    return np.clip(values @ values.T, -1, 1)


def rank_agreement(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError('Rank inputs must describe the same neighbors.')
    if len(a) < 3 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    if np.ptp(a) < 1e-10 or np.ptp(b) < 1e-10:
        return None
    score = float(spearmanr(a, b).statistic)
    return score if np.isfinite(score) else None


def pca_colors(vectors, valid):
    from sklearn.decomposition import PCA
    selected = vectors[valid]
    if len(selected) < 3:
        raise ValueError('PCA needs at least three valid cells; try another AOI/year.')
    colors = PCA(n_components=3, svd_solver='full').fit_transform(selected)
    lo, hi = np.percentile(colors, [2, 98], axis=0)
    colors = np.clip((colors-lo) / np.maximum(hi-lo, 1e-8), 0, 1)
    rgba = np.zeros((*valid.shape, 4))
    rgba[valid, :3] = colors
    rgba[valid, 3] = 1
    return rgba


def rgb_image(result):
    # Quarter 3 reference image, DN stretch for DISPLAY only; never feeds inference.
    rgb = result['dn'][:, :, 2, [2, 1, 0]]
    valid = result['input_valid'][:, :, 2, [2, 1, 0]].all(axis=-1)
    rgba = np.zeros((*valid.shape, 4))
    rgba[valid, :3] = np.clip(rgb[valid] / 3000, 0, 1) ** (1/1.6)
    rgba[valid, 3] = 1
    return rgba


def paint_chip(result, clusters=5):
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans
    models = ['olmo', 'alpha'] if 'alpha' in result else ['olmo']
    valid = result['shared_valid']
    panels = [('Sentinel-2 · Q3 RGB', rgb_image(result), None)]
    for name in models:
        panels.append((f'{name.title()} · independent PCA', pca_colors(result[name], valid), None))
    if len(models) == 1:
        selected = result['olmo'][valid]
        k = min(clusters, len(np.unique(selected, axis=0)))
        labels = np.full(valid.shape, np.nan)
        labels[valid] = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(selected)
        panels.append((f'Unsupervised groups · k={k}', labels, 'tab10'))
    panels.append(('Common valid cells' if 'alpha' in result else 'Valid output cells', valid.astype(float), 'gray'))
    fig, axes = plt.subplots(1, len(panels), figsize=(5*len(panels), 5), constrained_layout=True)
    width = result['grid']['dimensions']['width']*SCALE
    for ax, (title, values, cmap) in zip(axes, panels):
        ax.set_facecolor('#cbd5e1')
        ax.imshow(values, origin='upper', extent=[0, width, width, 0], cmap=cmap, interpolation='nearest')
        ax.set(title=title, xlabel='Metres east from west edge', ylabel='Metres south from north edge')
    fig.suptitle(f"{result['settings']['year']} · {result['grid']['crsCode']} · north up · gray/transparent = missing")
    plt.show()


def paint_query(result, row, column):
    import matplotlib.pyplot as plt
    models = ['olmo', 'alpha'] if 'alpha' in result else ['olmo']
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 5), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes.flat, models):
        scores = similarity_to_query(result[model], result['shared_valid'], row, column)
        im = ax.imshow(scores, vmin=-1, vmax=1, cmap='viridis', interpolation='nearest')
        ax.set_facecolor('#cbd5e1')
        ax.scatter([column], [row], c='white', edgecolors='black', marker='*', s=160)
        ax.set(title=f'{model.title()} · cosine to query', xlabel='40 m token column', ylabel='40 m token row')
        fig.colorbar(im, ax=ax, label='Within-model cosine')
    plt.show()


def compare_places(results, query_name):
    import pandas as pd
    import matplotlib.pyplot as plt
    from IPython.display import display
    names = list(results)
    if query_name not in results:
        raise ValueError('Choose one of the current places as ranking query.')
    models = ['olmo', 'alpha'] if 'alpha' in results[names[0]] else ['olmo']
    matrices = {m: pairwise([place_vector(results[n], m) for n in names]) for m in models}
    count = len(models) + (len(models) == 2)
    fig, axes = plt.subplots(1, count, figsize=(6*count, 5), squeeze=False, constrained_layout=True)
    for ax, model in zip(axes.flat, models):
        im = ax.imshow(matrices[model], vmin=-1, vmax=1, cmap='viridis')
        ax.set_title(f'{model.title()} · place cosine')
        fig.colorbar(im, ax=ax)
    if len(models) == 2:
        im = axes.flat[-1].imshow(matrices['olmo']-matrices['alpha'], vmin=-2, vmax=2, cmap='coolwarm')
        axes.flat[-1].set_title('OLMo − Alpha similarity\nDiagnostic, not accuracy')
        fig.colorbar(im, ax=axes.flat[-1])
    for ax in axes.flat:
        ax.set_xticks(range(len(names)), names, rotation=40, ha='right')
        ax.set_yticks(range(len(names)), names)
        # User names are literal text, including dollar signs and backslashes.
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_parse_math(False)
    plt.show()
    q = names.index(query_name)
    others = [i for i in range(len(names)) if i != q]
    table = pd.DataFrame({'place': [names[i] for i in others]})
    for model in models:
        table[model + ' cosine'] = matrices[model][q, others]
        table[model + ' rank'] = table[model + ' cosine'].rank(ascending=False, method='average')
    display(table.sort_values(models[0] + ' rank'))
    if len(models) == 2:
        agreement = rank_agreement(matrices['olmo'][q, others], matrices['alpha'][q, others])
        print('Neighbor Spearman:', 'undefined (need ≥3 nonconstant neighbor scores)' if agreement is None else f'{agreement:.3f}')
    return matrices
