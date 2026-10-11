"""Offline contracts. FakeEE / FakeModel below are test doubles, never notebook data."""
import contextlib
import getpass
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nbformat
import numpy as np
import torch
from IPython.core.interactiveshell import InteractiveShell
from pyproj import Transformer
from notebook_sources import earth_helpers as h
from scripts.build_earth_notebooks import build

# Execute the same standalone embedded source cells used in Colab.
NS = {'__name__': 'earth_notebook_contract_tests'}
# dataclasses uses the declaring module at runtime.
from types import ModuleType, SimpleNamespace
module = ModuleType(NS['__name__'])
sys.modules[module.__name__] = module
NS = module.__dict__
for name in ('earth_helpers.py', 'lab_ui.py', 'maps_key.py', 'maps_search.py'):
    exec(compile((ROOT/'notebook_sources'/name).read_text(), name, 'exec'), NS)


class FakeImage:
    """Record EE expression construction without making a network request."""
    def __init__(self, log): self.log = log
    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.log.append((name, args, kwargs))
            return self
        return call


class FakeEE:
    def __init__(self, empty=False, missing=False):
        self.log, self.requests = [], []
        self.empty, self.missing = empty, missing
        self.image = FakeImage(self.log)
        self.Geometry = SimpleNamespace(Rectangle=Mock(return_value='test-region'))
        self.data = SimpleNamespace(computePixels=self.compute)
        self.Authenticate, self.Initialize = Mock(), Mock()
    def ImageCollection(self, name):
        self.collection_name = name
        self.log.append(('collection', (name,), {}))
        return self
    def filterBounds(self, region): return self
    def filterDate(self, *dates):
        self.log.append(('dates', dates, {}))
        return self
    def size(self): return SimpleNamespace(getInfo=lambda: 0 if self.empty else 2)
    def map(self, callback):
        callback(self.image)
        return self
    def sort(self, key): return self
    def median(self): return self.image
    def mosaic(self): return self.image
    def compute(self, request):
        self.requests.append(request)
        n = request['grid']['dimensions']['width']
        # Deliberately reversed field order: decoder must use explicit names.
        out = np.zeros((n, n), dtype=[(b, 'f4') for b in reversed(request['bandIds'])])
        for i, band in enumerate(request['bandIds']):
            if band.startswith('valid_'):
                out[band] = 0 if self.missing else 1
            else:
                out[band] = (i + 1)*100 + np.indices((n, n))[0]
        return out


class FakeModel:
    """Known tensors for contract tests only; never a model fallback."""
    def __init__(self): self.sample = self.kwargs = None
    def to(self, device): return self
    def eval(self): return self
    def encoder(self, sample, **kwargs):
        self.sample, self.kwargs = sample, kwargs
        n = sample.sentinel2_l2a.shape[1]//4
        vals = torch.arange(1, 1+n*n*4*3*8, dtype=torch.float32).reshape(1, n, n, 4, 3, 8)
        self.expected = vals.mean(dim=(3, 4))[0].numpy()
        return {'tokens_and_masks': SimpleNamespace(sentinel2_l2a=vals)}


class EarthContracts(unittest.TestCase):
    def test_notebook_clean_sync_and_syntax(self):
        transformer = InteractiveShell.instance().input_transformer_manager
        for comparison in (False, True):
            name, expected = build(comparison)
            notebook = nbformat.read(ROOT/name, as_version=4)
            nbformat.validate(notebook)
            self.assertEqual(notebook, expected)
            self.assertTrue(notebook.metadata.colab.omit_outputs)
            self.assertNotIn('widgets', notebook.metadata)
            for c in notebook.cells:
                if c.cell_type == 'code':
                    self.assertFalse(c.outputs)
                    self.assertIsNone(c.execution_count)
                    compile(transformer.transform_cell(c.source), name, 'exec')
                for key in ('widgets', 'executionInfo', 'outputId'):
                    self.assertNotIn(key, c.metadata)
            for file in ('earth_helpers.py', 'lab_ui.py', 'maps_key.py', 'maps_search.py'):
                cell = next(c for c in notebook.cells if c.source.startswith(f'# Embedded source: notebook_sources/{file}\n'))
                self.assertEqual(cell.source.split('\n', 1)[1], (ROOT/'notebook_sources'/file).read_text().strip())
            sources = '\n'.join(c.source for c in notebook.cells)
            self.assertNotIn('load_weights=False', sources)
            self.assertNotIn('np.random', sources)
            self.assertNotIn('import notebook_sources', sources)
            self.assertIn('blob/main/'+name, notebook.cells[0].source)
            for credit in ('Ivan Zvonkov', 'Teodora', 'Artifact License', 'Ai2', 'Google'):
                self.assertIn(credit, sources)

    def test_validation(self):
        for bad in ('', ' \t'):
            with self.assertRaises(ValueError): h.Settings(bad)
        self.assertEqual(h.Settings(' test-project ').project, 'test-project')
        for lat, lon in [(float('nan'), 0), (0, float('inf')), (-81, 0), (85, 0), (0, 181), (True, 0)]:
            with self.assertRaises(ValueError): h.query_coordinates(lat, lon)
        for year in (2024.1, True, 2016):
            with self.assertRaises(ValueError): h.Settings('test-project', year=year)
        with self.assertRaises(ValueError): h.Settings('test-project', size=40)
        with self.assertRaises(ValueError): h.Settings('test-project', model='random')
        with self.assertRaises(ValueError): h.validate_places({'': (0, 0)}, minimum=1)
        with self.assertRaises(ValueError): h.validate_places({str(i): (0, 0) for i in range(9)})

    def test_grid_center_axis_and_query(self):
        for lat, lon, epsg in [(47.6062, -122.3321, 'EPSG:32610'), (-30, 20, 'EPSG:32734'),
                               (60, 6, 'EPSG:32632'), (78, 15, 'EPSG:32633'), (0, 180, 'EPSG:32660')]:
            grid = h.make_grid(lat, lon, 64)
            self.assertEqual(grid['crsCode'], epsg)
            a = grid['affineTransform']
            x, y = Transformer.from_crs('EPSG:4326', epsg, always_xy=True).transform(lon, lat)
            self.assertAlmostEqual(a['translateX']+320, x)
            self.assertAlmostEqual(a['translateY']-320, y)
            self.assertEqual(h.affine_list(grid), [10, 0, x-320, 0, -10, y+320])
            # Center sits on four cell boundaries; choose a point strictly in row 3, col 7.
            qlon, qlat = Transformer.from_crs(epsg, 'EPSG:4326', always_xy=True).transform(
                a['translateX']+7.5*40, a['translateY']-3.5*40)
            self.assertEqual(h.spatial_query(grid, qlat, qlon), (3, 7))
            with self.assertRaises(ValueError): h.spatial_query(grid, row=16, column=0)
            with self.assertRaises(ValueError): h.spatial_query(grid, row=-1, column=0)

    def test_real_normalizer_dn_timestamps_masks(self):
        from olmoearth_pretrain_minimal import Normalizer
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.constants import Modality
        from olmoearth_pretrain_minimal.olmoearth_pretrain_v1.utils.datatypes import MaskValue
        self.assertEqual(tuple(Modality.SENTINEL2_L2A.band_order), h.MODEL_BANDS)
        self.assertEqual(h.EE_BANDS, ('B2','B3','B4','B8','B5','B6','B7','B8A','B11','B12','B1','B9'))
        dn = np.full((32,32,4,12), 2500, dtype=np.float32)
        valid = np.ones_like(dn, dtype=bool)
        valid[0,0,0,0] = False
        dn[0,0,0,0] = np.nan
        sample = h.prepare_sample(dn, valid, 2024)
        self.assertEqual(tuple(sample.sentinel2_l2a.shape), (1,32,32,4,12))
        self.assertEqual(sample.sentinel2_l2a.shape, sample.sentinel2_l2a_mask.shape)
        self.assertEqual(sample.sentinel2_l2a_mask.dtype, torch.long)
        self.assertEqual(sample.sentinel2_l2a_mask[0,0,0,0,0], MaskValue.MISSING.value)
        self.assertEqual(sample.sentinel2_l2a_mask[0,1,1,0,0], MaskValue.ONLINE_ENCODER.value)
        self.assertEqual(sample.sentinel2_l2a[0,0,0,0,0], 0)
        norm = Normalizer(std_multiplier=2)
        stats = norm.norm_config['sentinel2_l2a']['B02']
        expected = (2500-(stats['mean']-2*stats['std']))/(4*stats['std'])
        self.assertAlmostEqual(float(sample.sentinel2_l2a[0,1,1,0,0]), expected, places=6)
        np.testing.assert_array_equal(sample.timestamps.numpy(), [[[15,1,2024],[15,4,2024],[15,7,2024],[15,10,2024]]])
        self.assertEqual(h.quarters(2024)[-1]['end'], '2025-01-01')
        self.assertFalse(h.conservative_tokens(valid)[0,0])
        self.assertEqual(h.conservative_tokens(valid).sum(), 63)

    def test_inference_pool_and_empty(self):
        dn = np.full((32,32,4,12), 3000, dtype=np.float32)
        valid = np.ones_like(dn, dtype=bool)
        valid[0,0,0,0] = False
        model = FakeModel()
        vectors, good = h.infer_olmo(model, dn, valid, 2024, 'cpu')
        self.assertEqual(model.kwargs, dict(patch_size=4, input_res=10, fast_pass=True))
        expected, _ = h.unit_vectors(model.expected, good)
        np.testing.assert_allclose(vectors, expected)
        self.assertTrue(np.isnan(vectors[0,0]).all())
        np.testing.assert_allclose(np.linalg.norm(vectors[good], axis=-1), 1, atol=1e-6)
        with self.assertRaisesRegex(ValueError, 'No fully valid'):
            h.infer_olmo(model, dn, np.zeros_like(valid), 2024, 'cpu')

    def test_alpha_aggregation_and_coverage(self):
        values = np.zeros((8,8,2), dtype=np.float32)
        values[:4,:4] = [1,0]
        values[:4,4:] = [0,1]
        values[4:] = [1,1]
        valid = np.ones((8,8), bool)
        valid[0,0] = False
        values[0,0] = [999,999]
        valid[4:,4:] = False
        out, good, fractions = h.aggregate_alpha(values, valid)
        np.testing.assert_allclose(out[0,0], [1,0])
        np.testing.assert_allclose(out[0,1], [0,1])
        np.testing.assert_allclose(out[1,0], [2**-.5, 2**-.5])
        self.assertFalse(good[1,1])
        self.assertTrue(np.isnan(out[1,1]).all())
        self.assertEqual(fractions[0,0], 15/16)
        self.assertEqual(fractions[1,1], 0)
        # Mean THEN normalization; a single valid vector still has documented partial coverage.
        valid[:] = False
        valid[1,1] = True
        _, good, fractions = h.aggregate_alpha(values, valid)
        self.assertEqual(good.sum(), 1)
        self.assertEqual(fractions[0,0], 1/16)

    def test_ee_contract_and_missing(self):
        ee = FakeEE()
        cfg = h.Settings('test-project', size=32)
        grid = h.make_grid(47.6062, -122.3321, 32)
        dn, valid, counts = h.acquire_s2(ee, cfg, grid)
        self.assertEqual(dn.shape, (32,32,4,12))
        self.assertTrue(valid.all())
        self.assertEqual(counts, [2]*4)
        np.testing.assert_equal(dn[0,0,0], np.arange(1,13)*100)
        self.assertEqual(len(ee.requests), 4)
        for req in ee.requests:
            self.assertEqual(req['grid'], grid)
            self.assertEqual(req['fileFormat'], 'NUMPY_NDARRAY')
            self.assertEqual(req['bandIds'], list(h.EE_BANDS)+['valid_'+b for b in h.EE_BANDS])
        self.assertEqual([args[0] for name,args,kw in ee.log if name=='eq'], [4,5,6,7]*4)
        self.assertEqual([kw for name,args,kw in ee.log if name=='reproject'],
                         [dict(crs=grid['crsCode'], crsTransform=h.affine_list(grid))]*4)
        alpha, alpha_valid = h.acquire_alpha(ee, cfg, grid)
        self.assertEqual(alpha.shape, (32,32,64))
        self.assertTrue(alpha_valid.all())
        self.assertEqual(ee.requests[-1]['grid'], grid)
        for fn in (h.acquire_s2, h.acquire_alpha):
            empty = FakeEE(empty=True)
            with self.assertRaises(ValueError): fn(empty, cfg, grid)
            self.assertFalse(empty.requests)
            with self.assertRaises(ValueError): fn(FakeEE(missing=True), cfg, grid)

    def test_load_pretrained_only_and_failure(self):
        pipeline = h.Pipeline()
        cfg = h.Settings('test-project')
        from olmoearth_pretrain_minimal import ModelID
        with patch('olmoearth_pretrain_minimal.load_model_from_id', return_value=FakeModel()) as load:
            pipeline.load(cfg)
            load.assert_called_once_with(ModelID.OLMOEARTH_V1_NANO, load_weights=True)
            pipeline.load(cfg)
            self.assertEqual(load.call_count, 1)
        with patch('olmoearth_pretrain_minimal.load_model_from_id', side_effect=RuntimeError('test download failure')):
            with self.assertRaises(RuntimeError): pipeline.load(h.Settings('test-project', model='OLMOEARTH_V1_TINY'))
        self.assertIsNone(pipeline.model)
        self.assertIsNone(pipeline.model_key)

    def test_pipeline_shared_support_provenance_and_no_cache(self):
        p = h.Pipeline()
        p.model, p.device, p.load_seconds = FakeModel(), 'cpu', 0
        p.model_key = (h.CODE_SHA, 'OLMOEARTH_V1_NANO', 'cpu')
        ee = FakeEE()
        cfg = h.Settings('test-project', size=32)
        r = p.run(ee, cfg, 47.6062, -122.3321, comparison=True)
        self.assertEqual(r['olmo'].shape, (8,8,8))
        self.assertEqual(r['alpha'].shape, (8,8,64))
        np.testing.assert_equal(r['shared_valid'], r['olmo_valid'] & r['alpha_valid'])
        self.assertEqual(r['provenance']['coverage']['shared_token_fraction'], 1)
        self.assertEqual(r['provenance']['code_sha'], h.CODE_SHA)
        self.assertIn('alpha_pipeline_seconds', r['provenance']['runtime'])
        p.run(ee, cfg, 47.6062, -122.3321, comparison=True)
        self.assertEqual(len(ee.requests), 10)  # No data cache.
        # Similarities only ever consume one feature space.
        a = h.similarity_to_query(r['olmo'], r['shared_valid'], 0, 0)
        b = h.similarity_to_query(r['alpha'], r['shared_valid'], 0, 0)
        self.assertEqual(a.shape, b.shape)
        with self.assertRaises(ValueError): h.pairwise([np.ones(8), np.ones(64)])
        self.assertAlmostEqual(a[0,0], 1, places=5)
        with self.assertRaises(ValueError): h.similarity_to_query(r['olmo'], np.zeros((8,8), bool), 0, 0)
        self.assertIsNone(h.rank_agreement([1,2], [2,1]))
        self.assertIsNone(h.rank_agreement([1,1,1], [1,2,3]))
        self.assertAlmostEqual(h.rank_agreement([1,2,3], [3,2,1]), -1)
        self.assertAlmostEqual(h.rank_agreement([1,1,3], [2,2,4]), 1)
        # Execute actual rendering code on explicitly synthetic offline fixtures.
        with patch('matplotlib.pyplot.show'), patch('IPython.display.display'):
            h.paint_chip(r)
            h.paint_query(r, 0, 0)
            h.compare_places({n:r for n in h.DEFAULT_PLACES}, next(iter(h.DEFAULT_PLACES)))
        import matplotlib.pyplot as plt
        plt.close('all')

    def test_partial_shared_coverage_and_place_pooling(self):
        class PartialEE(FakeEE):
            def compute(self, request):
                out = super().compute(request)
                if 'valid_B2' in out.dtype.names:
                    out['valid_B2'][0,0] = 0
                else:
                    for name in out.dtype.names:
                        if name.startswith('valid_'):
                            out[name][4:8,4:8] = 0
                return out
        p = h.Pipeline()
        p.model, p.device, p.load_seconds = FakeModel(), 'cpu', 0
        p.model_key = (h.CODE_SHA, 'OLMOEARTH_V1_NANO', 'cpu')
        result = p.run(PartialEE(), h.Settings('test-project', size=32), 47, -122, comparison=True)
        self.assertEqual(result['olmo_valid'].sum(), 63)
        self.assertEqual(result['alpha_valid'].sum(), 63)
        self.assertEqual(result['shared_valid'].sum(), 62)
        self.assertFalse(result['shared_valid'][0,0])
        self.assertFalse(result['shared_valid'][1,1])
        self.assertEqual(result['provenance']['coverage']['shared_token_fraction'], 62/64)
        for model in ('olmo', 'alpha'):
            mean = result[model][result['shared_valid']].mean(axis=0)
            np.testing.assert_allclose(h.place_vector(result, model), mean/np.linalg.norm(mean))
            scores = h.similarity_to_query(result[model], result['shared_valid'], 2, 2)
            self.assertTrue(np.isnan(scores[0,0]))
            self.assertTrue(np.isnan(scores[1,1]))

    def test_widgets_stale_atomic_and_no_network(self):
        lab = NS['LabControls'](h.DEFAULT_PLACES.copy())
        ee = FakeEE()
        with self.assertRaises(ValueError): lab.authenticate(ee)
        ee.Authenticate.assert_not_called()
        lab.project.value = ' test-project '
        lab.authenticate(ee)
        ee.Initialize.assert_called_once_with(project='test-project')
        lab.pipeline.run = Mock(return_value={'test': 'result'})
        lab.acquire_chip(ee)
        self.assertEqual(lab.require_chip(), {'test': 'result'})
        lab.year.value = 2023
        with self.assertRaises(ValueError): lab.require_chip()
        self.assertEqual(lab.pipeline.run.call_count, 1)  # Edits don't acquire.
        lab.acquire_chip(ee)
        lab.pipeline.run.side_effect = ValueError('test missing data')
        with self.assertRaises(ValueError): lab.acquire_chip(ee)
        with self.assertRaises(ValueError): lab.require_chip()
        lab.pipeline.run.side_effect = [dict(ok=1), ValueError('test missing data')]
        with self.assertRaises(ValueError): lab.acquire_places(ee)
        with self.assertRaises(ValueError): lab.require_places()
        editor = lab.editor
        editor.name.value = '<b>Test & site</b>'
        editor.latitude.value, editor.longitude.value = '47', '-122'
        editor.add.click()
        self.assertIn('<b>Test & site</b>', lab.places)
        self.assertIn('&lt;b&gt;', editor.summary.value)
        self.assertIn('&lt;b&gt;', editor.status.value)
        before = lab.places.copy()
        editor.latitude.value = 'nan'
        editor.add.click()
        self.assertEqual(lab.places, before)
        editor.remove.click()
        self.assertNotIn('<b>Test & site</b>', lab.places)
        for i in range(4):
            editor.name.value, editor.latitude.value = f'Test {i}', '47'
            editor.add.click()
        self.assertEqual(len(lab.places), 8)
        editor.name.value = 'Ninth'
        editor.add.click()
        self.assertEqual(len(lab.places), 8)
        editor.reset.click()
        self.assertEqual(lab.places, h.DEFAULT_PLACES)
        # Direct dictionary edits are caught even outside widget callbacks.
        lab.pipeline.run.side_effect = None
        lab.acquire_chip(ee)
        lab.places['Seattle urban'] = (40, -120)
        with self.assertRaises(ValueError): lab.require_chip()

    def test_maps_privacy_contract(self):
        # Synthetic fixture key, mocked transport only.
        with patch('getpass.getpass', return_value='test-runtime-secret'), contextlib.redirect_stdout(io.StringIO()) as out:
            NS['set_runtime_maps_key']()
        self.assertNotIn('test-runtime-secret', out.getvalue())
        response = Mock(status_code=200)
        response.json.return_value = {'places':[dict(displayName={'text':'<b>test-runtime-secret</b>'},
                                                      location={'latitude':47,'longitude':-122})]}
        with patch('requests.post', return_value=response) as post:
            candidates, message = NS['search_place_candidates']('test city')
        self.assertNotIn('test-runtime-secret', str(candidates))
        self.assertEqual(post.call_args.kwargs['headers']['X-Goog-Api-Key'], 'test-runtime-secret')
        self.assertFalse(post.call_args.kwargs['allow_redirects'])
        self.assertEqual(post.call_args.kwargs['timeout'], (5,20))
        with patch('getpass.getpass', side_effect=getpass.GetPassWarning), contextlib.redirect_stdout(io.StringIO()):
            NS['set_runtime_maps_key']()
        self.assertIsNone(NS['_maps_runtime_api_key'])
        with patch('getpass.getpass', return_value=''), contextlib.redirect_stdout(io.StringIO()):
            NS['set_runtime_maps_key']()
        self.assertIsNone(NS['_maps_runtime_api_key'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
