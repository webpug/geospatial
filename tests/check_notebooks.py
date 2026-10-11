import ast
import contextlib
import io
import json
import os
from pathlib import Path
import sys
from unittest.mock import Mock, patch

os.environ['MPLBACKEND'] = 'Agg'
import nbformat
from IPython.core.interactiveshell import InteractiveShell
import numpy as np
# pandas is unavailable in this offline venv; emulate the small frame interface
# used by this unchanged analysis code with synthetic records.
class OfflineFrame:
    def __init__(self, records): self.records = records
    def set_index(self, key):
        self.index = [r[key] for r in self.records]
        return self
    def reindex(self, names):
        by_name = {r['name']: r for r in self.records}
        self.records = [by_name[n] for n in names]
        self.index = names
        return self
    def __getitem__(self, keys):
        class Values:
            def to_numpy(inner, dtype):
                return np.array([[r[k] for k in keys] for r in self.records], dtype=dtype)
        return Values()
    def __len__(self): return len(self.records)
class pd:
    DataFrame = OfflineFrame
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
transformer = InteractiveShell.instance().input_transformer_manager
notebooks = {}
for path in root.glob('*.ipynb'):
    n = nbformat.read(path, as_version=4)
    nbformat.validate(n)
    for index, cell in enumerate(n.cells):
        if cell.cell_type == 'code':
            compile(transformer.transform_cell(cell.source), f'{path.name}:cell{index}', 'exec')
    notebooks[path.name] = n
print('PASS: all three notebooks validate and every transformed code cell compiles.')

main = notebooks['alpha_earth_embeddings_demo.ipynb']

# Execute the connection code with a fake ee module, skipping install/widget setup.
connect_source = next(c.source for c in main.cells if 'ee.Authenticate()' in c.source)
connect_source = connect_source[connect_source.index('PROJECT_ID ='):]
assert connect_source.startswith('PROJECT_ID = ""  #@param {type:"string"}')

def with_inputs(source, **values):
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, (ast.Constant, ast.UnaryOp)):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in values:
                    node.value = ast.Constant(values[target.id])
    return compile(ast.fix_missing_locations(tree), '<form inputs>', 'exec')

for project in ('', ' \t\n ', '  synthetic-user-project  '):
    fake_ee = Mock()
    connection = {}
    with patch.dict(sys.modules, {'ee': fake_ee}), contextlib.redirect_stdout(io.StringIO()):
        try:
            exec(with_inputs(connect_source, PROJECT_ID=project), connection)
        except ValueError as exc:
            assert not project.strip()
            assert 'Enter your own Earth Engine project ID' in str(exc)
            assert 'PROJECT_ID' in str(exc) and 'rerun' in str(exc)
        else:
            assert project.strip(), 'Blank project must fail before authentication'
    if project.strip():
        fake_ee.Authenticate.assert_called_once_with()
        fake_ee.Initialize.assert_called_once_with(project='synthetic-user-project')
        assert connection['PROJECT_ID'] == 'synthetic-user-project'
    else:
        fake_ee.Authenticate.assert_not_called()
        fake_ee.Initialize.assert_not_called()
print('PASS: blank/whitespace projects fail before auth; supplied project is stripped and passed to mocked ee.')

# Run startup in a fresh namespace, and verify edited PEEK inputs also drive the map.
peek_source = next(c.source for c in main.cells if 'PEEK_LAT ' in c.source)
paint_source = next(c.source for c in main.cells if 'm1 = new_map(' in c.source)
for inputs, expected in [({}, (47.6062, -122.3321)),
                         ({'PEEK_LAT': 10.0, 'PEEK_LON': 20.0}, (10.0, 20.0))]:
    fake_ee = Mock()
    fake_ee.Geometry.Rectangle.side_effect = tuple
    sample = fake_ee.ImageCollection.return_value.filterDate.return_value.mosaic.return_value.select.return_value
    sample.reduceRegion.return_value.getInfo.return_value = {}
    startup = {'ee': fake_ee, 'new_map': Mock()}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(with_inputs(peek_source, **inputs), startup)
        exec(compile(paint_source, '<startup map>', 'exec'), startup)
    assert startup['PEEK_NAME'] == 'Seattle'
    assert startup['CENTER'] == list(expected)
    west, south, east, north = startup['region']
    assert west < expected[1] < east and south < expected[0] < north
    assert np.allclose([(south + north) / 2, (west + east) / 2], expected)
    fake_ee.Geometry.Rectangle.assert_called_once()
    fake_ee.Geometry.Point.assert_called_once_with(list(reversed(expected)))
    startup['new_map'].assert_called_once_with(list(expected), 9)
    sample.clip.assert_called_once_with(startup['region'])
print('PASS: fresh startup initializes region and map from default or edited PEEK inputs without preexisting CENTER.')

editor_source = next(c.source for c in main.cells if 'class PlacesEditor:' in c.source)
analysis_source = next(c.source for c in main.cells if 'Run me: which places are twins?' in c.source)
ns = {}
def run_editor():
    with patch('IPython.display.display'):
        exec(compile(editor_source, '<editor>', 'exec'), ns)
    return ns['places_editor']

editor = run_editor()
places = ns['PLACES']
def enter(name, lat, lon):
    editor.name.value, editor.latitude.value, editor.longitude.value = name, lat, lon
    editor.add.click()

assert len(places) == 8
assert next(iter(places.items())) == ('Seattle', (47.6062, -122.3321))
initial = places.copy()
enter('  Synthetic site  ', '12.5', '-45.25')
assert places['Synthetic site'] == (12.5, -45.25) and len(places) == 9
enter('Synthetic site', '-90', '180')
assert places['Synthetic site'] == (-90, 180) and len(places) == 9
enter('Boundary site', '90', '-180')
assert places['Boundary site'] == (90, -180)
for name, lat, lon in [('', '1', '2'), ('  ', '1', '2'), ('Invalid', 'x', '2'),
                       ('Invalid', '', '2'), ('Invalid', '1', 'x'),
                       ('Invalid', 'nan', '2'), ('Invalid', '1', 'NaN'),
                       ('Invalid', 'inf', '2'), ('Invalid', '1', '-inf'),
                       ('Invalid', '1e999', '2'), ('Invalid', '-91', '2'),
                       ('Invalid', '91', '2'), ('Invalid', '1', '-181'),
                       ('Invalid', '1', '181')]:
    before = places.copy()
    enter(name, lat, lon)
    assert places == before and editor.status.value
html_name = '<img src=x onerror="alert(1)"> & synthetic'
enter(html_name, '1', '2')
assert html_name in places
assert html_name not in editor.summary.value and '&lt;img' in editor.summary.value
assert html_name not in editor.status.value and '&lt;img' in editor.status.value
editor.selection.value = 'Synthetic site'
assert editor.name.value == 'Synthetic site' and editor.latitude.value == '-90.0'
editor.remove.click()
assert 'Synthetic site' not in places
editor.selection.value = html_name
editor.remove.click()
assert html_name not in places and '&lt;img' in editor.status.value
assert editor.remove.disabled
before = places.copy()
editor.remove.click()
assert places == before
old_editor = editor
editor = run_editor()
assert ns['PLACES'] is places and places == before
old_editor.reset.click()
assert places == before
print('PASS: add/update, selection/removal, finite/range/name validation, HTML escaping, and rerun persistence.')

# A full comparison with a local stand-in, never importing or calling Earth Engine.
requests = []
class Geometry:
    @staticmethod
    def Point(coords):
        requests.append(tuple(coords))
        return Geometry()
    def buffer(self, radius):
        assert radius == 250
        return self
class Reducer:
    @staticmethod
    def mean(): return 'mean'
class FakeEE:
    Geometry = Geometry
    Reducer = Reducer
    @staticmethod
    def Feature(geometry, properties): return properties
    @staticmethod
    def FeatureCollection(features): return features
class Sample:
    def reduceRegions(self, collection, reducer, scale, tileScale):
        self.features = collection
        assert scale == 20 and tileScale == 4
        return self
    def select(self, bands, retainGeometry): return self
    def getInfo(self):
        return {'features': [{'properties': {'name': item['name'], 'b0': 1.0,
                                              'b1': float(i + 1), 'b2': float(i % 3)}}
                             for i, item in enumerate(self.features)]}
ns.update(ee=FakeEE, embedding=lambda year: Sample(), YEAR=2024, BANDS=['b0','b1','b2'],
          np=np, pd=pd, plt=plt, HEAT='viridis', caption=lambda *args: None)
enter('Synthetic analysis site', '10', '20')
expected = places.copy()
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(analysis_source, '<analysis>', 'exec'), ns)
plt.close('all')
assert requests == [(lon, lat) for lat, lon in expected.values()]
assert list(ns['P'].index) == list(expected) and places == expected
# A second run must use an update, without re-running the editor.
enter('Synthetic analysis site', '30', '40')
requests.clear()
with contextlib.redirect_stdout(io.StringIO()):
    exec(compile(analysis_source, '<analysis>', 'exec'), ns)
plt.close('all')
assert (40.0, 30.0) in requests and (20.0, 10.0) not in requests
for name in list(places):
    if len(places) <= 1: break
    editor.selection.value = name
    editor.remove.click()
for count in (1, 0):
    assert len(places) == count
    requests.clear()
    try:
        exec(compile(analysis_source, '<analysis>', 'exec'), ns)
    except ValueError as exc:
        assert 'at least two' in str(exc)
    else:
        raise AssertionError('Missing minimum-place guard')
    assert not requests
    if places:
        editor.selection.value = next(iter(places))
        editor.remove.click()
editor = run_editor()
assert ns['PLACES'] is places and not places
editor.reset.click()
assert places == initial and ns['PLACES'] is places
try:
    exec(compile(analysis_source, '<analysis>', 'exec'), {})
except ValueError as exc:
    assert 'editor cell first' in str(exc)
else:
    raise AssertionError('Missing initialization guard')
print('PASS: comparison uses current edits; 0/1-place guards precede requests; empty rerun and reset work.')

# Query controls: edits do not apply, errors preserve state, reruns retain applied state.
query_source = next(c.source for c in main.cells if 'class QueryEditor:' in c.source)
search_source = next(c.source for c in main.cells if 'class PlacesSearch:' in c.source)
map_source = next(c.source for c in main.cells if 'Run me: pick a spot' in c.source)

def run_cell(source):
    with patch('IPython.display.display'), contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, '<controls>', 'exec'), ns)

def applied():
    return ns['QUERY_LAT'], ns['QUERY_LON']

run_cell(query_source)
query = ns['query_editor']
assert applied() == initial['Seattle']
for lat, lon in [('12.5', '-45.25'), ('-90', '180'), ('90', '-180'), ('0', '0')]:
    before = applied()
    query.latitude.value, query.longitude.value = lat, lon
    assert applied() == before
    query.apply.click()
    assert applied() == (float(lat), float(lon))
for lat, lon in [('', '2'), ('x', '2'), ('1', 'x'), ('nan', '2'), ('1', 'NaN'),
                 ('inf', '2'), ('1', '-inf'), ('1e999', '2'), ('-91', '2'),
                 ('91', '2'), ('1', '-181'), ('1', '181')]:
    before = applied()
    query.latitude.value, query.longitude.value = lat, lon
    query.apply.click()
    assert applied() == before and query.status.value
old_query = query
run_cell(query_source)
query = ns['query_editor']
assert applied() == (0, 0) and query.latitude.value == '0.0'
old_query.latitude.value = '80'
old_query.apply.click()
assert applied() == (0, 0)

# Stop at the map's first dependency: confirm it consumes applied coordinates on reruns.
class StopMap(Exception): pass
centers = []
def square_region(center):
    centers.append(tuple(center))
    raise StopMap
ns['square_region'] = square_region
for coords in [(10, 20), (-30, 40)]:
    query.set_applied(*coords)
    query.latitude.value = '55'  # Unapplied edits must not affect analysis.
    for _ in range(2):
        try: exec(compile(map_source, '<map>', 'exec'), ns)
        except StopMap: pass
        assert centers[-1] == coords and applied() == coords
try: exec(compile(map_source, '<map>', 'exec'), {})
except ValueError as exc: assert 'query editor' in str(exc)
else: raise AssertionError('Missing query initialization guard')
print('PASS: query validation, explicit apply, applied-state reruns, and map consumption.')

import types
import requests as http_requests
# Fail closed if any test accidentally tries a real HTTP request.
network_guard = patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden'))
network_guard.start()
secret = 'synthetic-secret-never-display'
secret_get = Mock(return_value=secret)
colab = types.ModuleType('google.colab')
colab.userdata = types.SimpleNamespace(get=secret_get)
google = types.ModuleType('google')
google.colab = colab
modules_guard = patch.dict(sys.modules, {'google': google, 'google.colab': colab})
modules_guard.start()
run_cell(search_source)
search = ns['places_search']
assert not secret_get.called
search.query.value = 'museums in Chicago'
assert not secret_get.called
assert search.selection.value is None and search.add.disabled and search.use.disabled

success = {'places': [
    {'displayName': {'text': '<b>Museum</b>'}, 'formattedAddress': 'Public & city',
     'location': {'latitude': 41.88, 'longitude': -87.62},
     'attributions': [{'provider': '<script>provider</script>', 'providerUri': 'javascript:unsafe'}]},
    {'displayName': {'text': 'Museum 2'}, 'formattedAddress': 'Chicago',
     'location': {'latitude': 41.89, 'longitude': -87.63}},
    {'location': {'latitude': 91, 'longitude': 0}},
]}
def response(status=200, payload=success, json_error=None):
    result = Mock(status_code=status)
    result.json.return_value = payload
    result.json.side_effect = json_error
    return result

def assert_clean():
    contents = search.status.value + search.results.value + repr(search.selection.options)
    assert secret not in contents
    assert not search.search.disabled and not search.query.disabled

def click_search(result=None, error=None):
    def post(url, **kwargs):
        assert search.search.disabled and search.query.disabled
        assert not search.candidates and not search.results.value
        assert search.selection.value is None and search.add.disabled and search.use.disabled
        assert url == 'https://places.googleapis.com/v1/places:searchText' and secret not in url
        assert kwargs['json'] == {'textQuery': search.query.value.strip(), 'pageSize': 5}
        assert kwargs['headers'] == {'X-Goog-Api-Key': secret,
            'X-Goog-FieldMask': 'places.displayName,places.formattedAddress,places.location,places.attributions'}
        assert kwargs['timeout'] == (5, 20) and kwargs['allow_redirects'] is False
        if error: raise error
        return result
    before_places, before_query = places.copy(), applied()
    with patch('requests.post', side_effect=post) as mocked, contextlib.redirect_stdout(io.StringIO()) as out, contextlib.redirect_stderr(io.StringIO()) as err:
        search.search.click()
    assert secret not in out.getvalue() + err.getvalue()
    assert places == before_places and applied() == before_query
    assert_clean()
    return mocked

click_search(response())
assert len(search.candidates) == 2
assert search.selection.value is None and search.add.disabled and search.use.disabled
assert 'Google Maps' in search.results.value and '&lt;script&gt;' in search.results.value
assert '<script>' not in search.results.value and '<b>Museum</b>' not in search.results.value
assert 'Public &amp; city' in search.results.value
assert '41.88' in search.results.value and '-87.62' in search.results.value
before_places, before_query = places.copy(), applied()
search.add.click(); search.use.click()
assert places == before_places and applied() == before_query
search.selection.value = 0
assert places == before_places and applied() == before_query
assert not search.add.disabled and not search.use.disabled
# Transfers resolve newly rerun editors and avoid overwriting a duplicate name.
run_editor()
run_cell(query_source)
search.add.click()
assert places['<b>Museum</b>'] == (41.88, -87.62)
assert '&lt;b&gt;Museum&lt;/b&gt;' in ns['places_editor'].summary.value
search.add.click()
assert places['<b>Museum</b> (2)'] == (41.88, -87.62)
search.use.click()
assert applied() == (41.88, -87.62) and ns['query_editor'].latitude.value == '41.88'
assert ns['places_editor'].selection.value == '<b>Museum</b> (2)'
search.selection.value = 1
assert applied() == (41.88, -87.62)
search.use.click()
assert applied() == (41.89, -87.63)
search.query.value = 'Chicago'
assert not search.candidates and not search.results.value and search.use.disabled
# Each failure must discard previous successful results and never expose upstream text.
failures = [
    (response(403), None, '403'), (response(429), None, '429'),
    (response(500), None, 'temporarily'), (response(503), None, 'temporarily'),
    (response(400), None, 'could not'), (response(302), None, 'could not'),
    (None, http_requests.exceptions.Timeout(secret), 'timed out'),
    (None, http_requests.exceptions.ConnectionError(secret), 'could not'),
    (None, RuntimeError(secret), 'could not'),
    (response(json_error=ValueError(secret)), None, 'unreadable'),
    (response(payload=[]), None, 'unexpected'),
    (response(payload={'places': None}), None, 'unexpected'),
    (response(payload={}), None, 'No results'),
    (response(payload={'places': []}), None, 'No results'),
    (response(payload={'places': [None, {}, {'location': {'latitude': float('nan'), 'longitude': 1}},
                                {'location': {'latitude': True, 'longitude': 0}},
                                {'location': {'latitude': 1, 'longitude': -181}},
                                {'location': {'latitude': 'oops', 'longitude': 1}}]}), None, 'valid coordinates'),
]
for result, error, message in failures:
    click_search(response())
    search.selection.value = 0
    click_search(result, error)
    assert not search.candidates and not search.results.value and search.selection.value is None
    assert search.add.disabled and search.use.disabled and message in search.status.value
for returned, error in [(None, None), ('', None), ('  ', None), (None, RuntimeError(secret))]:
    click_search(response())
    secret_get.return_value, secret_get.side_effect = returned, error
    mocked = click_search(response())
    assert not mocked.called and 'Secrets' in search.status.value and not search.candidates
    secret_get.return_value, secret_get.side_effect = secret, None
# Redact and escape even malicious successful text payloads.
malicious = {'places': [{'displayName': {'text': secret}, 'formattedAddress': secret,
                        'location': {'latitude': 1, 'longitude': 2},
                        'attributions': [{'provider': secret, 'providerUri': secret}]}]}
click_search(response(payload=malicious))
assert '[redacted]' in search.results.value
before_calls = secret_get.call_count
search.query.value = '   '
assert not click_search(response()).called and secret_get.call_count == before_calls
search.query.value = 'Chicago'
click_search(response())
search.selection.value = 0
old_search = search
before_places, before_query = places.copy(), applied()
run_cell(search_source)
search = ns['places_search']
old_search.add.click(); old_search.use.click(); old_search.search.click()
assert places == before_places and applied() == before_query
assert not search.candidates and search.selection.value is None
assert not ns['query_editor'].apply.disabled  # Manual path still available.
ns['query_editor'].set_applied(10, 20)
assert applied() == (10, 20)
modules_guard.stop()
network_guard.stop()
print('PASS: mocked Places request contract, errors, secret redaction, attribution escaping, explicit transfers, and reruns.')

# Public fixtures only: tests must work from a checkout or ZIP without Git history.
assert initial == {
    'Seattle': (47.6062, -122.3321), 'Manhattan': (40.758, -73.9855),
    'Iowa cropland': (42.03, -93.6), 'Everglades': (25.75, -80.8),
    'Sahara, Algeria': (26.0, 2.0), 'Amazon, Para': (-3.0, -54.0),
    'Icelandic highland': (64.7, -18.5), 'Nile Delta': (30.9, 31.2),
}
assert main.metadata.colab.omit_outputs is True and 'widgets' not in main.metadata
for cell in main.cells:
    assert not any(k in cell.metadata for k in ('widgets', 'outputId', 'executionInfo'))
    if cell.cell_type == 'code': assert not cell.outputs and cell.execution_count is None
assert any('**Teodora**' in c.source for c in main.cells)
assert 'blob/main/alpha_earth_embeddings_demo.ipynb' in main.cells[0].source
for controller in ('places_editor', 'query_editor', 'places_search'):
    ns[controller].close()
print('PASS: public defaults, clean outputs/widgets, authorship, and launch link.')
