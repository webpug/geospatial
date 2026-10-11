"""Local controls; Earth Engine is called only by the explicitly run acquisition cells."""
from html import escape
import ipywidgets as widgets
from IPython.display import display, clear_output


class PlacesEditor:
    def __init__(self, places, changed=lambda: None):
        self.places, self.changed = places, changed
        self.name = widgets.Text(description='Name:')
        self.latitude = widgets.Text(description='Latitude:')
        self.longitude = widgets.Text(description='Longitude:')
        self.selection = widgets.Dropdown(description='Saved place:')
        self.add = widgets.Button(description='Add / update', button_style='primary')
        self.remove = widgets.Button(description='Remove selected')
        self.reset = widgets.Button(description='Reset defaults')
        self.status, self.summary = widgets.HTML(), widgets.HTML()
        self.selection.observe(self._select, names='value')
        self.add.on_click(self._add)
        self.remove.on_click(self._remove)
        self.reset.on_click(self._reset)
        self.view = widgets.VBox([self.selection, self.name, self.latitude, self.longitude,
                                 widgets.HBox([self.add, self.remove, self.reset]), self.status, self.summary])
        self._refresh()

    def _refresh(self, selected=None):
        self.selection.options = [('Choose a place…', None)] + [(n, n) for n in self.places]
        self.selection.value = selected
        self.remove.disabled = selected is None
        self.summary.value = f'<b>{len(self.places)} / {MAX_PLACES} places</b><ul>' + ''.join(
            f'<li>{escape(n)}: {lat}, {lon}</li>' for n, (lat, lon) in self.places.items()) + '</ul>'

    def _select(self, change):
        name = change['new']
        self.remove.disabled = name is None
        if name is not None:
            lat, lon = self.places[name]
            self.name.value, self.latitude.value, self.longitude.value = name, str(lat), str(lon)

    def _add(self, _):
        try:
            name = self.name.value.strip()
            coords = query_coordinates(self.latitude.value, self.longitude.value)
            candidate = {**self.places, name: coords}
            validate_places(candidate, minimum=1)
        except ValueError as exc:
            self.status.value = escape(str(exc))
            return
        self.places[name] = coords
        self.changed()
        self._refresh(name)
        self.status.value = escape(f'Saved {name}. Run acquisition to update results.')

    def _remove(self, _):
        name = self.selection.value
        if name is not None:
            del self.places[name]
            self.changed()
            self._refresh()
            self.status.value = escape(f'Removed {name}. Run acquisition to update results.')

    def _reset(self, _):
        self.places.clear()
        self.places.update(DEFAULT_PLACES)
        self.changed()
        self._refresh()
        self.status.value = 'Restored four public examples. Run acquisition to update results.'


class LabControls:
    def __init__(self, places, comparison=False):
        self.places, self.comparison = places, comparison
        self.pipeline = Pipeline()
        self.connected_project = None
        self.chip = self.place_results = None
        self.chip_key = self.places_key = None
        self.project = widgets.Text(description='EE project:', placeholder='Your own project ID (required)')
        self.year = widgets.IntText(value=2024, description='Year:')
        self.size = widgets.Dropdown(options=[32, 64, 128], value=64, description='Chip pixels:')
        self.model = widgets.Dropdown(options=[('V1 Nano · lightweight', 'OLMOEARTH_V1_NANO'),
                                              ('V1 Tiny', 'OLMOEARTH_V1_TINY'),
                                              ('V1 Base · more memory', 'OLMOEARTH_V1_BASE')],
                                      description='Pretrained:')
        self.latitude = widgets.Text(value='47.6062', description='Chip latitude:')
        self.longitude = widgets.Text(value='-122.3321', description='Chip longitude:')
        self.status = widgets.HTML(value='Edit locally, then run authentication and model cells below.')
        self.editor = PlacesEditor(places, self.invalidate)
        self.use_place = widgets.Button(description='Use selected as chip', layout=widgets.Layout(width='auto'))
        self.use_place.on_click(self._use_place)
        for widget in (self.project, self.year, self.size, self.model, self.latitude, self.longitude):
            widget.observe(self.invalidate, names='value')
        self.view = widgets.VBox([
            widgets.HTML('<h3>Explore a 640 m Seattle neighborhood</h3><p>64 × 64 inputs → 16 × 16 output cells. Edits stay local.</p>'),
            self.project, widgets.HBox([self.year, self.size]), self.model,
            widgets.HBox([self.latitude, self.longitude]), self.status,
            widgets.HTML('<h4>PLACES · compare up to eight neighborhoods</h4>'), self.editor.view, self.use_place])

    def invalidate(self, *_):
        self.chip = self.place_results = None
        self.chip_key = self.places_key = None
        self.status.value = 'Inputs changed. Previous plots are snapshots; rerun acquisition before using results.'

    def _use_place(self, _):
        name = self.editor.selection.value
        if name is not None:
            self.set_applied(*self.places[name])

    def set_applied(self, lat, lon):
        lat, lon = query_coordinates(lat, lon)
        self.latitude.value, self.longitude.value = str(lat), str(lon)

    def settings(self):
        return Settings(self.project.value, self.year.value, self.size.value, self.model.value)

    def snapshot(self):
        return (self.settings(), query_coordinates(self.latitude.value, self.longitude.value),
                tuple((n, *coords) for n, coords in self.places.items()), self.comparison)

    def authenticate(self, ee):
        self.invalidate()
        self.connected_project = None
        settings = self.settings()  # Blank fails before auth.
        ee.Authenticate()
        ee.Initialize(project=settings.project)
        self.connected_project = settings.project
        self.status.value = 'Connected. Run the pretrained model load cell next.'

    def load_model(self):
        self.invalidate()
        self.pipeline.load(self.settings())
        print(f'Loaded real pretrained {self.settings().model} on {self.pipeline.device}.')
        print(f'Model loading: {self.pipeline.load_seconds:.1f} s (reported separately from data pipelines).')

    def _ready(self):
        settings = self.settings()
        if self.connected_project != settings.project:
            raise ValueError('Run authentication with this project ID first.')
        return settings

    def acquire_chip(self, ee):
        self.chip = self.chip_key = None
        settings = self._ready()
        key = self.snapshot()
        result = self.pipeline.run(ee, settings, *key[1], comparison=self.comparison)
        self.chip, self.chip_key = result, key
        self.status.value = 'Chip ready. Run the panels and local spatial query cells.'
        return result

    def acquire_places(self, ee):
        self.place_results = self.places_key = None
        settings = self._ready()
        entries = validate_places(self.places)
        key = self.snapshot()
        results = {}
        for index, (name, lat, lon) in enumerate(entries, 1):
            print(f'Acquiring place {index}/{len(entries)} …')
            results[name] = self.pipeline.run(ee, settings, lat, lon, comparison=self.comparison)
        self.place_results, self.places_key = results, key  # Publish only complete runs.
        self.status.value = 'All places ready. Choose a ranking query below.'
        return results

    def require_chip(self):
        if self.chip is None or self.chip_key != self.snapshot():
            raise ValueError('No current chip. Run acquisition for the current inputs first.')
        return self.chip

    def require_places(self):
        if self.place_results is None or self.places_key != self.snapshot():
            raise ValueError('No current place results. Run place acquisition for the current inputs first.')
        return self.place_results


def spatial_query_controls(lab):
    result = lab.require_chip()
    n = result['grid']['dimensions']['width']//PATCH
    mode = widgets.Dropdown(options=['Token row / column', 'Latitude / longitude'], description='Query by:')
    row = widgets.BoundedIntText(value=n//2, min=0, max=n-1, description='Token row:')
    column = widgets.BoundedIntText(value=n//2, min=0, max=n-1, description='Token column:')
    latitude = widgets.Text(value=lab.latitude.value, description='Latitude:')
    longitude = widgets.Text(value=lab.longitude.value, description='Longitude:')
    button = widgets.Button(description='Show similarity', button_style='primary')
    output = widgets.Output()
    def show(_):
        with output:
            clear_output(wait=True)
            try:
                current = lab.require_chip()
                if mode.value == 'Token row / column':
                    r, c = spatial_query(current['grid'], row=row.value, column=column.value)
                else:
                    r, c = spatial_query(current['grid'], latitude.value, longitude.value)
                if not current['shared_valid'][r, c]:
                    raise ValueError('Query cell is missing. Choose a valid cell from the coverage panel.')
                paint_query(current, r, c)
            except ValueError as exc:
                print(str(exc))
    button.on_click(show)
    view = widgets.VBox([mode, widgets.HBox([row, column]), widgets.HBox([latitude, longitude]), button, output])
    display(view)
    return view


def ranking_controls(lab):
    results = lab.require_places()
    query = widgets.Dropdown(options=list(results), description='Query place:')
    button = widgets.Button(description='Compare neighbors', button_style='primary')
    output = widgets.Output()
    def show(_):
        with output:
            clear_output(wait=True)
            try:
                compare_places(lab.require_places(), query.value)
            except ValueError as exc:
                print(str(exc))
    button.on_click(show)
    view = widgets.VBox([query, button, output])
    display(view)
    return view
