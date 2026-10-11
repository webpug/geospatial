#@title ▶️ Optional: search for coordinates with Google Maps
import requests



def search_place_candidates(text_query):
    # Called only by Search. The optional runtime key takes precedence over Secrets.
    api_key = globals().get("_maps_runtime_api_key")
    guidance = ("Set GOOGLE_MAPS_API_KEY in Colab Secrets with notebook access (preferred), "
                "or run the optional runtime Google Maps API key setup cell immediately above search.")
    if not isinstance(api_key, str) or not api_key.strip():
        try:
            from google.colab import userdata
            api_key = userdata.get("GOOGLE_MAPS_API_KEY")
        except Exception:
            return [], guidance
    if not isinstance(api_key, str) or not api_key.strip():
        return [], guidance
    api_key = api_key.strip()

    try:
        response = requests.post(
            "https://places.googleapis.com/v1/places:searchText",
            json={"textQuery": text_query, "pageSize": 5},
            headers={"X-Goog-Api-Key": api_key,
                     "X-Goog-FieldMask": "places.displayName,places.formattedAddress,places.location,places.attributions"},
            timeout=(5, 20), allow_redirects=False,
        )
        if response.status_code == 403:
            return [], "Search denied (403). Check billing, Places API (New), and key restrictions."
        if response.status_code == 429:
            return [], "Search quota reached (429). Wait and check your quota before retrying."
        if response.status_code >= 500:
            return [], "Google Maps is temporarily unavailable. Try again later."
        if response.status_code != 200:
            return [], "Search could not complete. Check your API setup and try again."
        try:
            payload = response.json()
        except ValueError:
            return [], "Search returned an unreadable response. Try again later."
        if not isinstance(payload, dict) or not isinstance(payload.get("places", []), list):
            return [], "Search returned an unexpected response. Try again later."

        def clean_text(value):
            # Also redact a secret if a malformed service response echoes it.
            return value.replace(api_key, "[redacted]") if isinstance(value, str) else ""

        candidates = []
        for place in payload.get("places", [])[:5]:
            if not isinstance(place, dict):
                continue
            location = place.get("location")
            if not isinstance(location, dict):
                continue
            try:
                lat, lon = query_coordinates(location.get("latitude"), location.get("longitude"))
            except ValueError:
                continue
            display_name = place.get("displayName")
            name = clean_text(display_name.get("text")) if isinstance(display_name, dict) else ""
            address = clean_text(place.get("formattedAddress"))
            attributions = []
            providers = place.get("attributions", [])
            if isinstance(providers, list):
                for provider in providers:
                    if isinstance(provider, dict):
                        attribution = " — ".join(filter(None, [clean_text(provider.get("provider")),
                                                                 clean_text(provider.get("providerUri"))]))
                        if attribution:
                            attributions.append(attribution)
            candidates.append({"name": name or address or "Unnamed place", "address": address,
                               "lat": lat, "lon": lon, "attributions": attributions})
        if not candidates:
            message = ("No results. Try a more specific place or city." if not payload.get("places")
                       else "No results with valid coordinates. Try another search or enter coordinates manually.")
            return [], message
        return candidates, "Choose a candidate, then click a transfer button."
    except requests.exceptions.Timeout:
        return [], "Search timed out. Try again later."
    except Exception:
        # Never expose upstream exception text, response bodies, headers, or the key.
        return [], "Search could not complete. Check your connection and API setup."
    finally:
        api_key = None


class PlacesSearch:
    def __init__(self):
        self.candidates = []
        self.query = widgets.Text(description="Find:", placeholder="City, address, business, or description",
                                  layout=widgets.Layout(width="95%"))
        self.search = widgets.Button(description="Search", button_style="primary")
        self.selection = widgets.Dropdown(description="Candidate:", options=[("Choose a candidate…", None)],
                                           layout=widgets.Layout(width="95%"))
        self.add = widgets.Button(description="Add selected to PLACES", layout=widgets.Layout(width="auto"))
        self.use = widgets.Button(description="Use as chip center")
        self.status = widgets.HTML()
        self.results = widgets.HTML()
        self.search.on_click(self._search)
        self.selection.observe(self._select, names="value")
        self.query.observe(self._edited, names="value")
        self.add.on_click(self._add)
        self.use.on_click(self._use)
        self.view = widgets.VBox([self.query, self.search, self.status, self.results,
                                  self.selection, widgets.HBox([self.add, self.use])])
        self._clear()

    def _clear(self):
        self.candidates = []
        self.selection.options = [("Choose a candidate…", None)]
        self.selection.value = None
        self.add.disabled = self.use.disabled = True
        self.results.value = ""

    def _edited(self, _):
        self._clear()
        self.status.value = "Click Search to look up this query."

    def _select(self, _):
        self.add.disabled = self.use.disabled = self.selection.value is None

    def _search(self, _):
        if self.search.disabled:
            return
        self._clear()
        text_query = self.query.value.strip()
        if not text_query:
            self.status.value = "Enter a place or description to search."
            return
        self.search.disabled = self.query.disabled = True
        self.status.value = "Searching…"
        try:
            candidates, message = search_place_candidates(text_query)
            self.candidates = candidates
            self.status.value = escape(message)
            rows = []
            options = [("Choose a candidate…", None)]
            for i, candidate in enumerate(candidates):
                label = (f"{candidate['name']} — {candidate['address']} "
                         f"({candidate['lat']}, {candidate['lon']})")
                options.append((label, i))
                providers = "; ".join(candidate["attributions"])
                rows.append(f"<li>{escape(label)}<br>{escape(providers)}</li>")
            if rows:
                self.results.value = (
                    '<div style="border:1px solid #ccc;padding:10px">'
                    '<span translate="no" style="font:400 14px sans-serif;white-space:nowrap;'
                    'color:#1f1f1f;background:white">Google Maps</span><ol>'
                    + "".join(rows) + "</ol></div>"
                )
            self.selection.options = options
            self.selection.value = None
        except Exception:
            self._clear()
            self.status.value = "Search could not complete. Try again later."
        finally:
            self.search.disabled = self.query.disabled = False

    def _candidate(self):
        index = self.selection.value
        if index is None or not 0 <= index < len(self.candidates):
            self.status.value = "Choose a candidate first."
            return None
        return self.candidates[index]

    def _add(self, _):
        candidate = self._candidate()
        if candidate is None:
            return
        # Resolve the current editor at click time, even after its cell is rerun.
        name = candidate["name"]
        base, suffix = name, 2
        while name in PLACES:
            name = f"{base} ({suffix})"
            suffix += 1
        places_editor.name.value = name
        places_editor.latitude.value = str(candidate["lat"])
        places_editor.longitude.value = str(candidate["lon"])
        places_editor._add(None)
        self.status.value = places_editor.status.value

    def _use(self, _):
        candidate = self._candidate()
        if candidate is None:
            return
        lab.set_applied(candidate["lat"], candidate["lon"])
        self.status.value = "Chip center applied. Run chip acquisition to download imagery."

    def close(self):
        self.search.on_click(self._search, remove=True)
        self.add.on_click(self._add, remove=True)
        self.use.on_click(self._use, remove=True)
        self.selection.unobserve(self._select, names="value")
        self.query.unobserve(self._edited, names="value")
        for widget in (self.add, self.use, *self.view.children, self.view):
            widget.close()

