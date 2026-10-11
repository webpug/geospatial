#@title ▶️ Optional: set a runtime Google Maps API key
import getpass
import warnings


def set_runtime_maps_key():
    global _maps_runtime_api_key
    # Clear any previous key before prompting, including on cancellation or failure.
    _maps_runtime_api_key = None
    try:
        with warnings.catch_warnings():
            # Stop getpass before its fallback can read input with echo enabled.
            warnings.simplefilter("error", getpass.GetPassWarning)
            _maps_runtime_api_key = getpass.getpass(
                "GOOGLE_MAPS_API_KEY (hidden; blank clears runtime key and uses Secrets): "
            ).strip() or None
    except getpass.GetPassWarning:
        _maps_runtime_api_key = None
        print("Hidden input is unavailable. Runtime key cleared; use Colab Secrets with notebook access.")
    except (KeyboardInterrupt, EOFError):
        _maps_runtime_api_key = None
        print("Key setup cancelled. Runtime key cleared; Search will use Colab Secrets.")
    except Exception:
        _maps_runtime_api_key = None
        print("Key setup could not complete. Runtime key cleared; use Colab Secrets with notebook access.")
    else:
        if _maps_runtime_api_key:
            print("Runtime key set. Run the Maps search cell, then click Search.")
        else:
            print("Runtime key cleared. Search will use Colab Secrets with notebook access.")


