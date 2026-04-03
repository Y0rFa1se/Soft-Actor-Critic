import importlib.util


def _check_dependencies():
    _REQUIRED = ["numpy", "torch"]

    for spec in _REQUIRED:
        if not importlib.util.find_spec(spec):
            raise ImportError(f"Package '{spec}' is not installed.")
