"""ARGOS federated segmentation for vantage6.

The vantage6 wrapper imports this package and looks the requested function up by name, so the
algorithm functions are exposed at package level. They are resolved on first access rather than
at import time, so that modules without TensorFlow (``paths``, ``preprocess``, ``sort_slices``)
can be used and tested without loading it.
"""

import importlib

# _FUNCTION_MODULES = (".central", ".partial")


# def __getattr__(name: str):
#     for module_name in _FUNCTION_MODULES:
#         module = importlib.import_module(module_name, __name__)
#         if hasattr(module, name):
#             return getattr(module, name)
#     raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

_FUNCTIONS = {"central": ".central", "test_central": ".central", "test_gpu": ".central",
                "train_locally": ".partial", "test_train_locally": ".partial",
                "pass_weights_back": ".partial"}
def __getattr__(name):
    if name in _FUNCTIONS:
        return getattr(importlib.import_module(_FUNCTIONS[name], __name__), name)
    raise AttributeError(name)