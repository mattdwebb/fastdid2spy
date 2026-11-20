"""Independent static Gardner two-stage DiD implementation."""
__version__ = "0.1.0"
from .api import Design
from ._inference import InferenceError, assignment_draws
from ._fe import FEAbsorber
