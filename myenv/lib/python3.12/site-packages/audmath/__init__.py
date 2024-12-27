from audmath.core.api import db
from audmath.core.api import duration_in_seconds
from audmath.core.api import inverse_db
from audmath.core.api import inverse_normal_distribution
from audmath.core.api import rms
from audmath.core.api import samples
from audmath.core.api import similarity
from audmath.core.api import window


# Discourage from audmath import *
__all__ = []

# Dynamically get the version of the installed module
try:
    import importlib.metadata

    __version__ = importlib.metadata.version(__name__)
except Exception:  # pragma: no cover
    importlib = None  # pragma: no cover
finally:
    del importlib
