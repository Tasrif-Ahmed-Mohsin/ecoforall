"""Harmonization package — turns each ecodata source into a canonical long parquet."""
from . import gmd  # noqa: F401
# GMD-only clone. To fold the old 5-source panel back in, add the other
# modules here: `from . import clio_infra, imf, jst, maddison, wb`.