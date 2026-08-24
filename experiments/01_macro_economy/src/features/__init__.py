"""Feature engineering: pivot long parquets → ML-ready wide panel."""
from .build_panel import CORE_TARGETS, GDP_PC_CANDIDATES, build, load_long  # noqa: F401