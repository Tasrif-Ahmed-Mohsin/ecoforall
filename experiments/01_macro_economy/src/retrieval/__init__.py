"""FAISS-backed retrieval of similar historical country-year situations."""
from .faiss_index import FaissIndex, build_or_load, query_topk  # noqa: F401