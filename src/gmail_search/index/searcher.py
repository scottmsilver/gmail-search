import json
import logging
from pathlib import Path

import numpy as np
import scann

logger = logging.getLogger(__name__)


class ScannSearcher:
    def __init__(self, index_dir: Path, dimensions: int):
        self.index_dir = index_dir
        self.dimensions = dimensions

        ids_file = index_dir / "ids.json"
        if not ids_file.exists():
            raise FileNotFoundError(f"Index not found at {index_dir}. Run 'gmail-search reindex' first.")

        self.embedding_ids: list[int] = json.loads(ids_file.read_text())

        if not self.embedding_ids:
            self.searcher = None
            logger.warning("Empty index loaded")
            return

        self.searcher = scann.scann_ops_pybind.load_searcher(str(index_dir))
        logger.info(f"Loaded ScaNN index with {len(self.embedding_ids)} vectors")

    def search(self, query_vector: np.ndarray, top_k: int = 20) -> tuple[list[int], list[float]]:
        if self.searcher is None or not self.embedding_ids:
            return [], []

        neighbors, distances = self.searcher.search(query_vector, final_num_neighbors=top_k)

        embedding_ids = [self.embedding_ids[i] for i in neighbors]
        scores = distances.tolist()

        return embedding_ids, scores
