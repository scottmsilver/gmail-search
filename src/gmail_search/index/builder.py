import json
import logging
import math
import struct
from pathlib import Path

import numpy as np
import scann

from gmail_search.store.db import get_connection

logger = logging.getLogger(__name__)


def build_index(db_path: Path, index_dir: Path, model: str, dimensions: int) -> None:
    conn = get_connection(db_path)
    rows = conn.execute("SELECT id, embedding FROM embeddings WHERE model = ?", (model,)).fetchall()
    conn.close()

    if not rows:
        logger.warning("No embeddings found. Skipping index build.")
        index_dir.mkdir(parents=True, exist_ok=True)
        (index_dir / "ids.json").write_text("[]")
        return

    ids = [r["id"] for r in rows]
    vectors = np.array(
        [list(struct.unpack(f"{dimensions}f", r["embedding"])) for r in rows],
        dtype=np.float32,
    )

    logger.info(f"Building ScaNN index with {len(ids)} vectors, {dimensions} dims")

    num_leaves = max(int(math.sqrt(len(ids))), 1)
    num_leaves = min(num_leaves, len(ids))

    builder = scann.scann_ops_pybind.builder(vectors, 10, "dot_product")

    if len(ids) >= 100:
        builder = builder.tree(
            num_leaves=num_leaves,
            num_leaves_to_search=min(num_leaves, 100),
            training_sample_size=min(len(ids), 250000),
        )
        builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
        builder = builder.reorder(100)
    else:
        builder = builder.score_brute_force()

    searcher = builder.build()

    index_dir.mkdir(parents=True, exist_ok=True)
    searcher.serialize(str(index_dir))
    (index_dir / "ids.json").write_text(json.dumps(ids))

    logger.info(f"Index built and saved to {index_dir}")


def load_index_metadata(index_dir: Path) -> list[int]:
    ids_file = index_dir / "ids.json"
    if not ids_file.exists():
        return []
    return json.loads(ids_file.read_text())
