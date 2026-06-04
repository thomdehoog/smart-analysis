"""cluster_objects -- cluster object embeddings and write clustering artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _contracts import to_builtin, validate_overview  # noqa: E402


METADATA = {
    "description": "Cluster object embeddings with cosine kNN, Leiden, and UMAP",
    "version": "1.0",
    "max_workers": 1,
    "environment": "SMART--target_discovery--cluster",
}


def run(pipeline_data: dict, state: dict, **params) -> dict:
    """Cluster all objects in an overview and enrich their object tables."""
    inp = pipeline_data["input"]
    overview = validate_overview({"tiles": inp["tiles"]})

    output_dir = _setting(inp, params, "output_dir", None)
    n_neighbors = int(_setting(inp, params, "n_neighbors", 15))
    resolution = float(_setting(inp, params, "leiden_resolution", 1.0))
    random_state = int(_setting(inp, params, "random_state", 0))
    min_dist = float(_setting(inp, params, "umap_min_dist", 0.1))

    rows, vectors = _collect_embedding_rows(overview["tiles"])
    cluster_ids, umap_xy = _cluster_vectors(
        vectors,
        n_neighbors=n_neighbors,
        resolution=resolution,
        random_state=random_state,
        min_dist=min_dist,
    )

    for idx, row in enumerate(rows):
        row["cluster_id"] = int(cluster_ids[idx])
        row["umap_x"] = float(umap_xy[idx][0])
        row["umap_y"] = float(umap_xy[idx][1])

    _write_back_columns(overview["tiles"], rows)
    artifacts = _write_artifacts(rows, output_dir)

    clusters = to_builtin({
        "n_objects": len(rows),
        "n_clusters": len(set(cluster_ids)) if cluster_ids else 0,
        "method": "cosine_knn_leiden_umap",
        "n_neighbors": n_neighbors,
        "leiden_resolution": resolution,
        "random_state": random_state,
        "table": rows,
        "artifacts": artifacts,
    })

    result = dict(pipeline_data.get("target_discovery", {}))
    result["clusters"] = clusters
    pipeline_data["target_discovery"] = result
    pipeline_data["input"]["tiles"] = overview["tiles"]
    return pipeline_data


def _setting(inp: dict, params: dict, key: str, default):
    return inp[key] if key in inp else params.get(key, default)


def _collect_embedding_rows(tiles: list[dict]) -> tuple[list[dict], list[list[float]]]:
    rows = []
    vectors = []
    for tile_index, tile in enumerate(tiles):
        objects = tile["objects"]
        props = objects["properties"]
        embeddings = objects.get("embeddings")
        if not embeddings or "vectors" not in embeddings:
            raise ValueError("cluster_objects requires objects.embeddings.vectors.")
        labels = embeddings.get("label", props["label"])
        if labels != props["label"]:
            raise ValueError("objects.embeddings.label must align to properties.label.")

        for row_index, vector in enumerate(embeddings["vectors"]):
            if not vector:
                raise ValueError("Embedding vectors must be non-empty.")
            vectors.append([float(value) for value in vector])
            rows.append(_object_row(tile, tile_index, row_index))
    return rows, vectors


def _object_row(tile: dict, tile_index: int, row_index: int) -> dict:
    props = tile["objects"]["properties"]
    geometry = tile["geometry"]
    label = int(props["label"][row_index])
    return {
        "tile_index": int(tile_index),
        "row_index": int(row_index),
        "object_id": props.get("object_id", [None] * tile["objects"]["n_objects"])[
            row_index
        ],
        "tile_id": list(geometry["tile_id"]),
        "object_label": label,
        "stage_x_um": float(props["stage_x_um"][row_index]),
        "stage_y_um": float(props["stage_y_um"][row_index]),
        "centroid_row_px": float(props["centroid_row_px"][row_index]),
        "centroid_col_px": float(props["centroid_col_px"][row_index]),
        "area": float(props["area"][row_index]),
        "intensity_mean": float(props["intensity_mean"][row_index]),
        "eccentricity": float(props["eccentricity"][row_index]),
    }


def _cluster_vectors(
    vectors: list[list[float]],
    *,
    n_neighbors: int,
    resolution: float,
    random_state: int,
    min_dist: float,
) -> tuple[list[int], list[list[float]]]:
    n_objects = len(vectors)
    if n_objects == 0:
        return [], []
    if n_objects == 1:
        return [0], [[0.0, 0.0]]
    if n_objects == 2:
        return [0, 0], [[0.0, 0.0], [1.0, 0.0]]

    import igraph
    import leidenalg
    import numpy as np
    import umap
    from sklearn.neighbors import NearestNeighbors

    x = np.asarray(vectors, dtype=float)
    k = max(1, min(int(n_neighbors), n_objects - 1))
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(x)
    distances, indices = nn.kneighbors(x)

    edge_weights: dict[tuple[int, int], float] = {}
    for src in range(n_objects):
        for dist, dst in zip(distances[src][1:], indices[src][1:]):
            a, b = sorted((int(src), int(dst)))
            weight = max(0.0, 1.0 - float(dist))
            edge_weights[(a, b)] = max(edge_weights.get((a, b), 0.0), weight)

    graph = igraph.Graph(n=n_objects, edges=list(edge_weights))
    graph.es["weight"] = list(edge_weights.values())
    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=random_state,
    )
    cluster_ids = [0] * n_objects
    for cluster_id, members in enumerate(partition):
        for member in members:
            cluster_ids[int(member)] = int(cluster_id)

    reducer = umap.UMAP(
        n_neighbors=k,
        min_dist=min_dist,
        metric="cosine",
        random_state=random_state,
    )
    umap_xy = reducer.fit_transform(x).astype(float).tolist()
    return cluster_ids, umap_xy


def _write_back_columns(tiles: list[dict], rows: list[dict]) -> None:
    for tile in tiles:
        n = tile["objects"]["n_objects"]
        props = tile["objects"]["properties"]
        props["cluster_id"] = [None] * n
        props["umap_x"] = [None] * n
        props["umap_y"] = [None] * n

    for row in rows:
        props = tiles[row["tile_index"]]["objects"]["properties"]
        idx = row["row_index"]
        props["cluster_id"][idx] = row["cluster_id"]
        props["umap_x"][idx] = row["umap_x"]
        props["umap_y"][idx] = row["umap_y"]


def _write_artifacts(rows: list[dict], output_dir) -> dict:
    if output_dir is None:
        return {}
    features_dir = Path(output_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    csv_path = features_dir / "clusters.csv"
    json_path = features_dir / "clusters.json"
    svg_path = features_dir / "clusters_umap.svg"

    _write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin({"objects": rows}), handle, indent=2, allow_nan=False)
        handle.write("\n")
    _write_svg(svg_path, rows)
    return {
        "cluster_table_csv": str(csv_path),
        "cluster_table_json": str(json_path),
        "cluster_plot_svg": str(svg_path),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    columns = (
        "object_id",
        "tile_id",
        "object_label",
        "stage_x_um",
        "stage_y_um",
        "centroid_row_px",
        "centroid_col_px",
        "area",
        "intensity_mean",
        "eccentricity",
        "cluster_id",
        "umap_x",
        "umap_y",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["tile_id"] = json.dumps(row["tile_id"], separators=(",", ":"))
            writer.writerow(out)


def _write_svg(path: Path, rows: list[dict]) -> None:
    width, height, pad = 800, 600, 48
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    points = [
        (float(row["umap_x"]), float(row["umap_y"]), int(row["cluster_id"]), row)
        for row in rows
    ]
    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
    else:
        min_x = max_x = min_y = max_y = 0.0

    def scale(value, lo, hi, out_lo, out_hi):
        if hi == lo:
            return (out_lo + out_hi) / 2.0
        return out_lo + ((value - lo) / (hi - lo)) * (out_hi - out_lo)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad}" y="32" font-family="Arial" font-size="18">'
        "UMAP clusters</text>",
    ]
    for x, y, cluster_id, row in points:
        cx = scale(x, min_x, max_x, pad, width - pad)
        cy = scale(y, min_y, max_y, height - pad, pad)
        color = colors[cluster_id % len(colors)]
        label = row.get("object_id") or f"{row['tile_id']}:{row['object_label']}"
        elements.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5" fill="{color}" '
            f'stroke="#222" stroke-width="0.75">'
            f"<title>{label} cluster={cluster_id}</title></circle>"
        )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")
