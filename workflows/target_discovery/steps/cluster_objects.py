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
    umap_n_neighbors = int(_setting(inp, params, "umap_n_neighbors", 30))
    min_dist = float(_setting(inp, params, "umap_min_dist", 0.4))
    umap_metric = _resolve_umap_metric(_setting(inp, params, "umap_metric", "euclidean"))
    cluster_mode = str(_setting(inp, params, "cluster_mode", "manual")).lower()

    rows, vectors = _collect_embedding_rows(overview["tiles"])
    resolution_sweep = []
    selected_resolution = resolution
    if cluster_mode == "auto":
        auto = _auto_select_resolution(
            vectors,
            n_neighbors=n_neighbors,
            random_state=random_state,
            initial_grid=_float_list(
                _setting(
                    inp,
                    params,
                    "auto_resolution_grid",
                    [0.05, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0],
                )
            ),
            refine_steps=int(_setting(inp, params, "auto_refine_steps", 2)),
            stability_repeats=int(_setting(inp, params, "auto_stability_repeats", 3)),
            max_cluster_fraction=float(_setting(inp, params, "auto_max_cluster_fraction", 0.95)),
            min_cluster_size=int(_setting(inp, params, "auto_min_cluster_size", 2)),
            max_tiny_cluster_fraction=float(
                _setting(inp, params, "auto_max_tiny_cluster_fraction", 0.30)
            ),
        )
        selected_resolution = float(auto["selected_resolution"])
        resolution_sweep = auto["sweep"]
    elif cluster_mode != "manual":
        raise ValueError("cluster_mode must be 'manual' or 'auto'.")

    cluster_ids, umap_xy, effective_n_neighbors, effective_umap_n_neighbors = _cluster_vectors(
        vectors,
        n_neighbors=n_neighbors,
        umap_n_neighbors=umap_n_neighbors,
        resolution=selected_resolution,
        random_state=random_state,
        min_dist=min_dist,
        umap_metric=umap_metric,
    )

    for idx, row in enumerate(rows):
        row["cluster_id"] = int(cluster_ids[idx])
        row["umap_x"] = float(umap_xy[idx][0])
        row["umap_y"] = float(umap_xy[idx][1])

    _write_back_columns(overview["tiles"], rows)
    artifacts = _write_artifacts(rows, output_dir, resolution_sweep=resolution_sweep)

    clusters = to_builtin({
        "n_objects": len(rows),
        "n_clusters": len(set(cluster_ids)) if cluster_ids else 0,
        "method": "cosine_knn_leiden_umap",
        "cluster_mode": cluster_mode,
        "n_neighbors": n_neighbors,
        "effective_n_neighbors": effective_n_neighbors,
        "umap_n_neighbors": umap_n_neighbors,
        "effective_umap_n_neighbors": effective_umap_n_neighbors,
        "umap_min_dist": min_dist,
        "umap_metric": umap_metric,
        "leiden_resolution": selected_resolution,
        "requested_leiden_resolution": resolution,
        "random_state": random_state,
        "table": rows,
        "resolution_sweep": resolution_sweep,
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
    umap_n_neighbors: int,
    resolution: float,
    random_state: int,
    min_dist: float,
    umap_metric: str = "euclidean",
) -> tuple[list[int], list[list[float]], int, int]:
    n_objects = len(vectors)
    if n_objects == 0:
        return [], [], 0, 0
    if n_objects == 1:
        return [0], [[0.0, 0.0]], 0, 0
    if n_objects == 2:
        return [0, 0], [[0.0, 0.0], [1.0, 0.0]], 1, 1

    import numpy as np
    import umap

    x = np.asarray(vectors, dtype=float)
    graph, k = _build_knn_graph(x, n_neighbors)
    cluster_ids = _leiden_labels(graph, resolution, random_state)
    umap_k = _effective_umap_neighbors(umap_n_neighbors, n_objects)

    reducer = umap.UMAP(
        n_neighbors=umap_k,
        min_dist=min_dist,
        metric=umap_metric,
        random_state=random_state,
    )
    umap_xy = reducer.fit_transform(x).astype(float).tolist()
    return cluster_ids, umap_xy, k, umap_k


def _effective_umap_neighbors(n_neighbors: int, n_objects: int) -> int:
    """Clamp UMAP's layout neighborhood independently of the clustering graph."""
    if n_objects <= 2:
        return max(0, n_objects - 1)
    return max(2, min(int(n_neighbors), n_objects - 1))


def _resolve_umap_metric(metric) -> str:
    """Allow cosine (default) or euclidean for the UMAP layout metric."""
    metric = str(metric).lower()
    allowed = ("cosine", "euclidean")
    if metric not in allowed:
        raise ValueError(f"umap_metric must be one of {allowed}; got {metric!r}.")
    return metric


def _build_knn_graph(x, n_neighbors: int):
    import igraph
    from sklearn.neighbors import NearestNeighbors

    n_objects = int(x.shape[0])
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
    return graph, k


def _leiden_labels(graph, resolution: float, random_state: int) -> list[int]:
    import leidenalg

    partition = leidenalg.find_partition(
        graph,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=float(resolution),
        seed=int(random_state),
    )
    labels = [0] * graph.vcount()
    for cluster_id, members in enumerate(partition):
        for member in members:
            labels[int(member)] = int(cluster_id)
    return labels


def _auto_select_resolution(
    vectors: list[list[float]],
    *,
    n_neighbors: int,
    random_state: int,
    initial_grid: list[float],
    refine_steps: int,
    stability_repeats: int,
    max_cluster_fraction: float,
    min_cluster_size: int,
    max_tiny_cluster_fraction: float,
) -> dict:
    import numpy as np
    from sklearn.metrics import adjusted_rand_score

    n_objects = len(vectors)
    if n_objects < 3:
        return {
            "selected_resolution": 1.0,
            "sweep": [],
        }

    x = np.asarray(vectors, dtype=float)
    graph, _ = _build_knn_graph(x, n_neighbors)
    candidates = sorted({float(r) for r in initial_grid if float(r) > 0})
    if not candidates:
        raise ValueError("auto_resolution_grid must contain at least one positive value.")

    labels_by_resolution = _labels_for_resolutions(
        graph, candidates, random_state=random_state
    )
    for _ in range(max(0, int(refine_steps))):
        new_candidates = set(candidates)
        for left, right in zip(candidates, candidates[1:]):
            if adjusted_rand_score(labels_by_resolution[left], labels_by_resolution[right]) < 0.999:
                new_candidates.add((left + right) / 2.0)
        if len(new_candidates) == len(candidates):
            break
        candidates = sorted(new_candidates)
        labels_by_resolution.update(
            _labels_for_resolutions(
                graph,
                [r for r in candidates if r not in labels_by_resolution],
                random_state=random_state,
            )
        )

    rows = []
    unique_labels = []
    for resolution in candidates:
        labels = labels_by_resolution[resolution]
        duplicate_of = None
        for previous_resolution, previous_labels in unique_labels:
            if adjusted_rand_score(labels, previous_labels) >= 0.999:
                duplicate_of = previous_resolution
                break
        if duplicate_of is None:
            unique_labels.append((resolution, labels))

        row = _score_resolution(
            x,
            graph,
            labels,
            resolution=resolution,
            random_state=random_state,
            stability_repeats=stability_repeats,
            max_cluster_fraction=max_cluster_fraction,
            min_cluster_size=min_cluster_size,
            max_tiny_cluster_fraction=max_tiny_cluster_fraction,
            duplicate_of=duplicate_of,
        )
        rows.append(row)

    selected = _choose_resolution(rows)
    for row in rows:
        row["selected"] = bool(row["resolution"] == selected["resolution"])
    return {
        "selected_resolution": float(selected["resolution"]),
        "sweep": rows,
    }


def _labels_for_resolutions(graph, resolutions, *, random_state: int) -> dict[float, list[int]]:
    return {
        float(resolution): _leiden_labels(graph, float(resolution), random_state)
        for resolution in resolutions
    }


def _score_resolution(
    x,
    graph,
    labels: list[int],
    *,
    resolution: float,
    random_state: int,
    stability_repeats: int,
    max_cluster_fraction: float,
    min_cluster_size: int,
    max_tiny_cluster_fraction: float,
    duplicate_of,
) -> dict:
    import numpy as np
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    labels_arr = np.asarray(labels, dtype=int)
    _, counts = np.unique(labels_arr, return_counts=True)
    n_objects = int(labels_arr.size)
    n_clusters = int(counts.size)
    min_size = int(counts.min()) if counts.size else 0
    max_size = int(counts.max()) if counts.size else 0
    max_fraction = float(max_size / n_objects) if n_objects else 0.0
    tiny_clusters = int((counts < int(min_cluster_size)).sum()) if counts.size else 0
    tiny_cluster_fraction = float(tiny_clusters / n_clusters) if n_clusters else 0.0

    silhouette = None
    if 1 < n_clusters < n_objects:
        try:
            silhouette = float(silhouette_score(x, labels_arr, metric="cosine"))
        except ValueError:
            silhouette = None

    stability = None
    if stability_repeats > 0 and n_clusters > 1:
        scores = []
        for offset in range(1, int(stability_repeats) + 1):
            other = _leiden_labels(graph, resolution, random_state + offset)
            scores.append(float(adjusted_rand_score(labels, other)))
        stability = float(np.mean(scores)) if scores else None

    valid = (
        n_clusters >= 2
        and max_fraction <= float(max_cluster_fraction)
        and tiny_cluster_fraction <= float(max_tiny_cluster_fraction)
    )
    return {
        "resolution": float(resolution),
        "n_clusters": n_clusters,
        "min_cluster_size": min_size,
        "max_cluster_size": max_size,
        "max_cluster_fraction": max_fraction,
        "tiny_clusters": tiny_clusters,
        "tiny_cluster_fraction": tiny_cluster_fraction,
        "silhouette": silhouette,
        "stability_ari": stability,
        "valid": bool(valid),
        "duplicate_of": duplicate_of,
        "selected": False,
    }


def _choose_resolution(rows: list[dict]) -> dict:
    pool = [row for row in rows if row["valid"]]
    if not pool:
        pool = [row for row in rows if row["n_clusters"] >= 2] or list(rows)

    def key(row):
        stability = -1.0 if row["stability_ari"] is None else float(row["stability_ari"])
        silhouette = -1.0 if row["silhouette"] is None else float(row["silhouette"])
        return (
            row["valid"],
            stability,
            silhouette,
            -float(row["resolution"]),
        )

    return max(pool, key=key)


def _float_list(value) -> list[float]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return [float(part.strip()) for part in text.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


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


def _write_artifacts(rows: list[dict], output_dir, *, resolution_sweep=None) -> dict:
    if output_dir is None:
        return {}
    features_dir = Path(output_dir) / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    csv_path = features_dir / "clusters.csv"
    json_path = features_dir / "clusters.json"
    svg_path = features_dir / "clusters_umap.svg"
    sweep_csv_path = features_dir / "cluster_resolution_sweep.csv"
    sweep_json_path = features_dir / "cluster_resolution_sweep.json"

    _write_csv(csv_path, rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(to_builtin({"objects": rows}), handle, indent=2, allow_nan=False)
        handle.write("\n")
    _write_svg(svg_path, rows)
    artifacts = {
        "cluster_table_csv": str(csv_path),
        "cluster_table_json": str(json_path),
        "cluster_plot_svg": str(svg_path),
    }
    if resolution_sweep:
        _write_sweep_csv(sweep_csv_path, resolution_sweep)
        with sweep_json_path.open("w", encoding="utf-8") as handle:
            json.dump(to_builtin({"resolution_sweep": resolution_sweep}), handle, indent=2, allow_nan=False)
            handle.write("\n")
        artifacts["resolution_sweep_csv"] = str(sweep_csv_path)
        artifacts["resolution_sweep_json"] = str(sweep_json_path)
    return artifacts


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


def _write_sweep_csv(path: Path, rows: list[dict]) -> None:
    columns = (
        "resolution",
        "n_clusters",
        "min_cluster_size",
        "max_cluster_size",
        "max_cluster_fraction",
        "tiny_clusters",
        "tiny_cluster_fraction",
        "silhouette",
        "stability_ari",
        "valid",
        "duplicate_of",
        "selected",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
