from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine import Engine


WORKFLOW = Path(__file__).resolve().parents[1]
CLUSTER_YAML = WORKFLOW / "pipelines" / "target_discovery_cluster.yaml"


@pytest.mark.cluster
def test_cluster_pipeline_writes_table_plot_and_preserves_positions(tmp_path):
    overview = {"tiles": [_tile()]}
    output_dir = tmp_path / "target_discovery"

    payload = {
        "tiles": overview["tiles"],
        "output_dir": str(output_dir),
        "n_neighbors": 2,
        "leiden_resolution": 1.0,
        "random_state": 7,
        "feature": "area",
        "direction": "high",
        "n_per_tile": 2,
    }

    with Engine() as engine:
        engine.register("target_discovery_cluster", str(CLUSTER_YAML))
        engine.submit("target_discovery_cluster", payload)
        results = _wait_for_result(engine, "target_discovery_cluster")

    result = results[0]["target_discovery"]
    clusters = result["clusters"]
    targets = result["targets"]

    assert clusters["n_objects"] == 6
    assert clusters["n_clusters"] >= 1
    assert len(clusters["table"]) == 6
    assert [row["object_id"] for row in clusters["table"]] == [
        "R0_r000_c000_obj00001",
        "R0_r000_c000_obj00002",
        "R0_r000_c000_obj00003",
        "R0_r000_c000_obj00004",
        "R0_r000_c000_obj00005",
        "R0_r000_c000_obj00006",
    ]
    assert {row["stage_x_um"] for row in clusters["table"]} == {
        1000.0,
        1010.0,
        1020.0,
        1100.0,
        1110.0,
        1120.0,
    }
    assert all("cluster_id" in row for row in clusters["table"])
    assert all("umap_x" in row and "umap_y" in row for row in clusters["table"])

    artifacts = clusters["artifacts"]
    csv_path = Path(artifacts["cluster_table_csv"])
    json_path = Path(artifacts["cluster_table_json"])
    svg_path = Path(artifacts["cluster_plot_svg"])
    assert csv_path.exists()
    assert json_path.exists()
    assert svg_path.exists()
    assert "stage_x_um" in csv_path.read_text(encoding="utf-8")
    assert "<svg" in svg_path.read_text(encoding="utf-8")
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["objects"][0]["object_id"] == "R0_r000_c000_obj00001"
    assert saved["objects"][0]["stage_x_um"] == 1000.0

    assert [target["object_label"] for target in targets] == [6, 5]
    assert [target["stage_x_um"] for target in targets] == [1120.0, 1110.0]


@pytest.mark.cluster
def test_cluster_pipeline_auto_resolution_reports_sweep(tmp_path):
    overview = {"tiles": [_tile()]}
    output_dir = tmp_path / "target_discovery"

    payload = {
        "tiles": overview["tiles"],
        "output_dir": str(output_dir),
        "cluster_mode": "auto",
        "n_neighbors": 2,
        "leiden_resolution": 1.0,
        "auto_resolution_grid": [0.05, 0.2, 0.7, 1.5, 3.0],
        "auto_refine_steps": 1,
        "auto_stability_repeats": 1,
        "auto_max_cluster_fraction": 0.95,
        "auto_min_cluster_size": 2,
        "auto_max_tiny_cluster_fraction": 0.5,
        "random_state": 7,
        "feature": "area",
        "direction": "high",
        "n_per_tile": 2,
    }

    with Engine() as engine:
        engine.register("target_discovery_cluster", str(CLUSTER_YAML))
        engine.submit("target_discovery_cluster", payload)
        results = _wait_for_result(engine, "target_discovery_cluster")

    clusters = results[0]["target_discovery"]["clusters"]
    sweep = clusters["resolution_sweep"]
    selected = [row for row in sweep if row["selected"]]

    assert clusters["cluster_mode"] == "auto"
    assert clusters["leiden_resolution"] in {row["resolution"] for row in sweep}
    assert len(sweep) >= len(payload["auto_resolution_grid"])
    assert len(selected) == 1
    assert selected[0]["resolution"] == clusters["leiden_resolution"]
    assert selected[0]["n_clusters"] >= 2
    assert "silhouette" in selected[0]
    assert "stability_ari" in selected[0]

    artifacts = clusters["artifacts"]
    assert Path(artifacts["resolution_sweep_csv"]).exists()
    assert Path(artifacts["resolution_sweep_json"]).exists()
    saved = json.loads(Path(artifacts["resolution_sweep_json"]).read_text(encoding="utf-8"))
    assert saved["resolution_sweep"][0]["resolution"] == sweep[0]["resolution"]


def _wait_for_result(engine, name, timeout=120):
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        results = engine.results(name)
        if results:
            return results
        status = engine.status(name)
        if status["failed"]:
            failure = status["failures"][0]
            raise AssertionError(f"{failure['step']}: {failure['error']}")
        time.sleep(0.1)
    raise AssertionError(f"Timed out waiting for {name}")


def _tile():
    return {
        "objects": {
            "properties": {
                "label": [1, 2, 3, 4, 5, 6],
                "object_id": [
                    "R0_r000_c000_obj00001",
                    "R0_r000_c000_obj00002",
                    "R0_r000_c000_obj00003",
                    "R0_r000_c000_obj00004",
                    "R0_r000_c000_obj00005",
                    "R0_r000_c000_obj00006",
                ],
                "centroid_row_px": [40.0, 42.0, 44.0, 80.0, 82.0, 84.0],
                "centroid_col_px": [40.0, 42.0, 44.0, 80.0, 82.0, 84.0],
                "bbox_min_row_px": [35, 37, 39, 75, 77, 79],
                "bbox_min_col_px": [35, 37, 39, 75, 77, 79],
                "bbox_max_row_px": [45, 47, 49, 85, 87, 89],
                "bbox_max_col_px": [45, 47, 49, 85, 87, 89],
                "area": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                "intensity_mean": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "eccentricity": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                "stage_x_um": [1000.0, 1010.0, 1020.0, 1100.0, 1110.0, 1120.0],
                "stage_y_um": [2000.0, 2010.0, 2020.0, 2100.0, 2110.0, 2120.0],
            },
            "embeddings": {
                "label": [1, 2, 3, 4, 5, 6],
                "vectors": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.95, 0.05, 0.0, 0.0],
                    [0.9, 0.1, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.95, 0.05, 0.0],
                    [0.0, 0.9, 0.1, 0.0],
                ],
                "model": "test",
                "backend": "mock",
            },
            "n_objects": 6,
        },
        "geometry": {
            "tile_id": ["R0", 0, 0],
            "tile_stage_xy_um": [1000.0, 2000.0],
            "tile_zwide_um": 0.0,
            "source_pixel_size_um": [1.0, 1.0],
            "source_image_size_px": [128, 128],
            "image_to_stage": [[1.0, 0.0], [0.0, 1.0]],
        },
    }
