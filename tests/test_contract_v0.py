"""End-to-end handoff checks using a small synthetic room and RIR."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from jsonschema import Draft202012Validator, ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from echospace.contract_v0 import (  # noqa: E402
    ContractError,
    load_npz,
    read_manifest,
    validate_prediction,
    validate_sample,
)


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_contract_v0.json"
SCHEMA = ROOT / "contracts" / "v0.1.0" / "manifest.schema.json"
SCENE_SCHEMA = ROOT / "contracts" / "v0.1.0" / "room-scene.schema.json"
SCENE_FIXTURE = ROOT / "tests" / "fixtures" / "synthetic_room_scene_v1.json"


class ContractV0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
        cls.validator = Draft202012Validator(json.loads(SCHEMA.read_text(encoding="utf-8")))
        cls.validator.check_schema(cls.validator.schema)
        cls.scene_validator = Draft202012Validator(json.loads(SCENE_SCHEMA.read_text(encoding="utf-8")))
        cls.scene_validator.check_schema(cls.scene_validator.schema)

    def test_metric_room_scene_preserves_provenance(self) -> None:
        scene = json.loads(SCENE_FIXTURE.read_text(encoding="utf-8"))
        self.scene_validator.validate(scene)
        self.assertEqual(scene["source_sample_id"], self.fixture["record"]["sample_id"])
        self.assertEqual(scene["acoustic_measurements"][0]["rir_id"], self.fixture["record"]["rir_ids"][0])
        scene["inferred_geometry"][0]["provenance"] = "measured"
        with self.assertRaises(ValidationError):
            self.scene_validator.validate(scene)

    def test_fixture_runs_through_manifest_sample_and_prediction(self) -> None:
        record = copy.deepcopy(self.fixture["record"])
        self.validator.validate(record)
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            manifest_path = directory / "manifest.jsonl"
            sample_path = directory / "sample.npz"
            prediction_path = directory / "prediction.npz"
            manifest_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
            np.savez_compressed(sample_path, **self.fixture["sample"])
            np.savez_compressed(prediction_path, **self.fixture["prediction"])
            loaded_record, = read_manifest(manifest_path)
            validate_sample(loaded_record, load_npz(directory / loaded_record["sample_path"]))
            validate_prediction(loaded_record, load_npz(prediction_path))

    def test_rejects_wrong_grid_shape(self) -> None:
        sample = copy.deepcopy(self.fixture["sample"])
        sample["observed_cells"] = [[1, 0]]
        with self.assertRaisesRegex(ContractError, "observed_cells"):
            validate_sample(self.fixture["record"], sample)

    def test_rejects_observed_state_outside_visibility(self) -> None:
        sample = copy.deepcopy(self.fixture["sample"])
        sample["observed_free"][3][2] = 1
        with self.assertRaisesRegex(ContractError, "require visibility"):
            validate_sample(self.fixture["record"], sample)

    def test_rejects_unmatched_rir_count(self) -> None:
        sample = copy.deepcopy(self.fixture["sample"])
        sample["rir_waveforms"].append([0] * 8)
        with self.assertRaisesRegex(ContractError, "RIR waveforms"):
            validate_sample(self.fixture["record"], sample)

    def test_rejects_missing_source_receiver_positions(self) -> None:
        sample = copy.deepcopy(self.fixture["sample"])
        sample.pop("rir_positions_scene_m")
        with self.assertRaisesRegex(ContractError, "rir_positions_scene_m"):
            validate_sample(self.fixture["record"], sample)

    def test_rejects_rir_pose_in_hidden_space(self) -> None:
        sample = copy.deepcopy(self.fixture["sample"])
        sample["rir_positions_scene_m"][0][0] = [0.75, 0, 1.25]
        with self.assertRaisesRegex(ContractError, "observed free space"):
            validate_sample(self.fixture["record"], sample)

    def test_allows_missing_acoustics_with_mask(self) -> None:
        sample = copy.deepcopy(self.fixture["sample"])
        sample["rir_waveforms"] = [[0] * 8]
        sample["rir_positions_scene_m"] = [[[0, 0, 0], [0, 0, 0]]]
        sample["rir_valid"] = [0]
        validate_sample(self.fixture["record"], sample)

    def test_rejects_bad_transform(self) -> None:
        record = copy.deepcopy(self.fixture["record"])
        record["frame"]["source_to_scene"][3] = [0, 0, 1, 1]
        with self.assertRaisesRegex(ContractError, "affine"):
            validate_sample(record, self.fixture["sample"])

    def test_rejects_invalid_probability(self) -> None:
        prediction = copy.deepcopy(self.fixture["prediction"])
        prediction["occupancy_probability"][0][0] = 1.2
        with self.assertRaisesRegex(ContractError, "probabilities"):
            validate_prediction(self.fixture["record"], prediction)

    def test_rejects_room_duplicate_sample_ids(self) -> None:
        record = self.fixture["record"]
        with tempfile.TemporaryDirectory() as folder:
            manifest_path = Path(folder) / "manifest.jsonl"
            manifest_path.write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ContractError, "duplicate sample_id"):
                list(read_manifest(manifest_path))

    def test_rejects_pickle_objects(self) -> None:
        with tempfile.TemporaryDirectory() as folder:
            sample_path = Path(folder) / "unsafe.npz"
            np.savez_compressed(sample_path, objects=np.array([{"x": 1}], dtype=object))
            with self.assertRaises(ValueError):
                load_npz(sample_path)


if __name__ == "__main__":
    unittest.main()
