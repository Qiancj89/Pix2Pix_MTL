import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common import assert_patient_disjoint, proposed_patient_id
from importlib import import_module

split_module = import_module("01_make_patient_splits")
mapping_module = import_module("00_make_patient_mapping")


class PatientSplitTests(unittest.TestCase):
    def test_blank_string_mask_accepts_empty_numeric_series(self):
        result = split_module.blank_string_mask(pd.Series(dtype="float64"))
        self.assertTrue(result.empty)

    def test_mapping_ignores_blank_name_summary_rows(self):
        frame = pd.DataFrame({
            "English_Name": ["1_case", None, "  ", None],
            "Internal_Echo": [0, None, 21, 2],
            "Malignant": [0, None, None, None],
        })
        cleaned, skipped, skipped_with_values = mapping_module.filter_named_rows(frame)
        self.assertEqual(cleaned["English_Name"].tolist(), ["1_case"])
        self.assertEqual(skipped, 3)
        self.assertEqual(skipped_with_values, 2)

    def test_proposal_removes_augmentation_and_lesion_suffix(self):
        self.assertEqual(proposed_patient_id("your image"), "image ID")

    def test_patient_allocation_is_deterministic_and_disjoint(self):
        patients = pd.DataFrame({"patient_id": [f"p{i}" for i in range(20)],
                                 "Malignant": [i % 2 for i in range(20)]})
        first = split_module.allocate_validation_patients(patients, 0.2, 42)
        second = split_module.allocate_validation_patients(patients, 0.2, 42)
        self.assertEqual(first, second); self.assertEqual(len(first), 4)
        rows = []
        for patient_id in patients.patient_id:
            for lesion in range(2):
                rows.append({"patient_id": patient_id, "split": "val" if patient_id in first else "train"})
        assert_patient_disjoint(pd.DataFrame(rows))

    def test_audit_detects_leakage(self):
        with self.assertRaises(RuntimeError):
            assert_patient_disjoint(pd.DataFrame({"patient_id": ["p1", "p1"], "split": ["train", "val"]}))


if __name__ == "__main__":
    unittest.main()
