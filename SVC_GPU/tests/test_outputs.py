"""Verify that one Athena upload produces one annotation per tissue class."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from SVC_GPU.outputs import prediction_dataframe, save_combined_geojson, save_mask_geojson


class AnnotationExportTests(unittest.TestCase):
    def test_heatmap_and_mask_uploads_have_two_separate_annotations(self):
        df = prediction_dataframe(
            sample_name="slide",
            pt_path=Path("slide.pt"),
            coordinates=np.array([[0, 0], [512, 0]]),
            predictions=np.array([0, 1]),
            decision_scores=np.array([-2.0, 3.0]),
            threshold=0.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            heatmap = json.loads(save_combined_geojson(
                df, directory, "slide", tile_size=512, threshold=0.0,
            ).read_text())
            mask = json.loads(save_mask_geojson(
                df, directory, "slide", tile_size=512, threshold=0.0,
            ).read_text())

        self.assertEqual(
            [record["annotation"]["name"] for record in heatmap],
            ["cortex_heatmap", "medullary_heatmap"],
        )
        self.assertEqual(
            [record["annotation"]["name"] for record in mask],
            ["cortex_mask", "medullary_mask"],
        )
        for records in (heatmap, mask):
            self.assertEqual([len(record["annotation"]["elements"]) for record in records], [1, 1])
            self.assertEqual(
                [record["annotation"]["elements"][0]["center"] for record in records],
                [[256.0, 256.0, 0], [768.0, 256.0, 0]],
            )
        self.assertIn(", 1)", mask[0]["annotation"]["elements"][0]["fillColor"])
        self.assertIn("scaled_confidence", heatmap[0]["annotation"]["elements"][0]["user"])


if __name__ == "__main__":
    unittest.main()
