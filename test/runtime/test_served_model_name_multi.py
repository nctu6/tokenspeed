"""Multi-value ``--served-model-name`` (vLLM parity)."""

from __future__ import annotations

import unittest

from tokenspeed.runtime.utils.server_args import prepare_server_args


class TestServedModelNameMulti(unittest.TestCase):
    def test_single_name_normalizes_to_list(self):
        sa = prepare_server_args(
            ["--model", "m", "--served-model-name", "alias-a"]
        )
        self.assertEqual(sa.served_model_name, ["alias-a"])
        self.assertEqual(sa.served_model_names, ["alias-a"])
        self.assertEqual(sa.canonical_served_model_name, "alias-a")

    def test_multiple_names_preserve_order_and_canonical(self):
        sa = prepare_server_args(
            [
                "--model",
                "/models/google/gemma-3-27b-it",
                "--served-model-name",
                "gemma-3-27b-it",
                "gemma3",
                "g3",
            ]
        )
        self.assertEqual(
            sa.served_model_names, ["gemma-3-27b-it", "gemma3", "g3"]
        )
        self.assertEqual(sa.canonical_served_model_name, "gemma-3-27b-it")

    def test_duplicates_and_empties_dropped(self):
        sa = prepare_server_args(
            [
                "--model",
                "m",
                "--served-model-name",
                "a",
                "b",
                "a",
                "b",
            ]
        )
        self.assertEqual(sa.served_model_names, ["a", "b"])

    def test_default_is_model_path(self):
        sa = prepare_server_args(["--model", "my-model"])
        self.assertEqual(sa.canonical_served_model_name, "my-model")
        self.assertEqual(sa.served_model_names, ["my-model"])


if __name__ == "__main__":
    unittest.main()
