from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "assets" / "objects"


class HospitalTabletopAssetTests(unittest.TestCase):
    def test_local_usd_dependencies_are_present(self):
        for asset_name in (
            "hospital_hand_sanitizer.usda",
            "hospital_medicine_bottle.usda",
        ):
            with self.subTest(asset=asset_name):
                asset_path = ASSET_DIR / asset_name
                self.assertTrue(asset_path.is_file())
                source = asset_path.read_text(encoding="utf-8")
                for reference in re.findall(r"@([^@]+)@", source):
                    dependency = asset_path.parent / reference
                    self.assertTrue(
                        dependency.is_file(),
                        f"{asset_name} is missing referenced asset {reference}",
                    )

    def test_prescription_label_is_a_real_png(self):
        label_path = ASSET_DIR / "textures" / "hospital_pharmacy_label.png"
        self.assertTrue(label_path.is_file())
        self.assertEqual(label_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")

    def test_meta_quest_hospital_scene_has_all_requested_props(self):
        config_path = (
            REPO_ROOT
            / "tasks/g1_tasks/pickplace_redblock_hospital_g1_29dof_dex1"
            / "pickplace_redblock_hospital_g1_29dof_dex1_joint_env_cfg.py"
        )
        source = config_path.read_text(encoding="utf-8")
        for prop_name in (
            "hand_sanitizer",
            "medicine_bottle_a",
            "medicine_bottle_b",
        ):
            self.assertRegex(source, rf"\b{prop_name}\s*:")
            self.assertIn(f'"{prop_name}"', source)


if __name__ == "__main__":
    unittest.main()
