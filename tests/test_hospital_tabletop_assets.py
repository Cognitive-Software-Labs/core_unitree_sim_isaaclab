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

    def test_fixed_table_startup_places_every_configured_tabletop_prop(self):
        config_path = (
            REPO_ROOT
            / "tasks/g1_tasks/pickplace_redblock_hospital_g1_29dof_dex1"
            / "pickplace_redblock_hospital_g1_29dof_dex1_joint_env_cfg.py"
        )
        source = config_path.read_text(encoding="utf-8")
        reset_source = re.search(
            r"def reset_hospital_teleop_scene[\s\S]+?(?=\ndef reset_hospital_target)",
            source,
        )
        self.assertIsNotNone(reset_source)
        reset_source = reset_source.group(0)
        self.assertIn("randomize_pickplace_room_layout(", reset_source)
        self.assertIn("table_prop_names=REDBLOCK_TABLE_PROP_NAMES", reset_source)
        self.assertIn(
            "min_table_objects=len(REDBLOCK_TABLE_PROP_NAMES)", reset_source
        )
        self.assertIn(
            "randomize_table_position=randomize_table_position", reset_source
        )
        self.assertNotIn("randomize_wall_props_layout", reset_source)


if __name__ == "__main__":
    unittest.main()
