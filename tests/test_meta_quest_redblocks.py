from __future__ import annotations

import os
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

from tools.meta_quest import (
    CAMERA_SENSOR_NAMES,
    META_QUEST_REDBLOCK_PROFILES,
    MetaQuestConfigurationError,
    configure_meta_quest,
)
from tools.teleimager_compat import configure_camera_transports


def _args(task: str, **overrides):
    values = {
        "meta_quest": True,
        "task": task,
        "no_render": False,
        "replay_data": False,
        "enable_dex1_dds": False,
        "enable_dex3_dds": False,
        "enable_inspire_dds": False,
        "robot_type": "g129",
        "action_source": "dds",
        "enable_cameras": False,
        "camera_include": "",
        "camera_write_interval": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class MetaQuestRedBlockTests(unittest.TestCase):
    def test_all_registered_redblock_tasks_have_profiles(self):
        repo_root = Path(__file__).resolve().parents[1]
        registered = set()
        for init_file in (repo_root / "tasks").rglob("__init__.py"):
            source = init_file.read_text(encoding="utf-8")
            task_ids = [
                task_id
                for task_id in re.findall(r'id\s*=\s*["\']([^"\']+)["\']', source)
                if "redblock" in task_id.lower()
            ]
            registered.update(task_ids)
            if task_ids:
                config_sources = "\n".join(
                    path.read_text(encoding="utf-8")
                    for path in init_file.parent.glob("*env_cfg.py")
                )
                for sensor_name in (
                    "front_camera",
                    "left_wrist_camera",
                    "right_wrist_camera",
                    "camera_image",
                ):
                    self.assertIn(sensor_name, config_sources, f"{task_ids}: missing {sensor_name}")
        self.assertEqual(registered, set(META_QUEST_REDBLOCK_PROFILES))

    def test_every_verified_task_enables_matching_hand_and_cameras(self):
        for task, expected in META_QUEST_REDBLOCK_PROFILES.items():
            with self.subTest(task=task):
                args = _args(task)
                environ = {}
                profile = configure_meta_quest(args, environ)

                self.assertEqual(profile, expected)
                self.assertEqual(args.robot_type, expected.robot_type)
                self.assertTrue(getattr(args, expected.hand_flag))
                self.assertEqual(
                    sum(
                        bool(getattr(args, flag))
                        for flag in ("enable_dex1_dds", "enable_dex3_dds", "enable_inspire_dds")
                    ),
                    1,
                )
                self.assertTrue(args.enable_cameras)
                self.assertEqual(args.camera_include, CAMERA_SENSOR_NAMES)
                self.assertEqual(args.camera_write_interval, 1)
                self.assertEqual(environ["TELEIMAGER_DISABLE_WEBRTC"], "1")

    def test_no_render_is_rejected_but_headless_is_not(self):
        task = "Isaac-PickPlace-RedBlock-G129-Dex1-Joint"
        with self.assertRaisesRegex(MetaQuestConfigurationError, "Use --headless"):
            configure_meta_quest(_args(task, no_render=True), {})

        args = _args(task, headless=True)
        self.assertIsNotNone(configure_meta_quest(args, {}))

    def test_wrong_hand_is_rejected(self):
        task = "Isaac-PickPlace-RedBlock-G129-Dex3-Joint"
        with self.assertRaisesRegex(MetaQuestConfigurationError, "requires --enable_dex3_dds"):
            configure_meta_quest(_args(task, enable_dex1_dds=True), {})

    def test_replay_is_rejected(self):
        task = "Isaac-PickPlace-RedBlock-G129-Dex1-Joint"
        with self.assertRaisesRegex(MetaQuestConfigurationError, "live DDS mode"):
            configure_meta_quest(_args(task, replay_data=True), {})

    def test_unverified_task_is_rejected(self):
        with self.assertRaisesRegex(MetaQuestConfigurationError, "does not have a verified profile"):
            configure_meta_quest(_args("Isaac-Unknown-Task"), {})

    def test_non_quest_launch_is_untouched(self):
        args = _args("Isaac-Unknown-Task", meta_quest=False)
        environ = dict(os.environ)
        before = vars(args).copy(), environ.copy()
        self.assertIsNone(configure_meta_quest(args, environ))
        self.assertEqual(vars(args), before[0])
        self.assertEqual(environ, before[1])

    def test_pinned_teleimager_config_disables_direct_webrtc(self):
        config = {
            "head_camera": {"enable_zmq": True, "enable_webrtc": True},
            "left_wrist_camera": {"enable_zmq": True, "enable_webrtc": True},
            "right_wrist_camera": {"enable_zmq": True, "enable_webrtc": True},
        }

        configured = configure_camera_transports(
            config, {"TELEIMAGER_DISABLE_WEBRTC": "1"}
        )

        self.assertTrue(all(camera["enable_zmq"] for camera in configured.values()))
        self.assertTrue(all(not camera["enable_webrtc"] for camera in configured.values()))

    def test_hospital_full_reset_enables_table_randomization(self):
        repo_root = Path(__file__).resolve().parents[1]
        config_path = (
            repo_root
            / "tasks/g1_tasks/pickplace_redblock_hospital_g1_29dof_dex1"
            / "pickplace_redblock_hospital_g1_29dof_dex1_joint_env_cfg.py"
        )
        source = config_path.read_text(encoding="utf-8")

        self.assertIn("randomize_table_position: bool | None = None", source)
        self.assertIn("env._teleop_randomize_table_position", source)
        self.assertIn("env_ids: torch.Tensor | None,", source)
        self.assertNotIn('params={"randomize_table_position": None}', source)
        self.assertRegex(
            source,
            r'register\("reset_all_self"[\s\S]+randomize_table_position=True',
        )
        self.assertRegex(
            source,
            r'def reset_hospital_target[\s\S]+reset_target_on_current_table',
        )


if __name__ == "__main__":
    unittest.main()
