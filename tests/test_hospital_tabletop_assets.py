from __future__ import annotations

import importlib.util
import math
import random
import re
import sys
import types
import unittest
from pathlib import Path

import torch
from pxr import Usd, UsdGeom, UsdPhysics


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSET_DIR = REPO_ROOT / "assets" / "objects"
ROOM_RANDOMIZER_DIR = REPO_ROOT / "tasks" / "utils" / "room_randomizer"
ROOM_RANDOMIZER_TEST_PACKAGE = "_hospital_tabletop_room_randomizer_test"
GOAL_PATH = (
    REPO_ROOT
    / "tasks/g1_tasks/pickplace_medicine_bottle_hospital_g1_29dof_dex1"
    / "mdp/container_goal.py"
)


def _load_room_randomizer_module(name: str):
    package = sys.modules.get(ROOM_RANDOMIZER_TEST_PACKAGE)
    if package is None:
        package = types.ModuleType(ROOM_RANDOMIZER_TEST_PACKAGE)
        package.__path__ = [str(ROOM_RANDOMIZER_DIR)]
        sys.modules[ROOM_RANDOMIZER_TEST_PACKAGE] = package
    qualified_name = f"{ROOM_RANDOMIZER_TEST_PACKAGE}.{name}"
    spec = importlib.util.spec_from_file_location(
        qualified_name, ROOM_RANDOMIZER_DIR / f"{name}.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load room-randomizer module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


def _load_goal_module():
    module_name = "_hospital_pill_container_goal_test"
    spec = importlib.util.spec_from_file_location(module_name, GOAL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load pill-container goal module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class HospitalTabletopAssetTests(unittest.TestCase):
    CONFIG_PATH = (
        REPO_ROOT
        / "tasks/g1_tasks/pickplace_medicine_bottle_hospital_g1_29dof_dex1"
        / "pickplace_medicine_bottle_hospital_g1_29dof_dex1_joint_env_cfg.py"
    )

    @classmethod
    def setUpClass(cls):
        cls.config_source = cls.CONFIG_PATH.read_text(encoding="utf-8")

    def test_meta_quest_hospital_scene_has_all_requested_props(self):
        expected_props = (
            "pill_bottle_t",
            "pill_bottle_v",
            "medical_bottle_a",
            "medical_bottle_f",
            "marker_blue",
            "marker_yellow",
        )
        for prop_name in expected_props:
            self.assertRegex(self.config_source, rf"\b{prop_name}\s*:")
            self.assertIn(f'"{prop_name}"', self.config_source)
        self.assertIn("object = None", self.config_source)
        self.assertIn("blue_cube = None", self.config_source)
        self.assertIn("yellow_cube = None", self.config_source)

        prop_list = re.search(
            r"MEDICINE_BOTTLE_TABLE_PROP_NAMES\s*=\s*\[([\s\S]+?)\]",
            self.config_source,
        )
        self.assertIsNotNone(prop_list)
        self.assertEqual(tuple(re.findall(r'"([a-z_]+)"', prop_list.group(1))), expected_props)

    def test_screenshot_prop_specs_are_explicit_and_natural_scale(self):
        expected = {
            "pill_bottle_t": ("SM_PillBottle_01t.usd", "0.03", "0.029222", "HOSPITAL"),
            "pill_bottle_v": ("SM_PillBottle_01v.usd", "0.03", "0.027330", "HOSPITAL"),
            "medical_bottle_a": ("SM_BottleA.usd", "0.15", "0.062000", "HOSPITAL"),
            "medical_bottle_f": ("SM_BottleF.usd", "0.15", "0.060100", "HOSPITAL"),
            "marker_blue": ("SM_MarkerBlue.usd", "0.02", "0.024280", "OFFICE"),
            "marker_yellow": ("SM_MarkerYellow.usd", "0.02", "0.024280", "OFFICE"),
        }
        for name, (filename, mass, grasp_width, source) in expected.items():
            with self.subTest(prop=name):
                block = re.search(
                    rf'"{name}": HospitalPropSpec\(([\s\S]+?)\n    \),',
                    self.config_source,
                )
                self.assertIsNotNone(block)
                block = block.group(1)
                self.assertIn(f'filename="{filename}"', block)
                self.assertIn(f"asset_root={source}_PROPS_OMNIVERSE_ROOT", block)
                self.assertIn("scale=1.0", block)
                self.assertIn(f"mass={mass}", block)
                self.assertIn(f"grasp_width={grasp_width}", block)

        self.assertIn("Isaac/Environments/Hospital/Props", self.config_source)
        self.assertIn("Isaac/Environments/Office/Props", self.config_source)
        self.assertIn("scale=(spec.scale, spec.scale, spec.scale)", self.config_source)

    def test_every_prop_uses_a_convex_decomposition_rigid_body(self):
        spawner = re.search(
            r"def _spawn_graspable_hospital_usd[\s\S]+?(?=\n\n@configclass)",
            self.config_source,
        )
        self.assertIsNotNone(spawner)
        source = spawner.group(0)
        self.assertIn("UsdPhysics.CollisionAPI.Apply(collider_prim)", source)
        self.assertIn("UsdPhysics.MeshCollisionAPI.Apply(collider_prim)", source)
        self.assertIn("UsdPhysics.Tokens.convexDecomposition", source)
        self.assertIn("PhysxSchema.PhysxCollisionAPI.Apply(collider_prim)", source)
        self.assertIn("schemas.define_rigid_body_properties", source)
        self.assertIn("schemas.define_mass_properties", source)
        self.assertIn("rigid_roots != [root_prim]", source)
        self.assertNotIn("UsdGeom.Cylinder", source)

    def test_only_the_two_pill_bottles_need_dex1_safe_aperture(self):
        for name in ("pill_bottle_t", "pill_bottle_v"):
            block = re.search(
                rf'"{name}": HospitalPropSpec\(([\s\S]+?)\n    \),',
                self.config_source,
            ).group(1)
            width = float(re.search(r"grasp_width=([0-9.]+)", block).group(1))
            self.assertLessEqual(width, 0.030)

    def test_randomized_yaw_can_preserve_a_base_orientation(self):
        _load_room_randomizer_module("constants")
        placement_utils = _load_room_randomizer_module("placement_utils")
        default_state = torch.zeros((2, 13), dtype=torch.float32)
        default_state[:, 3] = 1.0
        result = placement_utils.build_root_state(
            pos=torch.zeros((2, 3), dtype=torch.float32),
            yaw_rad=torch.tensor([0.0, math.pi / 2], dtype=torch.float32),
            env_origins=torch.zeros((2, 3), dtype=torch.float32),
            env_ids=torch.tensor([0, 1]),
            default_state=default_state,
            base_orientation_wxyz=(0.70710678, 0.70710678, 0.0, 0.0),
        )
        expected = torch.tensor(
            [
                [0.70710678, 0.70710678, 0.0, 0.0],
                [0.5, 0.5, 0.5, 0.5],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(result[:, 3:7], expected, atol=1.0e-6, rtol=0.0)

    def test_reliable_grasp_contact_parameters_are_configured(self):
        for text in (
            'friction_combine_mode="max"',
            'restitution_combine_mode="min"',
            "static_friction=2.5",
            "dynamic_friction=2.0",
            "restitution=0.0",
            "contact_offset=0.001",
            "rest_offset=0.0",
            "linear_damping=1.5",
            "angular_damping=3.0",
            "solver_position_iteration_count=16",
            "solver_velocity_iteration_count=4",
            "func: Callable = _spawn_graspable_hospital_usd",
            "robot.spawn = robot.spawn.replace(func=_spawn_grasp_ready_dex1_usd)",
            "table_prop_meta_overrides=MEDICINE_BOTTLE_TABLE_PROP_META_OVERRIDES",
            "tabletop_spawn_region=HAND_REACHABLE_TABLETOP_REGION",
            "static_cluster_members=STATIC_LOGISTICS_CLUSTER",
        ):
            self.assertIn(text, self.config_source)

    def test_dex1_finger_contacts_get_task_local_material_and_offsets(self):
        spawner = re.search(
            r"def _spawn_grasp_ready_dex1_usd[\s\S]+?(?=\n\n@clone\ndef _spawn_graspable_hospital_usd)",
            self.config_source,
        )
        self.assertIsNotNone(spawner)
        spawner = spawner.group(0)
        self.assertIn('"/left_hand_Link"', spawner)
        self.assertIn('"/right_hand_Link"', spawner)
        self.assertIn(
            "stage.GetPrimAtPath(child_path).SetInstanceable(False)", spawner
        )
        self.assertIn("UsdPhysics.CollisionAPI.Apply(child)", spawner)
        self.assertIn("UsdPhysics.MeshCollisionAPI.Apply(child)", spawner)
        self.assertIn("UsdPhysics.Tokens.convexHull", spawner)
        self.assertIn("PhysxSchema.PhysxCollisionAPI.Apply(child)", spawner)
        self.assertIn('{"left": 6, "right": 6}', spawner)
        self.assertIn("schemas.modify_collision_properties(", spawner)
        self.assertIn("bind_physics_material(", spawner)
        self.assertIn("FINGER_COLLISION_PROPERTIES", spawner)
        self.assertIn("FINGER_PHYSICS_MATERIAL", self.config_source)
        self.assertIn("static_friction=3.0", self.config_source)
        self.assertIn(
            'PILL_GRASP_PAD_RADII = {"left": 0.009, "right": 0.010}',
            self.config_source,
        )
        self.assertIn("pad.CreateRadiusAttr().Set(PILL_GRASP_PAD_RADII[side])", spawner)
        self.assertIn("UsdGeom.Sphere.Define(stage, pad_path)", spawner)
        self.assertIn("finger_collider_paths + grasp_pad_paths", spawner)

    def test_dex1_jaw_drive_is_bounded_for_contact_stability(self):
        for expected in (
            'robot.actuators["hands"] = ImplicitActuatorCfg(',
            "effort_limit_sim=12.0",
            "velocity_limit_sim=0.5",
            "stiffness=600.0",
            "damping=8.0",
            "friction=0.0",
            "self.sim.physx.enable_ccd = False",
        ):
            self.assertIn(expected, self.config_source)

    def test_all_dex1_fingers_start_fully_open(self):
        self.assertIn("DEX1_OPEN_JOINT_POSITION = -0.02", self.config_source)
        for joint_name in (
            "left_hand_Joint1_1",
            "left_hand_Joint2_1",
            "right_hand_Joint1_1",
            "right_hand_Joint2_1",
        ):
            self.assertIn(
                f'"{joint_name}": DEX1_OPEN_JOINT_POSITION',
                self.config_source,
            )

    def test_fixed_table_startup_places_every_configured_tabletop_prop(self):
        reset_source = re.search(
            r"def reset_hospital_teleop_scene[\s\S]+?(?=\ndef reset_hospital_tabletop_props)",
            self.config_source,
        )
        self.assertIsNotNone(reset_source)
        reset_source = reset_source.group(0)
        self.assertIn("randomize_pickplace_room_layout(", reset_source)
        self.assertIn("table_prop_names=MEDICINE_BOTTLE_TABLE_PROP_NAMES", reset_source)
        self.assertIn(
            "min_table_objects=len(MEDICINE_BOTTLE_TABLE_PROP_NAMES)", reset_source
        )
        self.assertIn(
            "randomize_table_position=randomize_table_position", reset_source
        )
        self.assertIn(
            "table_prop_meta_overrides=MEDICINE_BOTTLE_TABLE_PROP_META_OVERRIDES",
            reset_source,
        )
        self.assertIn(
            "tabletop_object_margin=COMPACT_TABLETOP_OBJECT_MARGIN", reset_source
        )
        self.assertNotIn("randomize_wall_props_layout", reset_source)

    def test_scene_replaces_the_legacy_target_with_screenshot_props(self):
        scene_block = re.search(
            r"class HospitalMedicineBottleSceneCfg[\s\S]+?(?=\n\n##\n# MDP)",
            self.config_source,
        )
        self.assertIsNotNone(scene_block)
        scene_block = scene_block.group(0)
        self.assertIn("object = None", scene_block)
        for entity in (
            "pill_bottle_t",
            "pill_bottle_v",
            "medical_bottle_a",
            "medical_bottle_f",
            "marker_blue",
            "marker_yellow",
        ):
            self.assertRegex(
                scene_block,
                rf"{entity}:\s*RigidObjectCfg\s*=\s*_hospital_prop_cfg\(",
            )
        self.assertNotIn("tabletop_cube_cfg", self.config_source)
        self.assertNotIn("_redblock_spawn_cfg", self.config_source)

    def test_compact_spawn_region_is_inside_the_handward_table_area(self):
        region = re.search(
            r"HAND_REACHABLE_TABLETOP_REGION\s*=\s*TabletopSpawnRegion\(([\s\S]+?)\n\)",
            self.config_source,
        )
        self.assertIsNotNone(region)
        values = {
            key: float(value)
            for key, value in re.findall(
                r"(x_min|x_max|y_min|y_max)=(-?[0-9.]+)", region.group(1)
            )
        }
        self.assertEqual(
            values,
            {"x_min": -0.32, "x_max": 0.18, "y_min": -0.16, "y_max": 0.12},
        )
        self.assertIn("COMPACT_TABLETOP_OBJECT_MARGIN = 0.015", self.config_source)
        self.assertLess(
            (values["x_max"] - values["x_min"])
            * (values["y_max"] - values["y_min"]),
            0.18,
        )

    def test_compact_region_places_all_six_across_many_seeds(self):
        constants = _load_room_randomizer_module("constants")
        placement = _load_room_randomizer_module("placement_utils")
        props = (
            constants.BBox(0.014611, 0.014611),
            constants.BBox(0.013665, 0.013665),
            constants.BBox(0.031000, 0.031000),
            constants.BBox(0.030050, 0.030050),
            constants.BBox(0.058500, 0.012140),
            constants.BBox(0.058500, 0.012140),
        )
        bounds = (-0.32, 0.18, -0.16, 0.12)
        for seed in range(1000):
            rng = random.Random(seed)
            placed = []
            for bbox in props:
                for _ in range(300):
                    box = placement.make_obb(
                        rng.uniform(bounds[0], bounds[1]),
                        rng.uniform(bounds[2], bounds[3]),
                        bbox,
                        rng.uniform(0.0, 2.0 * math.pi),
                    )
                    corners = placement.obb_corners(*box)
                    if any(
                        x < bounds[0]
                        or x > bounds[1]
                        or y < bounds[2]
                        or y > bounds[3]
                        for x, y in corners
                    ):
                        continue
                    if placement.obb_overlap_any(box, placed, margin=0.015):
                        continue
                    placed.append(box)
                    break
                else:
                    self.fail(f"seed {seed} could not place all six props")

            # Robot table-local origin is (0.10, 0.50), facing -Y. Check the
            # complete OBB corners rather than only object centers.
            half_horizontal_fov = math.radians(96.0255750084 * 0.5)
            for box in placed:
                for x, y in placement.obb_corners(*box):
                    horizontal_angle = math.atan2(abs(x - 0.10), 0.50 - y)
                    self.assertLess(horizontal_angle, half_horizontal_fov)

    def test_object_only_reset_respawns_all_six_on_the_current_table(self):
        reset_source = re.search(
            r"def reset_hospital_tabletop_props[\s\S]+?(?=\ndef reset_hospital_room_fixed_table)",
            self.config_source,
        )
        self.assertIsNotNone(reset_source)
        source = reset_source.group(0)
        self.assertIn("for asset_name in MEDICINE_BOTTLE_TABLE_PROP_NAMES", source)
        self.assertIn("asset_name=asset_name", source)
        self.assertIn("tabletop_object_margin=COMPACT_TABLETOP_OBJECT_MARGIN", source)
        self.assertNotIn("reset_scene_to_default", source)
        self.assertIn(
            'self.event_manager.register("reset_object_self", SimpleEvent(',
            self.config_source,
        )
        self.assertIn("func=reset_hospital_tabletop_props", self.config_source)

    def test_goal_tracks_both_parented_crates_in_their_live_frames(self):
        goal = _load_goal_module()
        num_envs = 3
        identity = torch.tensor((1.0, 0.0, 0.0, 0.0)).repeat(num_envs, 1)
        left_quat = identity.clone()
        left_quat[1] = torch.tensor(
            (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))
        )
        left_pos = torch.tensor(
            ((1.0, 2.0, 0.1), (4.0, -3.0, 0.2), (-2.0, 1.0, 0.0))
        )
        right_pos = torch.tensor(
            ((-3.0, 0.5, 0.0), (2.0, 5.0, 0.1), (6.0, -4.0, 0.2))
        )

        def crate_point(parent_pos, parent_quat, crate_index, local_point):
            local_pos = torch.tensor(goal.CRATE_LOCAL_POSITIONS[crate_index]).repeat(
                num_envs, 1
            )
            local_quat = torch.tensor(
                goal.CRATE_LOCAL_ORIENTATIONS[crate_index]
            ).repeat(num_envs, 1)
            crate_pos = parent_pos + goal.quaternion_apply(parent_quat, local_pos)
            crate_quat = goal.quaternion_multiply(parent_quat, local_quat)
            point = torch.tensor(local_point).repeat(num_envs, 1)
            return crate_pos + goal.quaternion_apply(crate_quat, point)

        left_inside = crate_point(left_pos, left_quat, 0, (0.0, 0.0, 0.05))
        right_inside = crate_point(right_pos, identity, 1, (0.0, 0.0, 0.05))
        bottle_t_center = left_inside.clone()
        bottle_v_center = torch.stack(
            (right_inside[0], torch.tensor((50.0, 50.0, 50.0)), left_inside[2])
        )

        def rigid_data(center, center_offset, root_quat=identity):
            offset = torch.tensor(center_offset).repeat(num_envs, 1)
            root_pos = center - goal.quaternion_apply(root_quat, offset)
            return types.SimpleNamespace(root_pos_w=root_pos, root_quat_w=root_quat)

        scene = {
            "pill_bottle_t": types.SimpleNamespace(
                data=rigid_data(
                    bottle_t_center, goal.PILL_BOTTLE_LOCAL_CENTERS[0]
                )
            ),
            "pill_bottle_v": types.SimpleNamespace(
                data=rigid_data(
                    bottle_v_center, goal.PILL_BOTTLE_LOCAL_CENTERS[1]
                )
            ),
            "ridgeback_left": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pos_w=left_pos, root_quat_w=left_quat
                )
            ),
            "ridgeback_right": types.SimpleNamespace(
                data=types.SimpleNamespace(
                    root_pos_w=right_pos, root_quat_w=identity
                )
            ),
        }
        env = types.SimpleNamespace(scene=scene, num_envs=num_envs)
        torch.testing.assert_close(
            goal.pill_bottles_contained(env),
            torch.tensor(((True, True), (True, False), (True, True))),
        )
        torch.testing.assert_close(
            goal.both_pill_bottles_contained(env),
            torch.tensor((True, False, True)),
        )
        self.assertFalse(env._teleop_randomize_table_position)
        self.assertTrue(env._hospital_success_reset_pending)

    def test_reward_and_termination_use_the_two_pill_goal(self):
        self.assertIn(
            "success = DoneTerm(func=mdp.both_pill_bottles_contained)",
            self.config_source,
        )
        self.assertIn("func=mdp.compute_pill_bottle_reward", self.config_source)
        self.assertNotIn("post_min_x", self.config_source)
        rewards_source = GOAL_PATH.with_name("rewards.py").read_text(encoding="utf-8")
        self.assertIn("sum(dim=-1) * 0.5", rewards_source)

    def test_two_static_ridgebacks_each_carry_one_omniverse_crate(self):
        self.assertEqual(self.config_source.count("_static_ridgeback_cfg("), 3)
        self.assertEqual(self.config_source.count("_ridgeback_crate_cfg("), 3)
        for name in ("ridgeback_left", "ridgeback_right"):
            self.assertIn(f'asset_name="{name}"', self.config_source)
        for path in (
            "/RidgebackLeft/base_link/Crate",
            "/RidgebackRight/base_link/Crate",
        ):
            self.assertIn(path, self.config_source)
        self.assertIn("RidgebackUr/ridgeback_ur5.usd", self.config_source)
        self.assertIn("SM_CratePlastic_D_02.usd", self.config_source)
        self.assertIn("kinematic_enabled=True", self.config_source)
        self.assertIn("disable_gravity=True", self.config_source)
        self.assertNotIn("ArticulationCfg(\n        prim_path=\"/World/envs/env_.*/Ridgeback", self.config_source)

    def test_ridgeback_start_poses_are_mirrored_across_g1(self):
        robot_match = re.search(
            r"ROBOT_POS\s*=\s*\((-?[0-9.]+),\s*(-?[0-9.]+),\s*(-?[0-9.]+)\)",
            self.config_source,
        )
        self.assertIsNotNone(robot_match)
        robot_x = float(robot_match.group(1))

        positions = {}
        for side in ("LEFT", "RIGHT"):
            match = re.search(
                rf"RIDGEBACK_{side}_POS\s*=\s*\("
                r"(-?[0-9.]+),\s*(-?[0-9.]+),\s*(-?[0-9.]+)\)",
                self.config_source,
            )
            self.assertIsNotNone(match)
            positions[side] = tuple(float(value) for value in match.groups())

        left = positions["LEFT"]
        right = positions["RIGHT"]
        self.assertEqual(left, (-5.20520, -6.66770, 0.0328))
        self.assertAlmostEqual((left[0] + right[0]) / 2.0, robot_x)
        self.assertEqual(left[1:], right[1:])
        # Project each screenshot-rotated footprint onto world X: they retain
        # positive physical clearance despite their table-facing yaw.
        half_extent_x = (
            abs(math.cos(math.radians(154.092))) * 0.50
            + abs(math.sin(math.radians(154.092))) * 0.42
        )
        self.assertGreater(abs(right[0] - left[0]), 2.0 * half_extent_x)
        self.assertIn("robot_local_xy=(-0.33230, 0.69480)", self.config_source)
        self.assertIn("robot_local_xy=(-0.33230, -0.69480)", self.config_source)
        self.assertIn("RIDGEBACK_LEFT_ROT = (0.2241687075, 0.0, 0.0, -0.9745503530)", self.config_source)
        self.assertIn("RIDGEBACK_RIGHT_ROT = (0.9745503530, 0.0, 0.0, -0.2241687075)", self.config_source)
        self.assertIn("yaw_offset=RIDGEBACK_LEFT_YAW_OFFSET", self.config_source)
        self.assertIn("yaw_offset=RIDGEBACK_RIGHT_YAW_OFFSET", self.config_source)

    def test_fixed_startup_cluster_includes_latest_left_shift(self):
        for expected in (
            "TABLE_POS = (-6.0, -7.5, -0.2)",
            "ROBOT_POS = (-5.9, -7.0, 0.76)",
            "RIDGEBACK_LEFT_POS = (-5.20520, -6.66770, 0.0328)",
            "RIDGEBACK_RIGHT_POS = (-6.59480, -6.66770, 0.0328)",
            'pill_bottle_t: RigidObjectCfg = _hospital_prop_cfg(',
            'pill_bottle_v: RigidObjectCfg = _hospital_prop_cfg(',
            'medical_bottle_a: RigidObjectCfg = _hospital_prop_cfg(',
            'medical_bottle_f: RigidObjectCfg = _hospital_prop_cfg(',
            'marker_blue: RigidObjectCfg = _hospital_prop_cfg(',
            'marker_yellow: RigidObjectCfg = _hospital_prop_cfg(',
            "pos_offset=(-5.8, -8.2, 1.8)",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, self.config_source)

    def test_front_camera_uses_saved_viewport_optics(self):
        camera = re.search(
            r"front_camera\s*=\s*CameraBaseCfg\.get_camera_config\(([\s\S]+?)\n\s*\)",
            self.config_source,
        )
        self.assertIsNotNone(camera)
        camera = camera.group(1)
        self.assertIn("height=HOSPITAL_FRONT_CAMERA_OPTICS.height", camera)
        self.assertIn("width=HOSPITAL_FRONT_CAMERA_OPTICS.width", camera)
        self.assertIn("focal_length=HOSPITAL_FRONT_CAMERA_OPTICS.focal_length", camera)
        self.assertIn("focus_distance=HOSPITAL_FRONT_CAMERA_OPTICS.focus_distance", camera)
        self.assertIn(
            "horizontal_aperture=HOSPITAL_FRONT_CAMERA_OPTICS.horizontal_aperture",
            camera,
        )
        self.assertIn("Euler=(90,-90,0)", self.config_source)

    def test_front_camera_optics_are_reasserted_on_scene_resets(self):
        optics_update = re.search(
            r"def _apply_hospital_front_camera_optics[\s\S]+?(?=\ndef reset_hospital_teleop_scene)",
            self.config_source,
        )
        self.assertIsNotNone(optics_update)
        optics_update = optics_update.group(0)
        for expected in (
            "GetFocalLengthAttr().Set(optics.focal_length)",
            "GetFocusDistanceAttr().Set(optics.focus_distance)",
            "GetHorizontalApertureAttr().Set(optics.horizontal_aperture)",
            "GetVerticalApertureAttr().Set(vertical_aperture)",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, optics_update)

        self.assertEqual(
            self.config_source.count("\n    _apply_hospital_front_camera_optics(env)"),
            2,
        )

    def test_hospital_quest_releases_yaw_and_pitch_but_locks_roll(self):
        scene = re.search(
            r"class HospitalMedicineBottleSceneCfg[\s\S]+?(?=\n\n##\n# MDP)",
            self.config_source,
        )
        self.assertIsNotNone(scene)
        scene = scene.group(0)
        for expected in (
            'robot.actuators.pop("waist", None)',
            'robot.actuators["waist_yaw_pitch_teleop"]',
            'joint_names_expr=["waist_yaw_joint", "waist_pitch_joint"]',
            'robot.actuators["waist_roll_lock"]',
            'joint_names_expr=["waist_roll_joint"]',
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, scene)

    def test_ridgeback_containers_are_mirrored_across_g1(self):
        self.assertEqual(self.config_source.count("_ridgeback_crate_cfg("), 3)
        self.assertIn("pos=(local_x, local_y, local_z)", self.config_source)
        self.assertIn("local_x=0.22877", self.config_source)
        self.assertIn("local_y=0.00612", self.config_source)
        self.assertIn("local_y=-0.00612", self.config_source)
        self.assertIn("local_z=0.28576", self.config_source)
        self.assertIn("local_rot=CRATE_LEFT_ROT", self.config_source)
        self.assertIn("local_rot=CRATE_RIGHT_ROT", self.config_source)
        self.assertIn("preserving a world-X mirror", self.config_source)


if __name__ == "__main__":
    unittest.main()
