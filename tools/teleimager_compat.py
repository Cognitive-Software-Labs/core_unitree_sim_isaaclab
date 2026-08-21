"""Compatibility launcher for the teleimager version pinned by this repository."""

from __future__ import annotations

import os
from collections.abc import Mapping, MutableMapping
from typing import Any


_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def configure_camera_transports(
    camera_config: MutableMapping[str, MutableMapping[str, Any]],
    environ: Mapping[str, str] | None = None,
) -> MutableMapping[str, MutableMapping[str, Any]]:
    """Apply simulator transport overrides without requiring a submodule update.

    The teleimager commit currently pinned by the superproject predates its
    ``TELEIMAGER_DISABLE_WEBRTC`` environment override. Quest mode needs that
    override because xr_teleoperate consumes the simulator feeds over ZMQ and
    hosts the browser-facing Vuer server itself.
    """

    environ = os.environ if environ is None else environ
    disable_webrtc = environ.get("TELEIMAGER_DISABLE_WEBRTC", "").strip().lower()
    if disable_webrtc in _TRUE_VALUES:
        for config in camera_config.values():
            config["enable_webrtc"] = False
    return camera_config


def run_isaacsim_server():
    """Start teleimager after applying superproject transport overrides."""

    import yaml
    from teleimager import image_server

    try:
        with open(image_server.CONFIG_PATH, encoding="utf-8") as config_file:
            camera_config = yaml.safe_load(config_file)
    except Exception as exc:
        image_server.logger_mp.error(
            f"Failed to load configuration file at {image_server.CONFIG_PATH}: {exc}"
        )
        raise

    configure_camera_transports(camera_config)
    server = image_server.ImageServer(
        camera_config,
        realsense_enable=False,
        camera_finder_verbose=False,
        isaacsim_enable=True,
    )
    server.start()
    return server
