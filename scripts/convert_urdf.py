"""
Convert the Gripper-Drone URDF to USD format for Isaac Lab.

This must be run once before training to generate the USD asset.
Isaac Lab requires USD format for high-performance parallel simulation.

Usage:
    python scripts/convert_urdf.py

Output:
    assets/gripper_drone/gripper_drone.usd
"""

from __future__ import annotations

import os
import sys
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Convert URDF to USD")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg


def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    urdf_path = os.path.join(project_dir, "gripper_drone_v5.urdf")
    output_dir = os.path.join(project_dir, "assets", "gripper_drone")

    os.makedirs(output_dir, exist_ok=True)

    print(f"Converting URDF: {urdf_path}")
    print(f"Output directory: {output_dir}")

    cfg = UrdfConverterCfg(
        asset_path=urdf_path,
        usd_dir=output_dir,
        usd_file_name="gripper_drone.usd",
        fix_base=False,
        make_instanceable=True,
        force_usd_conversion=True,
        # Collision approximation
        self_collision=False,
        # Import inertia from URDF (already computed)
        import_inertia_tensor=True,
    )

    converter = UrdfConverter(cfg)
    usd_path = converter.usd_path

    print(f"\nConversion complete!")
    print(f"USD file: {usd_path}")
    print(f"\nUpdate env_cfg.py GRIPPER_DRONE_CFG spawn path if using USD directly.")

    simulation_app.close()


if __name__ == "__main__":
    main()
