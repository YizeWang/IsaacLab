# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn multiple objects in multiple environments.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p scripts/demos/multi_asset.py --num_envs 2048

    # Run with Newton MJWarp physics and the Newton visualizer
    ./isaaclab.sh -p scripts/demos/multi_asset.py --num_envs 2048 physics=newton_mjwarp --visualizer newton

"""

from __future__ import annotations

import argparse

from isaaclab_tasks.utils.demo_launcher import DemoAppLauncher, create_demo_physics_cfg

# add argparse arguments
parser = argparse.ArgumentParser(description="Demo on spawning different objects in multiple environments.")
parser.add_argument("--num_envs", type=int, default=512, help="Number of environments to spawn.")
# demos should open Kit visualizer by default
parser.set_defaults(visualizer=["kit"])
# parse the arguments
args_cli = DemoAppLauncher.parse_args(parser)
visualizers = getattr(args_cli, "visualizer", None) or []
if isinstance(visualizers, str):
    visualizers = [token.strip() for token in visualizers.split(",") if token.strip()]
uses_newton_visualizer = "newton" in {str(visualizer).lower() for visualizer in visualizers}
if args_cli.physics == "newton_mjwarp" and uses_newton_visualizer and not hasattr(args_cli, "max_visible_envs"):
    args_cli.max_visible_envs = min(args_cli.num_envs, 1)

# launch omniverse app
simulation_app = DemoAppLauncher(args_cli)

"""Rest everything follows."""

import random
import time

from pxr import Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
    RigidObjectCollection,
    RigidObjectCollectionCfg,
)
from isaaclab.physics import PhysicsCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.sim.utils.stage import get_current_stage
from isaaclab.utils import Timer
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

##
# Pre-defined Configuration
##

from isaaclab_assets.robots.anymal import ANYDRIVE_3_LSTM_ACTUATOR_CFG  # isort: skip


##
# Randomization events.
##


def randomize_shape_color(prim_path_expr: str):
    """Randomize the color of the geometry."""
    # get stage handle
    stage = get_current_stage()
    # resolve prim paths for spawning and cloning
    prim_paths = sim_utils.find_matching_prim_paths(prim_path_expr)
    # manually clone prims if the source prim path is a regex expression
    with Sdf.ChangeBlock():
        for prim_path in prim_paths:
            # spawn single instance
            prim_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), prim_path)

            # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
            # Note: Just need to acquire the right attribute about the property you want to set
            # Here is an example on setting color randomly
            color_spec = prim_spec.GetAttributeAtPath(prim_path + "/geometry/material/Shader.inputs:diffuseColor")
            if color_spec is not None:
                color_spec.default = Gf.Vec3f(random.random(), random.random(), random.random())


##
# Scene Configuration
##


@configclass
class MultiObjectSceneCfg(InteractiveSceneCfg):
    """Configuration for a multi-object scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # rigid object
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.ConeCfg(
                    radius=0.3,
                    height=0.6,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.3, 0.3, 0.3),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.SphereCfg(
                    radius=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2.0)),
    )

    # object collection
    object_collection: RigidObjectCollectionCfg = RigidObjectCollectionCfg(
        rigid_objects={
            "object_A": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object_A",
                spawn=sim_utils.SphereCfg(
                    radius=0.1,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.5, 2.0)),
            ),
            "object_B": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object_B",
                spawn=sim_utils.CuboidCfg(
                    size=(0.1, 0.1, 0.1),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.5, 2.0)),
            ),
            "object_C": RigidObjectCfg(
                prim_path="/World/envs/env_.*/Object_C",
                spawn=sim_utils.ConeCfg(
                    radius=0.1,
                    height=0.3,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=4, solver_velocity_iteration_count=0
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 2.0)),
            ),
        }
    )

    # articulation
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[
                f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
                f"{ISAACLAB_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-D/anymal_d.usd",
            ],
            random_choice=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=4, solver_velocity_iteration_count=0
            ),
            activate_contact_sensors=True,
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.6),
            joint_pos={
                ".*HAA": 0.0,  # all HAA
                ".*F_HFE": 0.4,  # both front HFE
                ".*H_HFE": -0.4,  # both hind HFE
                ".*F_KFE": -0.8,  # both front KFE
                ".*H_KFE": 0.8,  # both hind KFE
            },
        ),
        actuators={"legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG},
    )


def make_mjwarp_scene_homogeneous(scene_cfg: MultiObjectSceneCfg) -> MultiObjectSceneCfg:
    """Use homogeneous clone sources for Newton MJWarp."""
    if not simulation_app.is_newton_mjwarp:
        return scene_cfg

    scene_cfg.object.spawn.assets_cfg = scene_cfg.object.spawn.assets_cfg[1:2]
    scene_cfg.object.spawn.random_choice = False
    object_c_cfg = scene_cfg.object_collection.rigid_objects["object_C"]
    object_c_spawn = object_c_cfg.spawn
    object_c_cfg.spawn = sim_utils.CuboidCfg(
        size=(0.1, 0.1, 0.3),
        visual_material=object_c_spawn.visual_material,
        rigid_props=object_c_spawn.rigid_props,
        mass_props=object_c_spawn.mass_props,
        collision_props=object_c_spawn.collision_props,
    )
    if isinstance(scene_cfg.robot.spawn.usd_path, list):
        scene_cfg.robot.spawn.usd_path = scene_cfg.robot.spawn.usd_path[0]
    scene_cfg.robot.spawn.random_choice = False
    scene_cfg.robot.actuators = {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            effort_limit_sim=80.0,
            velocity_limit_sim=7.5,
            stiffness={".*": 40.0},
            damping={".*": 5.0},
        )
    }
    return scene_cfg


def create_multi_asset_physics_cfg() -> PhysicsCfg | None:
    """Create Newton MJWarp settings for the multi-asset demo."""
    if not simulation_app.is_newton_mjwarp:
        return None

    physics_cfg = create_demo_physics_cfg("newton_mjwarp")
    physics_cfg.use_cuda_graph = False
    return physics_cfg


##
# Simulation Loop
##


def reset_scene_state(
    scene: InteractiveScene,
    rigid_object: RigidObject,
    rigid_object_collection: RigidObjectCollection,
    robot: Articulation,
    *,
    reset_object_collection: bool,
) -> None:
    """Reset the scene entities to their configured default state."""
    # object
    root_pose = rigid_object.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    rigid_object.write_root_pose_to_sim_index(root_pose=root_pose)
    root_vel = rigid_object.data.default_root_vel.torch.clone()
    rigid_object.write_root_velocity_to_sim_index(root_velocity=root_vel)
    # object collection
    if reset_object_collection:
        default_pose_w = rigid_object_collection.data.default_body_pose.torch.clone()
        default_pose_w[..., :3] += scene.env_origins.unsqueeze(1)
        rigid_object_collection.write_body_pose_to_sim_index(body_poses=default_pose_w)
        default_vel_w = rigid_object_collection.data.default_body_vel.torch.clone()
        rigid_object_collection.write_body_com_velocity_to_sim_index(body_velocities=default_vel_w)
    # robot
    # -- root state
    root_pose = robot.data.default_root_pose.torch.clone()
    root_pose[:, :3] += scene.env_origins
    robot.write_root_pose_to_sim_index(root_pose=root_pose)
    root_vel = robot.data.default_root_vel.torch.clone()
    robot.write_root_velocity_to_sim_index(root_velocity=root_vel)
    # -- joint state
    joint_pos, joint_vel = (
        robot.data.default_joint_pos.torch.clone(),
        robot.data.default_joint_vel.torch.clone(),
    )
    robot.write_joint_position_to_sim_index(position=joint_pos)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel)
    # clear internal buffers
    scene.reset()
    print("[INFO]: Resetting scene state...")


def run_simulator(sim: SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability.
    rigid_object: RigidObject = scene["object"]
    rigid_object_collection: RigidObjectCollection = scene["object_collection"]
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    render_dt = max(sim.get_rendering_dt(), 1.0 / 60.0)
    is_rendering = sim.is_rendering
    count = 0
    sim_step_count = 0

    # Simulation loop
    while simulation_app.is_running():
        # Reset
        if count % 250 == 0:
            # reset counter
            count = 0
            # reset the scene entities
            reset_scene_state(scene, rigid_object, rigid_object_collection, robot, reset_object_collection=True)

        # Apply action to robot
        robot.set_joint_position_target_index(target=robot.data.default_joint_pos.torch)
        # Write data to sim
        scene.write_data_to_sim()
        # Perform step
        sim.step(render=False)
        sim_step_count += 1
        if sim_step_count % sim.cfg.render_interval == 0 and is_rendering:
            sim.render()
            if simulation_app.is_newton_mjwarp:
                time.sleep(render_dt)
        # Increment counter
        count += 1
        # Update buffers
        scene.update(sim_dt)


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    if simulation_app.is_newton_mjwarp:
        sim_cfg.render_interval = 4
        from isaaclab_visualizers.newton import NewtonVisualizerCfg

        newton_visualizer_cfg = NewtonVisualizerCfg()
        newton_visualizer_cfg.max_visible_envs = getattr(args_cli, "max_visible_envs", min(args_cli.num_envs, 1))
        newton_visualizer_cfg.randomly_sample_visible_envs = False
        sim_cfg.visualizer_cfgs = [newton_visualizer_cfg]
    sim = simulation_app.create_context(sim_cfg, SimulationContext, newton_cfg=create_multi_asset_physics_cfg())
    # Set main camera
    sim.set_camera_view([2.5, 0.0, 4.0], [0.0, 0.0, 2.0])

    # Design scene
    scene_num_envs = args_cli.num_envs
    if simulation_app.is_newton_mjwarp and uses_newton_visualizer and hasattr(args_cli, "max_visible_envs"):
        scene_num_envs = min(scene_num_envs, args_cli.max_visible_envs)
        if scene_num_envs != args_cli.num_envs:
            print(f"[INFO]: Limiting Newton viewer scene to {scene_num_envs} envs (requested {args_cli.num_envs}).")
    scene_cfg = MultiObjectSceneCfg(num_envs=scene_num_envs, env_spacing=2.0, replicate_physics=True)
    scene_cfg = make_mjwarp_scene_homogeneous(scene_cfg)
    scene_cfg = simulation_app.configure_scene_cfg(scene_cfg)
    with Timer("[INFO] Time to create scene: "):
        scene = InteractiveScene(scene_cfg)

    with Timer("[INFO] Time to randomize scene: "):
        # DO YOUR OWN OTHER KIND OF RANDOMIZATION HERE!
        # Note: Just need to acquire the right attribute about the property you want to set
        # Here is an example on setting color randomly
        randomize_shape_color(scene_cfg.object.prim_path)

    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main execution
    main()
    # close sim app
    simulation_app.close()
