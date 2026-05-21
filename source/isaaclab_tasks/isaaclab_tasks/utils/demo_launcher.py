# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared launcher utilities for demo scripts under scripts/demos."""

from __future__ import annotations

import argparse
import copy
from contextlib import AbstractContextManager
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

from .hydra import parse_overrides
from .preset_cli import fold_preset_tokens
from .sim_launcher import add_launcher_args, launch_simulation

__all__ = ["DemoAppLauncher", "create_demo_physics_cfg", "tune_mjwarp_articulation_cfg"]

if TYPE_CHECKING:
    from isaaclab_newton.physics import NewtonCfg

    from isaaclab.assets import ArticulationCfg
    from isaaclab.physics.physics_manager_cfg import PhysicsCfg
    from isaaclab.sim import SimulationCfg, SimulationContext

# Conservative armature defaults used by demos to stabilize MJWarp articulation examples.
MJWARP_ARMATURE = 0.02
MJWARP_GRIPPER_ARMATURE = 0.005
PHYSICS_PRESETS = ("physx", "newton_mjwarp")
_PHYSICS_PRESET_MAP = {"env": {"demo.physics": {preset: None for preset in PHYSICS_PRESETS}}, "agent": {}}


def create_demo_physics_cfg(physics: str, *, newton_cfg: NewtonCfg | None = None) -> PhysicsCfg:
    """Create the selected demo physics backend configuration.

    Args:
        physics: Name of the physics preset to create.
        newton_cfg: Newton MJWarp configuration to use instead of the demo default.
            The configuration is copied before it is returned.

    Returns:
        Physics backend configuration for the selected preset.
    """
    if physics == "physx":
        from isaaclab_physx.physics import PhysxCfg

        return PhysxCfg()

    if physics != "newton_mjwarp":
        raise ValueError(f"Unsupported physics preset: {physics}")

    if newton_cfg is not None:
        return copy.deepcopy(newton_cfg)

    from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

    return NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=512,
            nconmax=256,
            iterations=100,
            ls_iterations=50,
            solver="newton",
            ls_parallel=False,
            cone="elliptic",
            impratio=10,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
    )


def tune_mjwarp_articulation_cfg(cfg: ArticulationCfg) -> ArticulationCfg:
    """Tune articulation actuator armature for the MJWarp solver."""
    for actuator_name, actuator_cfg in getattr(cfg, "actuators", {}).items():
        if any(token in actuator_name.lower() for token in ("finger", "gripper", "hand")):
            actuator_cfg.armature = MJWARP_GRIPPER_ARMATURE
        else:
            actuator_cfg.armature = MJWARP_ARMATURE
    return cfg


class DemoAppLauncher:
    """Proxy layer shared by demo scripts.

    The proxy keeps demo scripts close to their original launcher shape while supporting
    the Newton MJWarp backend and Kit-less visualizers through :func:`launch_simulation`.
    """

    def __init__(
        self,
        args_cli: argparse.Namespace,
        *,
        kit_required: bool = False,
        newton_cfg: NewtonCfg | None = None,
    ):
        """Initialize the demo launcher proxy.

        Args:
            args_cli: Parsed demo command line arguments.
            kit_required: Whether the script imports or uses Kit APIs before a simulation config exists.
            newton_cfg: Newton MJWarp configuration to use instead of the demo default when
                ``physics=newton_mjwarp`` is selected.
        """
        self.args_cli = args_cli
        self.kit_required = kit_required
        self._newton_cfg = newton_cfg
        self._app: Any | None = None
        self._launch_context: AbstractContextManager | None = None
        self._sim: SimulationContext | None = None
        self._saw_visualizers = False

        if self._needs_early_app_launch():
            self._launch_app()

    @staticmethod
    def parse_args(parser: argparse.ArgumentParser) -> argparse.Namespace:
        """Parse demo arguments."""
        parser_defaults = dict(parser._defaults)
        add_launcher_args(parser)
        if parser_defaults:
            parser.set_defaults(**parser_defaults)

        if "physics" not in parser_defaults:
            parser.set_defaults(physics="physx")

        args_cli, unknown_args = parser.parse_known_args()
        folded_tokens = fold_preset_tokens(list(unknown_args))
        physics_presets, _, _, _ = parse_overrides(folded_tokens, _PHYSICS_PRESET_MAP)

        if physics_presets:
            args_cli.physics = physics_presets[0]
        else:
            default_physics, _, _, _ = parse_overrides([f"presets={args_cli.physics}"], _PHYSICS_PRESET_MAP)
            args_cli.physics = default_physics[0]

        return args_cli

    @property
    def app(self):
        """Return the underlying Kit app, if one was launched."""
        return self._app

    @property
    def is_newton_mjwarp(self) -> bool:
        """Whether the selected demo physics backend is Newton MJWarp."""
        return self.args_cli.physics == "newton_mjwarp"

    def configure_sim(self, sim_cfg: SimulationCfg, *, newton_cfg: NewtonCfg | None = None) -> SimulationCfg:
        """Apply selected physics and launch the required simulation runtime.

        Args:
            sim_cfg: Simulation configuration to update.
            newton_cfg: Newton MJWarp configuration to use instead of the demo default.

        Returns:
            Updated simulation configuration.
        """
        self._configure_sim_physics(sim_cfg, newton_cfg=newton_cfg)
        self._launch_runtime(SimpleNamespace(sim=sim_cfg))
        return sim_cfg

    def configure_env_cfg(self, env_cfg, *, newton_cfg: NewtonCfg | None = None):
        """Apply selected physics to ``env_cfg.sim`` and launch the required runtime.

        Args:
            env_cfg: Environment configuration to update.
            newton_cfg: Newton MJWarp configuration to use instead of the demo default.

        Returns:
            Updated environment configuration.
        """
        if not self.is_newton_mjwarp:
            self._launch_runtime(env_cfg)
            return env_cfg

        from .hydra import resolve_presets

        env_cfg = resolve_presets(env_cfg, {self.args_cli.physics})
        sim_cfg = getattr(env_cfg, "sim", None)
        if sim_cfg is not None:
            self._configure_sim_physics(sim_cfg, newton_cfg=newton_cfg)
            scene_cfg = getattr(env_cfg, "scene", None)
            if scene_cfg is not None:
                self.configure_scene_cfg(scene_cfg)
            for asset_cfg in env_cfg.__dict__.values():
                self._configure_scene_asset_cfg(asset_cfg)
        self._launch_runtime(env_cfg)
        return env_cfg

    def configure_scene_cfg(self, scene_cfg):
        """Apply Newton MJWarp demo adjustments to an interactive scene config."""
        if self.is_newton_mjwarp:
            for asset_cfg in scene_cfg.__dict__.values():
                self._configure_scene_asset_cfg(asset_cfg)
        return scene_cfg

    def create_context(
        self,
        sim_cfg: SimulationCfg,
        context_cls: type[SimulationContext] | None = None,
        *,
        newton_cfg: NewtonCfg | None = None,
    ) -> SimulationContext:
        """Create and bind a :class:`~isaaclab.sim.SimulationContext` for the demo.

        Args:
            sim_cfg: Simulation configuration used to create the context.
            context_cls: Simulation context class to instantiate.
            newton_cfg: Newton MJWarp configuration to use instead of the demo default.

        Returns:
            Created simulation context.
        """
        sim_cfg = self.configure_sim(sim_cfg, newton_cfg=newton_cfg)
        if context_cls is None:
            from isaaclab.sim import SimulationContext

            context_cls = SimulationContext
        return self.bind_sim(context_cls(sim_cfg))

    def bind_sim(self, sim: SimulationContext) -> SimulationContext:
        """Bind an externally created simulation context to the launcher proxy."""
        self._sim = sim
        self._saw_visualizers = self._saw_visualizers or bool(sim.visualizers)
        return sim

    def tune_articulation_cfg(self, cfg: ArticulationCfg) -> ArticulationCfg:
        """Apply MJWarp articulation tuning when the Newton backend is selected."""
        if self.is_newton_mjwarp:
            return tune_mjwarp_articulation_cfg(cfg)
        return cfg

    def is_running(self) -> bool:
        """Return whether the demo should continue stepping."""
        if self._sim is not None:
            self._saw_visualizers = self._saw_visualizers or bool(self._sim.visualizers)
            if self._saw_visualizers:
                return any(visualizer.is_running() for visualizer in self._sim.visualizers)
        if self._app is not None:
            return self._app.is_running()
        return False

    def close(self) -> None:
        """Close the simulation runtime owned by the proxy."""
        if self._launch_context is not None:
            self._launch_context.__exit__(None, None, None)
            self._launch_context = None
            if self._app is None and self._sim is not None:
                type(self._sim).clear_instance()
                self._sim = None
        elif self._sim is not None and self._app is None:
            type(self._sim).clear_instance()
            self._sim = None

        if self._app is not None:
            self._app.close()
            self._app = None
            self._sim = None

    def _needs_early_app_launch(self) -> bool:
        """Return whether Kit must be launched before the rest of the script imports."""
        visualizers = getattr(self.args_cli, "visualizer", None) or []
        if isinstance(visualizers, str):
            visualizers = [token.strip() for token in visualizers.split(",") if token.strip()]
        visualizer_types = {str(visualizer).strip().lower() for visualizer in visualizers if str(visualizer).strip()}
        return self.kit_required or not self.is_newton_mjwarp or "kit" in visualizer_types or "none" in visualizer_types

    def _launch_app(self) -> None:
        """Launch Kit through :class:`~isaaclab.app.AppLauncher`."""
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(self.args_cli)
        self._app = app_launcher.app
        self._sync_sensor_cfg_modules()
        if hasattr(app_launcher, "device"):
            self.args_cli.device = app_launcher.device

    @staticmethod
    def _sync_sensor_cfg_modules() -> None:
        """Reload sensor config modules whose base config class was loaded during Kit startup."""
        import importlib
        import sys

        from isaaclab.sensors.sensor_base_cfg import SensorBaseCfg

        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith("isaaclab.sensors.") or not module_name.endswith("_cfg"):
                continue
            module_sensor_base = getattr(module, "SensorBaseCfg", None)
            if module_sensor_base is None or module_sensor_base is SensorBaseCfg:
                continue

            module = importlib.reload(module)
            module_parts = module_name.split(".")
            parent_packages = (".".join(module_parts[:i]) for i in range(1, len(module_parts)))
            for attr_name, attr in vars(module).items():
                if not isinstance(attr, type) or attr.__module__ != module_name:
                    continue
                for package_name in parent_packages:
                    package = sys.modules.get(package_name)
                    if package is not None and attr_name in vars(package):
                        setattr(package, attr_name, attr)

    def _launch_runtime(self, cfg: Any) -> None:
        """Launch the simulation runtime if this launcher has not done so yet."""
        if self._launch_context is not None:
            return

        self._launch_context = launch_simulation(cfg, self.args_cli)
        self._launch_context.__enter__()

    def _configure_scene_asset_cfg(self, asset_cfg: Any) -> None:
        """Apply MJWarp compatibility adjustments to a scene asset config."""
        if asset_cfg is None:
            return

        from isaaclab.assets import ArticulationCfg

        if isinstance(asset_cfg, ArticulationCfg):
            tune_mjwarp_articulation_cfg(asset_cfg)

    def _configure_sim_physics(self, sim_cfg: SimulationCfg, *, newton_cfg: NewtonCfg | None = None) -> None:
        """Apply selected demo physics to a simulation config."""
        if not self.is_newton_mjwarp:
            return

        from isaaclab_newton.physics import NewtonCfg

        selected_newton_cfg = newton_cfg if newton_cfg is not None else self._newton_cfg
        if selected_newton_cfg is None and isinstance(getattr(sim_cfg, "physics", None), NewtonCfg):
            return
        sim_cfg.physics = create_demo_physics_cfg("newton_mjwarp", newton_cfg=selected_newton_cfg)
