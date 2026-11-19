"""Run the G1 MuJoCo environment or the standalone GetUpMimic recovery cases.

The shared environment uses unitree_mujoco and DDS on lo. GetUpMimic uses
the training bundle's MJCF and steps its C++ policy in simulation time.
"""
import argparse
from contextlib import ExitStack
import os
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mujoco-root", type=Path, help="External unitree_mujoco checkout for the DDS environment")
    getup = parser.add_argument_group("GetUpMimic")
    getup_mode = getup.add_mutually_exclusive_group()
    getup_mode.add_argument("--getup-mimic", action="store_true", help="Run fixed fallen-pose recovery cases")
    getup_mode.add_argument("--model_test", "--model-test", action="store_true",
                            help="Run a single-frame get-up expert")
    getup.add_argument("--expert", help="model_test profile (faceup or backup, default: faceup)")
    getup.add_argument("--mjcf", type=Path, help="Path to g1.xml from the GetUpMimic training bundle")
    getup.add_argument("--pose", choices=("all", "backup", "faceup"),
                       help="Recovery cases to run (default: all for GetUpMimic, selected pose for model_test)")
    getup.add_argument("--headless", action="store_true", help="Run GetUpMimic without opening the viewer")
    getup.add_argument("--interactive", action="store_true", help="Use 9 to take over, P to get up and Space to stop")
    getup.add_argument("--hardware-config", action="store_true",
                       help="Use the real GetUpMimic control period and motion protections in MuJoCo")
    getup.add_argument("--monitor", nargs="?", const=8765, type=int, metavar="PORT",
                       help="Serve the joint dashboard on localhost (default port 8765)")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--depth", action=argparse.BooleanOptionalAction, default=True,
                        help="Publish Parkour depth observations (DWaQ ignores them)")
    parser.add_argument("--depth-preview", action="store_true",
                        help="Show raw and model-input depth images in the MuJoCo viewer's top-right corner")
    parser.add_argument("--follow", action="store_true",
                        help="Start with a camera tracking the robot torso")
    parser.add_argument("--terrain", default="original",
                        choices=("original", "mixed", "platforms", "stones", "gaps", "stairs", "hurdles"),
                        help="Obstacle course; mixed places categories in parallel lanes (default: original external scene)")
    parser.add_argument("--window-backend", choices=("auto", "x11", "wayland"), default="x11",
                        help="Window backend (default: x11); auto prefers X11/XWayland when DISPLAY is available")
    args = parser.parse_args()
    if args.expert is not None and (not args.model_test or not args.expert):
        parser.error("--expert requires --model_test and a profile name")
    if args.getup_mimic or args.model_test:
        if args.mjcf is None or args.mujoco_root is not None:
            parser.error("--getup-mimic and --model_test require --mjcf and run independently of --mujoco-root")
        if args.domain != 1 or args.terrain != "original" or args.depth_preview:
            parser.error("DDS domain, terrain generation and depth preview require --mujoco-root")
        if args.interactive and args.headless:
            parser.error("--interactive requires the MuJoCo viewer")
        if args.monitor is not None and not (args.hardware_config and args.interactive and 0 <= args.monitor <= 65535):
            parser.error("--monitor requires --hardware-config and --interactive, with a valid port")
        try:
            if not args.headless:
                select_window_backend(args.window_backend)
            from getup_mimic_sim import run

            expert = (args.expert or "faceup") if args.model_test else None
            return run(PROJECT_ROOT, args.mjcf, args.pose, args.headless, args.follow,
                       args.interactive, args.hardware_config, args.monitor, expert=expert)
        except ModuleNotFoundError as error:
            parser.exit(1, f"GetUpMimic requires a Python environment with MuJoCo and NumPy: {error}\n")
        except (KeyError, OSError, RuntimeError, ValueError) as error:
            parser.exit(1, f"GetUpMimic: {error}\n")
        except KeyboardInterrupt:
            parser.exit(130, "GetUpMimic interrupted\n")
    if (args.mjcf is not None or args.pose is not None or args.headless or args.interactive
            or args.hardware_config or args.monitor is not None):
        parser.error("Get-up simulation options require --getup-mimic or --model_test")
    if args.mujoco_root is None:
        parser.error("--mujoco-root is required for the DDS environment")
    if args.domain <= 0:
        parser.error("Use a positive simulation domain")
    if args.depth_preview and not args.depth:
        parser.error("--depth-preview requires depth; remove --no-depth")

    root = args.mujoco_root.resolve()
    simulation = root / "simulate_python"
    simulator = simulation / "unitree_mujoco.py"
    scene = root / "unitree_robots/g1/scene_29dof.xml"
    if not simulator.is_file():
        parser.error("Missing external simulate_python/unitree_mujoco.py")
    if not scene.is_file():
        parser.error("Missing external unitree_robots/g1/scene_29dof.xml")

    previous_path, previous_argv = sys.path[:], sys.argv
    previous_backend = os.environ.get("PYGLFW_LIBRARY_VARIANT")
    window_backend = select_window_backend(args.window_backend)
    bridge = original_publisher = None
    temporary_scene = None
    try:
        sys.path.insert(0, str(simulation))
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        if args.terrain != "original":
            from terrain import generate_scene
            temporary_scene = TemporaryDirectory(prefix="g1-terrain-")
            scene = Path(temporary_scene.name) / "scene.xml"
            sections = generate_scene(root, args.terrain, scene)
            for category, start, end, y in sections:
                print(f"Terrain {category}: x={start:.2f}..{end:.2f} m, lane center y={y:.2f} m, along +X", flush=True)
        import config

        if args.depth:
            import parkour_observation as bridge
            original_publisher = bridge.ParkourObservationPublisher
            project_config = PROJECT_ROOT / "apps/parkour/config/g1_parkour.json"

            class Publisher(original_publisher):
                def __init__(self, model, data, port=None, config_path=None):
                    super().__init__(model, data, config_path=project_config)

            bridge.ParkourObservationPublisher = Publisher

        config.ROBOT = "g1"
        config.ROBOT_SCENE = str(scene)
        config.PHYSICS_BACKEND = "mujoco"
        config.DOMAIN_ID, config.INTERFACE = args.domain, "lo"
        config.USE_JOYSTICK = 0
        config.PARKOUR_OBSERVATIONS = args.depth
        config.PARKOUR_OBSERVATION_PORT = None # 外部启动接口的旧参数，不再使用网络
        config.PARKOUR_DEPTH_PREVIEW = args.depth_preview
        print(f"Shared G1 simulation: DDS domain {args.domain}, interface lo, "
              f"depth {'on' if args.depth else 'off'}, window backend {window_backend}", flush=True)
        sys.argv = [str(simulator)]
        with ExitStack() as extensions:
            if args.follow:
                from viewer_camera import follow_robot
                extensions.enter_context(follow_robot())
            if args.depth_preview:
                from viewer_depth_overlay import install_depth_overlay
                extensions.enter_context(install_depth_overlay())
            runpy.run_path(str(simulator), run_name="__main__")
    finally:
        if bridge is not None and original_publisher is not None:
            bridge.ParkourObservationPublisher = original_publisher
        sys.path[:] = previous_path
        sys.argv = previous_argv
        if previous_backend is None:
            os.environ.pop("PYGLFW_LIBRARY_VARIANT", None)
        else:
            os.environ["PYGLFW_LIBRARY_VARIANT"] = previous_backend
        if temporary_scene is not None:
            temporary_scene.cleanup()


def select_window_backend(backend):
    if backend == "auto":
        backend = os.environ.get("PYGLFW_LIBRARY_VARIANT") or (
            "x11" if sys.platform == "linux" and os.environ.get("DISPLAY") else "auto"
        )
    # Select the GLFW wheel library before importing the simulator or viewer.
    if backend != "auto":
        os.environ["PYGLFW_LIBRARY_VARIANT"] = backend
    return backend


if __name__ == "__main__":
    raise SystemExit(main())
