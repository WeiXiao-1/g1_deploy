"""Standalone G1 29DOF DDS subscriber and local read-only web monitor."""
import argparse
import math
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
import signal
import sqlite3
import threading
import time
from urllib.parse import parse_qs, urlsplit

if __package__:
    from .run_history import RunHistory, history_path
else:
    from run_history import RunHistory, history_path


JOINT_NAMES = (
    [f"{side}_{joint}_joint" for side in ("left", "right")
     for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + [f"waist_{joint}_joint" for joint in ("yaw", "roll", "pitch")]
    + [f"{side}_{joint}_joint" for side in ("left", "right")
       for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow",
                     "wrist_roll", "wrist_pitch", "wrist_yaw")]
)


def read_imu(msg):
    """LowState quaternion is w,x,y,z; derive RPY for bridges omitting rpy."""
    return read_imu_state(getattr(msg, "imu_state", None))


def read_imu_state(source):
    if source is None:
        return None
    def vector(name, size):
        values = [float(v) for v in getattr(source, name, [])]
        return values if len(values) == size and all(math.isfinite(v) for v in values) else None
    quaternion = vector("quaternion", 4)
    rpy = None
    if quaternion is not None:
        norm = math.sqrt(sum(v * v for v in quaternion))
        if norm > 1e-9:
            w, x, y, z = [v / norm for v in quaternion]
            rpy = [math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y)),
                   math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x)))),
                   math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))]
    return {"quaternion": quaternion, "rpy": rpy,
            "gyroscope": vector("gyroscope", 3),
            "accelerometer": vector("accelerometer", 3)}


def finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


class JointMonitor:
    def __init__(self, host="127.0.0.1", port=8765, rate_hz=50, *, record_rate=250, database=None,
                 source="DDS"):
        if not math.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("Joint monitor rate must be positive")
        if not math.isfinite(record_rate) or not 0 < record_rate <= 500:
            raise ValueError("Joint recording rate must be in (0, 500] Hz")
        self.period = 1.0 / rate_hz
        self.record_period = 1.0 / record_rate
        self.source = source
        self.last_capture = None
        self.torso_imu = None
        self.torso_at = None
        self.torso_count = 0
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self.stopping = threading.Event()
        self.snapshot = {"sample_id": 0, "simulation_time": 0, "joints": []}
        self.names = JOINT_NAMES
        self.command = None
        self.command_at = None
        self.command_info = None
        self.target = None
        self.target_at = None
        self.dry_target = None
        self.dry_target_at = None
        self.enable = None
        self.enable_at = None
        self.started_at = time.monotonic()
        self.next_capture = self.started_at
        self.next_record = self.started_at
        self.history = None
        self.history_error = None
        self.html = (Path(__file__).parent / "static" / "joint_monitor.html").read_bytes()
        monitor = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                request = urlsplit(self.path)
                path = request.path
                if path == "/api/stream":
                    self.stream()
                    return
                if path in ("/", "/index.html"):
                    content, mime = monitor.html, "text/html; charset=utf-8"
                elif path == "/api/joints":
                    with monitor.lock:
                        state = monitor.snapshot_locked()
                    self.send_json(state)
                    return
                elif path == "/api/runs" or path.startswith("/api/runs/"):
                    self.runs(path, request.query)
                    return
                else:
                    self.send_error(404)
                    return
                self.send_content(content, mime)

            def send_json(self, data):
                self.send_content(json.dumps(data, allow_nan=False, separators=(",", ":")).encode(),
                                  "application/json; charset=utf-8")

            def send_content(self, content, mime):
                self.send_response(200)
                self.send_header("Content-Type", mime)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def runs(self, path, query):
                history = monitor.history
                if history is None:
                    self.send_error(503, "Joint history unavailable")
                    return
                try:
                    if path == "/api/runs":
                        self.send_json({"runs": history.runs(), **history.status()})
                        return
                    route = re.fullmatch(r"/api/runs/([a-f0-9]{32})/(window|joints\.csv)", path)
                    if route is None or not history.exists(route[1]):
                        self.send_error(404)
                        return
                    if route[2] == "joints.csv":
                        self.download(route[1])
                        return
                    params = parse_qs(query, max_num_fields=4)
                    end = float(params["end"][0])
                    duration = float(params.get("seconds", ["5"])[0])
                    joints = sorted(set(map(int, params["joints"][0].split(",")))) if "joints" in params else []
                    if not 0 <= end <= 1e9 or not 0 < duration <= 20 or any(
                            index < 0 or index >= len(monitor.names) for index in joints):
                        raise ValueError("Invalid replay window")
                    self.send_json(history.window(route[1], round(max(0, end-duration)*1e6), round(end*1e6), joints))
                except (KeyError, ValueError):
                    self.send_error(400, "Invalid replay window")
                except sqlite3.Error:
                    self.send_error(503, "Joint history unavailable")

            def download(self, run_id):
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="g1-joints-{run_id}.csv"')
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.connection.settimeout(5)
                try:
                    for chunk in monitor.history.csv_chunks(run_id):
                        self.wfile.write(chunk)
                except (OSError, sqlite3.Error):
                    pass

            def log_message(self, *_):
                pass

            def stream(self):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Accel-Buffering", "no")
                self.end_headers()
                self.connection.settimeout(5)
                previous = -1
                try:
                    while not monitor.stopping.is_set():
                        with monitor.changed:
                            monitor.changed.wait_for(
                                lambda: monitor.stopping.is_set() or
                                monitor.snapshot["sample_id"] != previous, timeout=1)
                            if monitor.stopping.is_set():
                                break
                            previous = monitor.snapshot["sample_id"]
                            state = monitor.snapshot_locked()
                        content = json.dumps(state, allow_nan=False, separators=(",", ":"))
                        self.wfile.write(("data: " + content + "\n\n").encode())
                        self.wfile.flush()
                except (OSError, TimeoutError):
                    pass

        self.server = ThreadingHTTPServer((host, port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       name="joint-monitor", daemon=True)
        try:
            self.history = RunHistory(database if database is not None else history_path(), record_rate)
        except (OSError, RuntimeError, sqlite3.Error) as exc:
            self.history_error = str(exc)

    def start(self):
        self.thread.start()
        host, port = self.server.server_address[:2]
        print(f"Joint monitor: http://{host}:{port}", flush=True)
        if self.history is not None:
            print(f"Joint history: {self.history.path} (up to {self.history.rate:g} Hz)", flush=True)
        else:
            print(f"Joint history unavailable: {self.history_error}", flush=True)

    def snapshot_locked(self):
        # Called with self.lock held. Recompute torso freshness even if LowState stops.
        state = dict(self.snapshot)
        now = time.monotonic()
        state["age_ms"] = None if self.last_capture is None else round((now-self.last_capture)*1000, 1)
        state["torso_imu"] = self.torso_snapshot_locked(now)
        state["history"] = self.history.status() if self.history is not None else {"error": self.history_error}
        return state

    def torso_snapshot_locked(self, now):
        age = None if self.torso_at is None else max(0, (now-self.torso_at)*1000)
        return {"topic": "rt/secondary_imu", "age_ms": None if age is None else round(age, 1),
                "fresh": age is not None and age < 100 and self.torso_imu is not None,
                "valid": self.torso_imu is not None, "samples": self.torso_count, "data": self.torso_imu}

    def on_torso_imu(self, msg):
        data = read_imu_state(msg)
        if data is not None:
            q = data["quaternion"]
            norm = sum(v*v for v in q) if q is not None else 0
            if not .5 <= norm <= 1.5 or data["gyroscope"] is None:
                data = None
            else:
                w,x,y,z = q
                data["projected_gravity"] = [2*(w*y-x*z)/norm, -2*(w*x+y*z)/norm,
                                             (2*(x*x+y*y)-norm)/norm]
        with self.changed:
            self.torso_imu = data
            self.torso_at = time.monotonic()
            self.torso_count += 1
            self.changed.notify_all()

    def on_command(self, msg):
        if len(msg.motor_cmd) < len(self.names):
            return
        command = [tuple(float(getattr(m, k)) for k in ("q", "dq", "kp", "kd", "tau"))
                   for m in msg.motor_cmd[:len(self.names)]]
        if not all(math.isfinite(v) for motor in command for v in motor):
            return
        with self.lock:
            self.command = command
            self.command_at = time.monotonic()
            self.command_info = getattr(msg, "sample_info", None)

    def on_state(self, msg):
        if len(msg.motor_state) < len(self.names):
            return
        if getattr(msg, "mode_machine", 0) not in (0, 5):
            return  # refuse known non-29DOF modes
        with self.lock:
            now = time.monotonic()
            live = now >= self.next_capture
            recording = self.history is not None and now >= self.next_record
            if not live and not recording:
                return
            if live:
                self.next_capture += (int((now - self.next_capture) / self.period) + 1) * self.period
            if recording:
                self.next_record += (int((now - self.next_record) / self.record_period) + 1) * self.record_period
            command_age = None if self.command_at is None else (now - self.command_at) * 1000
            command = self.command if command_age is not None and command_age < 500 else None
            command_info = self.command_info if command is not None else None
            enable = self.enable if self.enable_at is not None and now - self.enable_at < .5 else None
            dry_run = (self.dry_target_at is not None and now - self.dry_target_at < .5
                       and not (enable and enable[0]))
            if dry_run:
                desired, desired_at, source = self.dry_target, self.dry_target_at, "preview"
            elif self.target_at is not None and now - self.target_at < .5:
                desired, desired_at, source = self.target, self.target_at, "controller_target"
            else:
                desired, desired_at, source = command, self.command_at, "lowcmd" if command else None
            torso = self.torso_snapshot_locked(now) if recording else None
        joints = []
        for i, name in enumerate(self.names):
            motor = msg.motor_state[i]
            q, dq, tau = finite(motor.q), finite(motor.dq), finite(motor.tau_est)
            q_target, dq_target, kp, kd, _ = desired[i] if desired else (None,) * 5
            q_cmd, dq_cmd, kp_cmd, kd_cmd, tau_cmd = command[i] if command else (None,) * 5
            joints.append({
                "index": i, "name": name, "q": q, "dq": dq, "tau": tau,
                "q_target": q_target, "dq_target": dq_target,
                "kp": kp, "kd": kd, "error": q_target - q if q is not None and q_target is not None else None,
                "q_cmd": q_cmd, "dq_cmd": dq_cmd, "kp_cmd": kp_cmd, "kd_cmd": kd_cmd, "tau_cmd": tau_cmd,
                "target_available": desired is not None, "enabled": enable[i] if enable else None,
                "active": command is not None and any((kp_cmd, kd_cmd, tau_cmd)),
            })
        state = {"dry_run": dry_run,
                 "simulation_time": now - self.started_at, "joints": joints,
                 "output_enabled": enable[0] if enable else None, "imu": read_imu(msg),
                 "lowstate_tick": getattr(msg, "tick", None), "command_age_ms": command_age,
                 "target_source": source, "target_age_ms": (now - desired_at)*1000 if desired else None}
        if recording:
            state_info = getattr(msg, "sample_info", None)
            self.history.capture(dict(state, wall_ns=time.time_ns(), age_ms=0, torso_imu=torso,
                                      lowstate_dds_ns=getattr(state_info, "source_timestamp", None),
                                      lowcmd_dds_ns=getattr(command_info, "source_timestamp", None),
                                      lowcmd_publisher=getattr(command_info, "publication_handle", None)), now)
        if live:
            self.publish_snapshot(state, now)

    def publish_snapshot(self, state, now):
        try:
            json.dumps(state, allow_nan=False)
        except ValueError:
            return
        with self.changed:
            if self.last_capture is not None and now < self.last_capture:
                return
            self.snapshot = dict(state, sample_id=self.snapshot["sample_id"] + 1)
            self.last_capture = now
            self.changed.notify_all()

    def record_simulation(self, state, now):
        if self.history is None:
            return
        with self.lock:
            torso = self.torso_snapshot_locked(now)
        enabled = state["output_enabled"]
        if enabled:
            self.history.signal(self.source, enabled, now)
        self.history.capture(dict(state, wall_ns=time.time_ns(), age_ms=0, torso_imu=torso), now)
        if not enabled:
            self.history.signal(self.source, enabled, now)

    def on_target(self, msg):
        if len(msg.motor_cmd) < len(self.names):
            return
        target = [tuple(float(getattr(m, k)) for k in ("q", "dq", "kp", "kd", "tau"))
                  for m in msg.motor_cmd[:len(self.names)]]
        if not all(math.isfinite(v) for motor in target for v in motor):
            return
        with self.lock:
            self.target = target
            self.target_at = time.monotonic()

    def on_enable(self, msg):
        if len(msg.motor_cmd) < len(self.names):
            return
        modes = [int(m.mode) for m in msg.motor_cmd[:len(self.names)]]
        if any(mode not in (0, 1) for mode in modes):
            return
        enabled = [mode == 1 for mode in modes]
        publisher = getattr(getattr(msg, "sample_info", None), "publication_handle", None)
        with self.lock:
            self.enable = enabled
            now = self.enable_at = time.monotonic()
            if self.history is not None:
                self.history.signal(self.source, enabled[0], now, start_when_disabled=True, publisher=publisher)

    def on_dry_target(self, msg):
        if len(msg.motor_cmd) < len(self.names):
            return
        target = [tuple(float(getattr(m, k)) for k in ("q", "dq", "kp", "kd", "tau"))
                  for m in msg.motor_cmd[:len(self.names)]]
        if not all(math.isfinite(v) for motor in target for v in motor):
            return
        with self.lock:
            self.dry_target = target
            self.dry_target_at = time.monotonic()

    def close(self):
        self.stopping.set()
        with self.changed:
            self.changed.notify_all()
        if self.thread.is_alive():
            self.server.shutdown()
            self.thread.join(timeout=2)
        self.server.server_close()
        if self.history is not None:
            self.history.close()


def main():
    parser = argparse.ArgumentParser(description="G1 29DOF standalone web joint monitor")
    parser.add_argument("--domain", type=int, default=1)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--rate", type=float, default=50,
                        help="Live display sampling limit in Hz (default: 50)")
    parser.add_argument("--record-rate", type=float, default=250,
                        help="Joint history sampling limit in Hz (default: 250, maximum: 500)")
    parser.add_argument("--history-db", type=Path, default=history_path(), help="SQLite file for joint runs")
    args = parser.parse_args()
    if args.domain < 0 or not 1 <= args.port <= 65535 or not math.isfinite(args.rate) or args.rate <= 0:
        parser.error("domain must be nonnegative; rate must be positive; port must be 1..65535")
    if not math.isfinite(args.record_rate) or not 0 < args.record_rate <= 500:
        parser.error("record-rate must be in (0, 500]")
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_, LowCmd_, IMUState_

    monitor = JointMonitor(args.host, args.port, args.rate, record_rate=args.record_rate,
                           database=args.history_db, source=f"DDS {args.interface} · domain {args.domain}")
    exit_requested = False

    def request_exit(*_):
        nonlocal exit_requested
        exit_requested = True

    signal.signal(signal.SIGTERM, request_exit)
    subscribers = []
    try:
        ChannelFactoryInitialize(args.domain, args.interface)
        for topic, message_type, callback in (
                ("rt/lowstate", LowState_, monitor.on_state),
                ("rt/secondary_imu", IMUState_, monitor.on_torso_imu),
                ("rt/lowcmd", LowCmd_, monitor.on_command),
                ("rt/g1/reset_target", LowCmd_, monitor.on_target),
                ("rt/g1/dry_run_target", LowCmd_, monitor.on_dry_target),
                ("rt/g1/reset_enable", LowCmd_, monitor.on_enable)):
            subscriber = ChannelSubscriber(topic, message_type)
            subscribers.append(subscriber)
            subscriber.Init(callback, 1)
        monitor.start()
        print(f"Waiting for G1 29DOF DDS data: domain={args.domain}, interface={args.interface}. Ctrl+C to exit.", flush=True)
        while not exit_requested:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        for subscriber in subscribers:
            subscriber.Close()
        monitor.close()


if __name__ == "__main__":
    main()
