"""Persist sampled joint telemetry independently of browser connections."""

from collections import deque
from contextlib import closing
import csv
import io
import json
import os
from pathlib import Path
from queue import Empty, SimpleQueue
import sqlite3
import threading
import time
from uuid import uuid4
import zlib


PRE_ROLL_SECONDS = 2.0
HEARTBEAT_TIMEOUT = 3.0
MAX_PENDING = 4096
WRITE_BATCH = 256
WRITE_INTERVAL = .05
JOINT_FIELDS = ("q", "dq", "tau", "q_target", "dq_target", "error", "kp", "kd",
                "q_cmd", "dq_cmd", "kp_cmd", "kd_cmd", "tau_cmd", "enabled", "active")


def history_path():
    override = os.environ.get("G1_JOINT_HISTORY")
    if override:
        return Path(override).expanduser()
    state_home = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local/state")
    return state_home / "g1_deploy/joint_runs.sqlite3"


class RunHistory:
    def __init__(self, path, rate):
        self.path = Path(path).expanduser().resolve()
        self.rate = rate
        self.lock = threading.Lock()
        self.pending = SimpleQueue()
        self.before = deque()
        self.replaced_publishers = deque(maxlen=8)
        self.run_id = None
        self.error = None
        self.closed = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, source TEXT NOT NULL, started_ns INTEGER NOT NULL,
                    ended_ns INTEGER, reason TEXT, rate REAL NOT NULL, dropped INTEGER NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS samples (
                    run_id TEXT NOT NULL, elapsed_us INTEGER NOT NULL, data BLOB NOT NULL,
                    PRIMARY KEY (run_id, elapsed_us)
                ) WITHOUT ROWID;
            """)
        self.writer = threading.Thread(target=self.write, name="joint-history", daemon=True)
        self.writer.start()

    def connect(self):
        db = sqlite3.connect(self.path, timeout=2)
        db.row_factory = sqlite3.Row
        return db

    def status(self):
        with self.lock:
            return {"id": self.run_id, "error": self.error, "rate": self.rate,
                    "dropped": self.dropped if self.run_id else 0}

    def _start(self, source, now, wall_ns):
        self.run_id = uuid4().hex
        self.dropped = 0
        while self.before and self.before[0][0] < now - PRE_ROLL_SECONDS:
            self.before.popleft()
        self.origin = self.before[0][0] if self.before else now
        started = self.before[0][1]["wall_ns"] if self.before else wall_ns
        self.pending.put(("start", self.run_id, source, started, self.rate))
        for captured, frame in self.before:
            self._enqueue(captured, frame)
        self.before.clear()

    def _finish(self, wall_ns, reason):
        self.pending.put(("finish", self.run_id, wall_ns, reason, self.dropped))
        self.run_id = None
        self.before.clear()

    def _expire(self, now):
        if self.run_id and now - self.last_signal >= HEARTBEAT_TIMEOUT:
            self._finish(time.time_ns(), "signal_timeout")

    def _enqueue(self, now, frame):
        if self.pending.qsize() >= MAX_PENDING:
            self.dropped += 1
            return
        elapsed = round((now - self.origin) * 1_000_000)
        if elapsed >= 0:
            self.pending.put(("sample", self.run_id, elapsed, frame))

    def signal(self, source, enabled, now, *, start_when_disabled=False, publisher=None):
        with self.lock:
            if self.closed or self.error:
                return
            self._expire(now)
            if publisher in self.replaced_publishers:
                return
            if self.run_id and publisher is not None and self.publisher not in (None, publisher):
                if self.enabled and not enabled:
                    return
                self.replaced_publishers.append(self.publisher)
                self._finish(time.time_ns(), "publisher_changed")
            if self.run_id and self.enabled and not enabled:
                self._finish(time.time_ns(), "output_disabled")
                return
            if self.run_id is None:
                if not enabled and not start_when_disabled:
                    return
                self._start(source, now, time.time_ns())
                self.publisher = None
            if publisher is not None:
                self.publisher = publisher
            self.last_signal = now
            self.enabled = enabled

    def capture(self, frame, now):
        with self.lock:
            if self.closed or self.error:
                return
            self._expire(now)
            if self.run_id:
                self._enqueue(now, frame)
            else:
                self.before.append((now, frame))
                while self.before[0][0] < now - PRE_ROLL_SECONDS:
                    self.before.popleft()

    def write(self):
        try:
            with closing(self.connect()) as db:
                while True:
                    try:
                        batch = [self.pending.get(timeout=.25)]
                    except Empty:
                        with self.lock:
                            self._expire(time.monotonic())
                        continue
                    deadline = time.monotonic() + WRITE_INTERVAL
                    while len(batch) < WRITE_BATCH and batch[-1] is not None:
                        try:
                            batch.append(self.pending.get(timeout=max(0., deadline - time.monotonic())))
                        except Empty:
                            break
                    with db:
                        for item in batch:
                            if item is None:
                                return
                            kind, run_id, *args = item
                            if kind == "start":
                                db.execute("INSERT INTO runs (id, source, started_ns, rate) VALUES (?, ?, ?, ?)",
                                           (run_id, *args))
                            elif kind == "finish":
                                db.execute("UPDATE runs SET ended_ns=?, reason=?, dropped=? WHERE id=?",
                                           (*args, run_id))
                            else:
                                elapsed, frame = args
                                payload = json.dumps(frame, separators=(",", ":"), allow_nan=False).encode()
                                db.execute("INSERT OR IGNORE INTO samples VALUES (?, ?, ?)",
                                           (run_id, elapsed, zlib.compress(payload, 1)))
        except (OSError, sqlite3.Error, ValueError) as exc:
            with self.lock:
                self.error = f"{type(exc).__name__}: {exc}"
                self.run_id = None
                self.before.clear()
                while not self.pending.empty():
                    self.pending.get_nowait()

    def close(self):
        with self.lock:
            self.closed = True
            if self.run_id and not self.error:
                self._finish(time.time_ns(), "monitor_stopped")
            self.pending.put(None)
        self.writer.join()

    def runs(self):
        with closing(self.connect()) as db:
            rows = db.execute("""
                SELECT runs.*, COALESCE((SELECT max(elapsed_us) FROM samples WHERE run_id=runs.id), 0)
                    AS duration_us FROM runs ORDER BY started_ns DESC
            """).fetchall()
        return [dict(row) for row in rows]

    def exists(self, run_id):
        with closing(self.connect()) as db:
            return db.execute("SELECT 1 FROM runs WHERE id=?", (run_id,)).fetchone() is not None

    def frames(self, db, run_id, begin, end, latest_only=False):
        order = "DESC LIMIT 1" if latest_only else "ASC"
        rows = db.execute("""
            SELECT elapsed_us, data FROM samples WHERE run_id=? AND elapsed_us BETWEEN ? AND ?
            ORDER BY elapsed_us
        """ + order, (run_id, begin, end))
        for elapsed, data in rows:
            yield elapsed / 1_000_000, json.loads(zlib.decompress(data))

    def window(self, run_id, begin, end, joints):
        points = {index: [] for index in joints}
        snapshot = None
        with closing(self.connect()) as db:
            for elapsed, frame in self.frames(db, run_id, begin, end, latest_only=not joints):
                snapshot = dict(frame, elapsed_s=elapsed)
                for index in joints:
                    joint = frame["joints"][index]
                    available = joint["target_available"]
                    points[index].append((elapsed, joint["q"], joint["q_target"] if available else None,
                                          joint["dq"], joint["dq_target"] if available else None,
                                          joint["tau"], joint.get("tau_cmd")))
        return {"snapshot": snapshot, "points": points}

    def csv_chunks(self, run_id):
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer)
        writer.writerow(("elapsed_s", "wall_time_unix_ns", "source_time_s", "lowstate_tick",
                         "lowstate_dds_ns", "lowcmd_dds_ns", "lowcmd_publisher",
                         "output_enabled", "dry_run", "simulation_phase", "simulation_fault",
                         "command_age_ms", "target_source", "target_age_ms", "joint_id", "joint_name",
                         "target_available", *JOINT_FIELDS))
        with closing(self.connect()) as db:
            # Fix the upper bound so a download of an active run has a finite endpoint.
            end = db.execute("SELECT max(elapsed_us) FROM samples WHERE run_id=?", (run_id,)).fetchone()[0]
            if end is not None:
                for elapsed, frame in self.frames(db, run_id, 0, end):
                    sim = frame.get("simulation", {})
                    common = (f"{elapsed:.6f}", frame["wall_ns"], frame["simulation_time"],
                              frame.get("lowstate_tick"), frame.get("lowstate_dds_ns"), frame.get("lowcmd_dds_ns"),
                              frame.get("lowcmd_publisher"), frame.get("output_enabled"), frame.get("dry_run"),
                              sim.get("phase"), sim.get("fault"), frame.get("command_age_ms"),
                              frame.get("target_source"), frame.get("target_age_ms"))
                    for joint in frame["joints"]:
                        values = [joint.get(field) for field in JOINT_FIELDS]
                        writer.writerow((*common, joint["index"], joint["name"], joint["target_available"], *values))
                    if buffer.tell() >= 65536:
                        yield buffer.getvalue().encode()
                        buffer.seek(0)
                        buffer.truncate()
        yield buffer.getvalue().encode()
