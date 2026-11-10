"""
LineCrossing and Multi-Camera Event Fusion Engine.

Implements:
1. LineCrossing: Per-camera single line crossing detection with dead-band margin
   and cooldown to eliminate jitter double-counts.
2. Fusion: Multi-camera temporal clustering with 2-of-3 consensus commit rule,
   late third-camera absorption, and single-camera unconfirmed logging.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)


class LineCrossing:
    """
    Per-camera single crossing line with dead-band margin, K-tick confirmation,
    and direction state machine to eliminate jitter double-counts.
    
    A line is defined by two points p1 and p2 in pixel coordinates.
    The normal vector n = [-d[1], d[0]] / norm(d) points toward side +1.
    - dist > +margin  -> side = +1  (typically LOADED / entering truck)
    - dist < -margin  -> side = -1  (typically RETURNED / dock side)
    - -margin <= dist <= +margin -> side = 0 (dead-band: ignore jitter)
    """

    def __init__(
        self,
        cam: str,
        p1: Sequence[float],
        p2: Sequence[float],
        margin: float = 12.0,
        cooldown: float = 1.0,
        k_confirm: int = 1,
    ) -> None:
        self.cam = str(cam)
        self.margin = float(margin)
        self.cooldown = float(cooldown)
        self.k_confirm = max(1, int(k_confirm))

        p1_arr = np.array(p1, dtype=float)
        p2_arr = np.array(p2, dtype=float)
        d = p2_arr - p1_arr
        norm = np.linalg.norm(d)
        if norm < 1e-6:
            raise ValueError(f"Line points p1={p1} and p2={p2} are degenerate for camera {cam}")

        self.p1 = p1_arr
        self.p2 = p2_arr
        self.n = np.array([-d[1], d[0]], dtype=float) / norm  # Line normal vector
        self.confirmed_side: Dict[Any, int] = {}
        self.candidate_side: Dict[Any, int] = {}
        self.candidate_ticks: Dict[Any, int] = {}
        self.last_evt: Dict[Any, float] = {}
        self.last_dir: Dict[Any, str] = {}

    @property
    def last_side(self) -> Dict[Any, int]:
        """Backward compatibility alias for callers inspecting confirmed side."""
        return self.confirmed_side

    def reset(self) -> None:
        """Reset internal tracking state for a new session."""
        self.confirmed_side.clear()
        self.candidate_side.clear()
        self.candidate_ticks.clear()
        self.last_evt.clear()
        self.last_dir.clear()

    def update(
        self,
        t: float,
        tracks: Sequence[Union[Tuple, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Update tracker positions at timestamp t (frame presentation timestamp / PTS).
        
        Args:
            t: Current frame PTS timestamp (seconds).
            tracks: List of tracks. Supports:
                - (track_id, cx, cy)
                - (track_id, cx, cy, qty)
                - Object with .local_track_id/.track_id, .center (x, y), and optional .estimated_qty

        Returns:
            List of crossing event dicts.
        """
        events: List[Dict[str, Any]] = []

        for item in tracks:
            # Parse track payload flexibly
            if isinstance(item, (tuple, list)):
                tid = item[0]
                cx, cy = float(item[1]), float(item[2])
                qty = int(item[3]) if len(item) > 3 else 1
            else:
                tid = getattr(item, "local_track_id", getattr(item, "track_id", None))
                center = getattr(item, "center", (0.0, 0.0))
                cx, cy = float(center[0]), float(center[1])
                qty = int(getattr(item, "estimated_qty", 1))

            if tid is None:
                continue

            dist = float(np.dot(np.array([cx, cy], dtype=float) - self.p1, self.n))
            side = 1 if dist > self.margin else -1 if dist < -self.margin else 0

            # Dead-band check: ignore tracks within dead-band to prevent jitter
            if side == 0:
                self.candidate_side.pop(tid, None)
                self.candidate_ticks.pop(tid, None)
                continue

            prev = self.confirmed_side.get(tid)

            # First observation outside deadband: establish confirmed side
            if prev is None:
                self.confirmed_side[tid] = side
                self.candidate_side.pop(tid, None)
                self.candidate_ticks.pop(tid, None)
                continue

            # Track is on the confirmed side: clear any transient candidate
            if side == prev:
                self.candidate_side.pop(tid, None)
                self.candidate_ticks.pop(tid, None)
                continue

            # Track is on the opposite side of confirmed_side: track candidate ticks
            if self.candidate_side.get(tid) != side:
                self.candidate_side[tid] = side
                self.candidate_ticks[tid] = 1
            else:
                self.candidate_ticks[tid] += 1

            if self.candidate_ticks[tid] >= self.k_confirm:
                last_t = self.last_evt.get(tid, -999999.0)
                target_dir = "LOADED" if side == 1 else "RETURNED"
                last_dir = self.last_dir.get(tid)

                cooldown_ok = (t - last_t) > self.cooldown
                direction_ok = (last_dir != target_dir)

                if cooldown_ok and direction_ok:
                    events.append({
                        "cam": self.cam,
                        "t": float(t),
                        "track": tid,
                        "dir": target_dir,
                        "qty": max(1, qty),
                        "dist": round(dist, 2),
                    })
                    self.last_evt[tid] = t
                    self.last_dir[tid] = target_dir
                    self.confirmed_side[tid] = side
                    self.candidate_side.pop(tid, None)
                    self.candidate_ticks.pop(tid, None)
                else:
                    # Suppressed crossing (cooldown or direction invariant).
                    # CRITICAL BUGFIX: We do NOT update self.confirmed_side[tid]!
                    # The track remains confirmed on `prev` so oscillation
                    # cannot flip the state machine and trigger duplicate events.
                    pass

        return events


class Fusion:
    """
    Multi-Camera Event Fusion Engine.
    
    Clusters single-camera crossing events within a temporal window.
    - If len(cams) >= min_cams (e.g. 2 of 3 cameras agree): commits count immediately.
    - If 3rd camera arrives late within window: absorbs it into existing cluster,
      preventing double-counting.
    - If only 1 camera observes crossing: moves to unconfirmed list on flush,
      keeping verified count strict.
    """

    def __init__(self, window: float = 1.0, min_cams: int = 2) -> None:
        self.window = float(window)
        self.min_cams = int(min_cams)
        self.clusters: List[Dict[str, Any]] = []
        self.unconfirmed: List[Dict[str, Any]] = []
        self.ledger: List[Dict[str, Any]] = []
        self.loaded: int = 0
        self.returned: int = 0

    def reset(self) -> None:
        """Reset all counters and clusters."""
        self.clusters.clear()
        self.unconfirmed.clear()
        self.ledger.clear()
        self.loaded = 0
        self.returned = 0

    def add(self, ev: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Add a single-camera crossing event to the fusion clusters.
        
        Args:
            ev: Crossing event with keys 'cam', 't', 'dir', and optional 'qty', 'track'.

        Returns:
            The matched or newly created cluster.
        """
        ev_dir = ev["dir"]
        ev_cam = ev["cam"]
        ev_t = ev["t"]
        ev_qty = ev.get("qty", 1)

        # Look for matching active cluster
        for c in self.clusters:
            if (
                c["dir"] == ev_dir
                and ev_cam not in c["cams"]
                and abs(ev_t - c["t0"]) <= self.window
            ):
                c["cams"][ev_cam] = ev
                c["qty"] = max(c.get("qty", 1), ev_qty)
                self._maybe_commit(c)
                return c

        # Create new candidate cluster
        c = {
            "dir": ev_dir,
            "t0": ev_t,
            "last_t": ev_t,
            "cams": {ev_cam: ev},
            "qty": ev_qty,
            "committed": False,
        }
        self.clusters.append(c)
        self._maybe_commit(c)
        return c

    def _maybe_commit(self, c: Dict[str, Any]) -> bool:
        """Commit count if min_cams threshold is satisfied and not yet committed."""
        if not c["committed"] and len(c["cams"]) >= self.min_cams:
            c["committed"] = True
            qty = c.get("qty", 1)
            if c["dir"] == "LOADED":
                self.loaded += qty
            else:
                self.returned += qty
            self.ledger.append({
                "t": c["t0"],
                "dir": c["dir"],
                "cams": sorted(list(c["cams"].keys())),
                "qty": qty,
                "events": dict(c["cams"]),
            })
            return True
        return False

    def flush(self, now: float) -> List[Dict[str, Any]]:
        """
        Retires old clusters past window + grace period.
        Uncommitted single-camera clusters are recorded in unconfirmed list.
        
        Returns:
            List of newly expired unconfirmed clusters.
        """
        new_unconfirmed: List[Dict[str, Any]] = []
        for c in list(self.clusters):
            if now - c["t0"] > (self.window + 0.5):
                if not c["committed"]:
                    self.unconfirmed.append(c)
                    new_unconfirmed.append(c)
                self.clusters.remove(c)
        return new_unconfirmed

    @property
    def net(self) -> int:
        """Net cartons loaded = loaded - returned."""
        return self.loaded - self.returned

    def summary(self) -> Dict[str, Any]:
        """Summary of current fusion state."""
        return {
            "loaded": self.loaded,
            "returned": self.returned,
            "net": self.net,
            "committed_events": len(self.ledger),
            "unconfirmed_events": len(self.unconfirmed),
            "active_clusters": len(self.clusters),
        }
