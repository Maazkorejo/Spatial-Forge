"""
Versioned edit session: propose -> (preview / diff) -> commit; undo / redo / goto.
propose() never mutates anything. Multi-operation proposals are ATOMIC: if any step fails the gate, nothing is
committed and the response says which operation failed and why.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cadquery as cq

import gate as G
from build import BuildError, build
from model import FeatureTree, OpError, apply_op, describe


@dataclass
class Snapshot:
    version: int
    tree: FeatureTree
    ops: list
    stats: dict
    trace: list
    warnings: list = field(default_factory=list)


@dataclass
class Proposal:
    ok: bool
    tree: Optional[FeatureTree] = None
    messages: list = field(default_factory=list)
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)
    diff: dict = field(default_factory=dict)
    failed_op: Optional[int] = None
    trace: list = field(default_factory=list)


def diff_solids(old: cq.Workplane, new: cq.Workplane) -> dict:
    out = {"volume_before": float(old.val().Volume()), "volume_after": float(new.val().Volume())}
    try:
        out["removed"] = float(old.val().cut(new.val()).Volume())
        out["added"] = float(new.val().cut(old.val()).Volume())
    except Exception:
        out["removed"] = out["added"] = None
    return out


class Session:
    def __init__(self, tree: FeatureTree, roundtrip: bool = True):
        self.roundtrip = roundtrip
        tree = tree.with_ids()
        trace: list = []
        solid = build(tree, trace)
        rep = G.check(tree, solid, trace, None, roundtrip)
        if not rep.ok:
            raise ValueError("initial model fails the gate: " + "; ".join(rep.errors))
        self.history = [Snapshot(0, tree, ["Initial model"], rep.stats, trace, rep.warnings)]
        self.cursor = 0

    # ------------------------------------------------------------------ state
    @property
    def current(self) -> Snapshot:
        return self.history[self.cursor]

    def solid(self, version: Optional[int] = None) -> cq.Workplane:
        return build(self.history[self.cursor if version is None else version].tree)

    # ------------------------------------------------------------------ edit
    def propose(self, ops: list) -> Proposal:
        tree, prev_trace = self.current.tree, self.current.trace
        old_solid = None
        msgs, warns = [], []
        rep = None
        for i, op in enumerate(ops):
            msgs.append(describe(op, tree))
            try:
                new_tree = apply_op(tree, op)
            except OpError as e:
                return Proposal(False, None, msgs, [f"Operation {i + 1} ({msgs[-1]}): {e}"], warns, failed_op=i)
            trace: list = []
            try:
                solid = build(new_tree, trace)
            except BuildError as e:
                return Proposal(False, None, msgs, [f"Operation {i + 1} ({msgs[-1]}): CAD kernel could not build {e.fid}: {e.msg}"],
                                warns, failed_op=i)
            last = i == len(ops) - 1
            rep = G.check(new_tree, solid, trace, prev_trace, roundtrip=self.roundtrip and last)
            if not rep.ok:
                return Proposal(False, None, msgs, [f"Operation {i + 1} ({msgs[-1]}): {e}" for e in rep.errors],
                                warns + rep.warnings, rep.stats, failed_op=i)
            warns += [w for w in rep.warnings if w not in warns]
            tree, prev_trace, new_solid = new_tree, trace, solid
        if old_solid is None:
            old_solid = self.solid()
        return Proposal(True, tree, msgs, [], warns, rep.stats, diff_solids(old_solid, new_solid), None, prev_trace)

    def commit(self, p: Proposal) -> Snapshot:
        if not p.ok:
            raise ValueError("cannot commit a rejected proposal")
        self.history = self.history[: self.cursor + 1]            # editing after undo discards the redo branch
        snap = Snapshot(len(self.history), p.tree, p.messages, p.stats, p.trace, p.warnings)
        self.history.append(snap)
        self.cursor = len(self.history) - 1
        return snap

    def apply(self, ops: list) -> Proposal:
        p = self.propose(ops)
        if p.ok:
            self.commit(p)
        return p

    # ------------------------------------------------------------------ history
    def undo(self) -> Snapshot:
        self.cursor = max(0, self.cursor - 1)
        return self.current

    def redo(self) -> Snapshot:
        self.cursor = min(len(self.history) - 1, self.cursor + 1)
        return self.current

    def goto(self, version: int) -> Snapshot:
        if not (0 <= version < len(self.history)):
            raise ValueError("no such version")
        self.cursor = version
        return self.current
