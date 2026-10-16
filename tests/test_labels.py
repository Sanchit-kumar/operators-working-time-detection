"""Label editing logic of the labelling tool: painting ranges, going back to fix, undo."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from label_tool import Labels  # noqa: E402


def test_from_now_and_whole(tmp_path):
    L = Labels(tmp_path / "l.json")
    L.paint(1, 0.0, None, "working")            # everything from 0 s
    assert L.state(1, 50) == "working"
    L.paint(1, 10.0, None, "idle")              # from 10 s on
    assert (L.state(1, 5), L.state(1, 15)) == ("working", "idle")


def test_go_back_and_fix_a_range(tmp_path):
    L = Labels(tmp_path / "l.json")
    L.paint(1, 0.0, 30.0, "working")
    L.paint(1, 10.0, 15.0, "idle")              # go back and mark a pause in the middle
    assert [L.state(1, t) for t in (5, 12, 20)] == ["working", "idle", "working"]
    L.paint(1, 12.0, 20.0, "working")           # correct it again, overlapping
    assert [L.state(1, t) for t in (11, 13, 25)] == ["idle", "working", "working"]
    assert L.state(1, 31) is None               # after the painted range stays unlabelled


def test_from_now_stops_at_next_change(tmp_path):
    L = Labels(tmp_path / "l.json")
    L.paint(1, 0.0, 30.0, "working")
    L.paint(1, 20.0, 30.0, "idle")
    L.paint(1, 5.0, None, "ignore")             # from 5 s until the next existing change (20 s)
    assert [L.state(1, t) for t in (2, 10, 25)] == ["working", "ignore", "idle"]


def test_undo_and_persistence(tmp_path):
    path = tmp_path / "l.json"
    L = Labels(path)
    L.paint(1, 0.0, 10.0, "working")
    L.paint(1, 0.0, 10.0, "idle")
    assert L.undo() and L.state(1, 5) == "working"
    assert Labels(path).state(1, 5) == "working"  # saved immediately
    assert L.undo() and L.state(1, 5) is None
    assert not L.undo()


def test_accept_suggestion_is_one_undo_step(tmp_path):
    L = Labels(tmp_path / "l.json")
    L.paint_many(1, [(0.0, 5.0, "working"), (5.0, 9.0, "idle"), (9.0, 12.0, "working")])
    assert [L.state(1, t) for t in (2, 6, 10)] == ["working", "idle", "working"]
    assert L.undo() and L.state(1, 2) is None and L.state(1, 10) is None
