import time

from pairbench.timing import mean_ms, merge, report_lines, reset, snapshot, stage


def test_stage_accumulates_calls_and_time():
    reset()
    with stage("a"):
        time.sleep(0.01)
    with stage("a"):
        time.sleep(0.01)
    snap = snapshot()
    assert snap["a"]["calls"] == 2
    assert snap["a"]["total_s"] >= 0.02


def test_reset_clears_state():
    reset()
    with stage("a"):
        pass
    reset()
    assert snapshot() == {}


def test_stage_isolates_names():
    reset()
    with stage("a"):
        pass
    with stage("b"):
        pass
    with stage("b"):
        pass
    snap = snapshot()
    assert snap["a"]["calls"] == 1
    assert snap["b"]["calls"] == 2


def test_stage_times_even_on_exception():
    reset()
    try:
        with stage("a"):
            raise ValueError("boom")
    except ValueError:
        pass
    assert snapshot()["a"]["calls"] == 1


def test_merge_sums_across_snapshots():
    snap1 = {"a": {"total_s": 1.0, "calls": 2}}
    snap2 = {"a": {"total_s": 0.5, "calls": 1}, "b": {"total_s": 2.0, "calls": 4}}
    merged = merge(snap1, snap2)
    assert merged["a"] == {"total_s": 1.5, "calls": 3}
    assert merged["b"] == {"total_s": 2.0, "calls": 4}


def test_mean_ms():
    assert mean_ms({"total_s": 1.0, "calls": 4}) == 250.0
    assert mean_ms({"total_s": 0.0, "calls": 0}) == 0.0


def test_report_lines_empty_snapshot():
    assert report_lines({}) == []


def test_report_lines_nonempty_has_table():
    lines = report_lines({"a": {"total_s": 1.0, "calls": 2}}, title="X")
    assert lines[0] == "## X"
    assert any("| a | 2 | 1.000 | 500.000 |" in line for line in lines)
