import numpy as np

from pairbench.io.dxf_template import _chain_loops, render_template


def test_chain_loops_square_from_segments():
    segs = [
        np.array([[0.0, 0.0], [10.0, 0.0]]),
        np.array([[10.0, 10.0], [0.0, 10.0]]),
        np.array([[10.0, 0.0], [10.0, 10.0]]),
        np.array([[0.0, 0.0], [0.0, 10.0]]),  # reversed orientation
    ]
    loops = _chain_loops(segs)
    assert len(loops) == 1
    assert len(loops[0]) >= 4


def test_l_profile_two_loops_and_scale():
    t = render_template("l-profile")
    assert len(t.loops_mm) == 2  # outer boundary + one hole
    # section extent ~69 x 23.7 mm at 2.5 px/mm
    h, w = t.image.shape
    assert 160 <= w <= 185 and 55 <= h <= 65
    assert t.max_val_ref > 0
    fill = (t.image > 127).mean()
    assert 0.10 < fill < 0.30  # thin section, not solid, not empty


def test_square_profile_is_hollow():
    t = render_template("square-profile")
    h, w = t.image.shape
    assert abs(h - 100) <= 2 and abs(w - 100) <= 2  # 40 mm at 2.5 px/mm
    assert t.image[h // 2, w // 2] == 0          # hole (even-odd fill)
    assert t.image[h // 2, 2] > 0                # 2 mm wall
    fill = (t.image > 127).mean()
    assert 0.10 < fill < 0.30  # ring area fraction = (40^2-36^2)/40^2 = 0.19
