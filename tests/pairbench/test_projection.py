import numpy as np

from pairbench.io.projection import depth_channels, project_yz


def _grid_points(ny=40, nz=30, step=1.0):
    ys, zs = np.meshgrid(np.arange(ny) * step, np.arange(nz) * step)
    pts = np.zeros((ys.size, 3))
    pts[:, 1] = ys.ravel()
    pts[:, 2] = zs.ravel()
    return pts


def test_output_geometry():
    # 39x29 mm dense grid at 2.5 px/mm eff -> unpadded ~ (39*10+1)/4 x (29*10+1)/4
    proj = project_yz(_grid_points(), nsigma=None, padding_px=10)
    h, w = proj.image.shape
    assert abs((w - 20) - 39 * 2.5) < 4
    assert abs((h - 20) - 29 * 2.5) < 4
    # 1 mm grid leaves sub-pixel holes between dilated disks; blur caps peak below 255
    assert proj.image.max() >= 200
    assert proj.image[:5, :5].max() == 0  # padding is black


def test_pixel_to_yz_roundtrip():
    pts = _grid_points()
    pts[:, 1] += 100.0  # offset so min_y != 0
    pts[:, 2] += -50.0
    proj = project_yz(pts, nsigma=None)
    # unpadded top-left pixel maps near (min_y, max_z)
    y, z = proj.pixel_to_yz(proj.padding_px, proj.padding_px)
    assert abs(y - 100.0) < 1.0
    assert abs(z - (29.0 - 50.0)) < 1.0


def test_z_flip():
    # single high point must land ABOVE (smaller row) a low point
    pts = np.array([[0, 0, 0], [0, 0, 0.1], [0, 10, 20], [0, 10, 20.1], [0, 5, 0.05]])
    proj = project_yz(pts, nsigma=None, padding_px=0, blur_kernel=0)
    img = proj.image
    ys, xs = np.nonzero(img)
    # highest-Z point (z=20.1, y=10) should be at min row and max col region
    assert ys.min() < ys.max()
    top_cols = xs[ys < ys.mean()]
    bot_cols = xs[ys > ys.mean()]
    assert top_cols.mean() > bot_cols.mean()


def _spread_plus(extra: np.ndarray) -> np.ndarray:
    """A small square of corner points (gives project_yz a normal-sized
    canvas) plus extra points to test, all appended after."""
    corners = np.array([
        [0.0, 0.0, 0.0], [0.0, 20.0, 0.0],
        [0.0, 0.0, 20.0], [0.0, 20.0, 20.0],
    ])
    return np.vstack([corners, extra])


def test_depth_channels_near_far_side1():
    # two points at the same (y,z) pixel, different X: side 1 -> nearer
    # scanner is the LARGER X, per the module's documented convention.
    pts = _spread_plus(np.array([
        [100.0, 10.0, 10.0],   # far
        [110.0, 10.0, 10.0],   # near (larger X)
        [50.0, 10.0, 10.0],    # farther still
    ]))
    proj = project_yz(pts, nsigma=None, padding_px=5)
    ch = depth_channels(pts, proj, side=1)
    assert np.nanmax(ch.near_x_mm) == 110.0
    # the pixel all three test points share must read density=3, near_x=110
    py, px = np.unravel_index(np.nanargmax(ch.near_x_mm), ch.near_x_mm.shape)
    assert ch.density[py, px] == 3.0
    assert ch.near_x_mm[py, px] == 110.0
    assert ch.x_spread_mm[py, px] == 110.0 - 50.0


def test_depth_channels_near_far_side2():
    pts = _spread_plus(np.array([
        [100.0, 10.0, 10.0],
        [110.0, 10.0, 10.0],
        [50.0, 10.0, 10.0],    # near for side 2 (smaller X)
    ]))
    proj = project_yz(pts, nsigma=None, padding_px=5)
    ch = depth_channels(pts, proj, side=2)
    py, px = np.unravel_index(np.nanargmax(ch.density), ch.density.shape)
    assert ch.near_x_mm[py, px] == 50.0


def test_depth_channels_observed_mask_matches_data_extent():
    pts = _spread_plus(np.array([[0.0, 10.0, 10.0]]))
    proj = project_yz(pts, nsigma=None, padding_px=5, blur_kernel=0)
    ch = depth_channels(pts, proj, side=1)
    # unobserved pixels must be genuinely empty, not conflated with the
    # dilated/blurred silhouette that proj.image alone provides
    assert ch.observed.sum() <= len(pts)
    assert ch.density[~ch.observed].sum() == 0
    assert np.all(np.isnan(ch.near_x_mm[~ch.observed]))


def test_depth_channels_respects_keep_mask():
    # an nsigma-dropped outlier point must not leak into density/near_x
    pts = _spread_plus(np.array([[0.0, 10.0, 10.0]] * 20 + [[999.0, 500.0, 500.0]]))
    proj = project_yz(pts, nsigma=3.0, padding_px=5)
    assert proj.keep_mask[-1] == False  # the outlier must actually be dropped
    ch = depth_channels(pts, proj, side=1)
    assert ch.density.sum() == proj.n_points
