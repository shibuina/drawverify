import numpy as np

from libero_sketch import geometry as G

H, W = 256, 256
K = np.array([[200.0, 0, 128], [0, 200.0, 128], [0, 0, 1]])
# camera 1.5m above origin looking straight down (-z world), x aligned
T_WC = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 1.5],
    [0, 0, 0, 1.0],
])


def test_project_center():
    uv = G.project(np.array([[0.0, 0.0, 0.0]]), K, T_WC, H, W)
    assert np.allclose(uv[0], [128, 128], atol=1e-6)


def test_project_offsets_move_pixels():
    uv_x = G.project(np.array([[0.1, 0.0, 0.0]]), K, T_WC, H, W)[0]
    uv_y = G.project(np.array([[0.0, 0.1, 0.0]]), K, T_WC, H, W)[0]
    assert uv_x[0] != 128 and abs(uv_x[1] - 128) < 1e-6
    assert uv_y[1] != 128 and abs(uv_y[0] - 128) < 1e-6


def test_calibration_recovers_projection():
    pt_w = np.array([[0.1, 0.05, 0.0]])
    true_var = {"flip_u": False, "flip_v": True, "transpose": False}
    uv = G.project(pt_w, K, T_WC, H, W, true_var)[0]
    mask = np.zeros((H, W), dtype=bool)
    u, v = int(round(uv[0])), int(round(uv[1]))
    mask[max(v - 4, 0):v + 5, max(u - 4, 0):u + 5] = True
    got, score = G.calibrate_pixel_variant(pt_w, mask, K, T_WC)
    assert got is not None and score < 1e4  # some variant lands inside the mask
    uv2 = G.project(pt_w, K, T_WC, H, W, got)[0]
    assert bool(mask[int(round(uv2[1])), int(round(uv2[0]))])


def test_sphere_radius_scales_with_depth():
    near = G.sphere_radius_px(np.array([[0, 0, 1.0]]), 0.05, K, T_WC)
    far = G.sphere_radius_px(np.array([[0, 0, 0.0]]), 0.05, K, T_WC)
    assert near > far > 0


def test_rdp_straight_line_two_points():
    pts = np.stack([np.linspace(0, 10, 50), np.linspace(0, 5, 50)], axis=1)
    out = G.rdp(pts, eps=0.1)
    assert len(out) == 2


def test_rdp_keeps_corner():
    a = np.stack([np.linspace(0, 1, 20), np.zeros(20)], axis=1)
    b = np.stack([np.ones(20), np.linspace(0, 1, 20)], axis=1)
    out = G.rdp(np.concatenate([a, b]), eps=0.05)
    assert len(out) == 3


def test_normalize_roundtrip():
    uv = np.array([[64.0, 200.0]])
    n = G.normalize_uv(uv, H, W)
    back = G.denormalize_uv(n, H, W)
    assert np.allclose(back, uv)
