import numpy as np

# ---------- физические параметры ----------
# Коэффициенты диффузии D (чем больше, тем сильнее размытие), м^2/с
DIFFUSION_MOLECULAR = 1e-7  # м^2/с
DIFFUSION_TURBULENT = 1e-3  # м^2/с


def apply_boundary_shifts(C: np.ndarray, boundary: str):
    """Возвращает shifted массивы (left, right, up, down) с учетом граничных условий."""
    if boundary == "periodic":
        return (
            np.roll(C, 1, axis=1),
            np.roll(C, -1, axis=1),
            np.roll(C, 1, axis=0),
            np.roll(C, -1, axis=0),
        )
    elif boundary == "neumann":
        left = np.empty_like(C)
        left[:, 1:] = C[:, :-1]
        left[:, 0] = C[:, 0]
        right = np.empty_like(C)
        right[:, :-1] = C[:, 1:]
        right[:, -1] = C[:, -1]
        up = np.empty_like(C)
        up[1:, :] = C[:-1, :]
        up[0, :] = C[0, :]
        down = np.empty_like(C)
        down[:-1, :] = C[1:, :]
        down[-1, :] = C[-1, :]
        return left, right, up, down
    else:  # dirichlet
        left = np.empty_like(C)
        left[:, 1:] = C[:, :-1]
        left[:, 0] = 0.0
        right = np.empty_like(C)
        right[:, :-1] = C[:, 1:]
        right[:, -1] = 0.0
        up = np.empty_like(C)
        up[1:, :] = C[:-1, :]
        up[0, :] = 0.0
        down = np.empty_like(C)
        down[:-1, :] = C[1:, :]
        down[-1, :] = 0.0
        return left, right, up, down


def semi_lagrangian_advection(
    C: np.ndarray,
    u: float,
    v: float,
    dx_px: float,
    dt: float,
    boundary: str = "dirichlet",
) -> np.ndarray:
    """Выполняет semi-Lagrangian адвекцию поля C на шаг dt.

    C: поле концентрации shape (ny, nx).
    u, v — скорости в пикселях/сек (скаляры, постоянные по всему полю).
    dx_px: размер ячейки в пикселях (для перевода координат).
    boundary: тип граничных условий.
    """
    ny, nx = C.shape
    xs_grid = (np.arange(nx) + 0.5) * dx_px
    ys_grid = (np.arange(ny) + 0.5) * dx_px
    X_GRID, Y_GRID = np.meshgrid(xs_grid, ys_grid)

    # точки отправления (backtrace)
    Xd = X_GRID - u * dt
    Yd = Y_GRID - v * dt

    # нормированные координаты в индексной системе
    xi = Xd / dx_px - 0.5
    yi = Yd / dx_px - 0.5

    i0 = np.floor(xi).astype(int)
    j0 = np.floor(yi).astype(int)
    wx = xi - i0
    wy = yi - j0

    i1 = i0 + 1
    j1 = j0 + 1

    def fetch(ix, jy):
        if boundary == "periodic":
            return C[np.mod(jy, ny), np.mod(ix, nx)]

        elif boundary == "neumann":
            return C[np.clip(jy, 0, ny - 1), np.clip(ix, 0, nx - 1)]
        else:  # dirichlet
            mask = (ix >= 0) & (ix < nx) & (jy >= 0) & (jy < ny)
            res = np.zeros_like(ix, dtype=float)
            valid_ix = np.clip(ix, 0, nx - 1)
            valid_jy = np.clip(jy, 0, ny - 1)
            res[mask] = C[valid_jy[mask], valid_ix[mask]]
            return res

    # Векторизованная выборка и билинейная интерполяция
    val00 = fetch(i0, j0)
    val10 = fetch(i1, j0)
    val01 = fetch(i0, j1)
    val11 = fetch(i1, j1)

    wx_inv = 1.0 - wx
    wy_inv = 1.0 - wy
    return (
        wx_inv * wy_inv * val00
        + wx * wy_inv * val10
        + wx_inv * wy * val01
        + wx * wy * val11
    )


def laplacian(
    C: np.ndarray,
    dx_m: float,
    dy_m: float,
    boundary: str = "dirichlet",
) -> np.ndarray:
    """Вычисляет лапласиан поля C с учетом выбранных граничных условий."""
    left, right, up, down = apply_boundary_shifts(C, boundary)
    dx2_inv = 1.0 / (dx_m * dx_m)
    dy2_inv = 1.0 / (dy_m * dy_m)
    return (left - 2.0 * C + right) * dx2_inv + (up - 2.0 * C + down) * dy2_inv


def crank_nicolson_diffusion(
    C_in: np.ndarray,
    D: float,
    dx_m: float,
    dy_m: float,
    dt: float,
    boundary: str = "dirichlet",
    jacobi_iters: int = 40,
) -> np.ndarray:
    """Выполняет шаг диффузии по схеме Crank–Nicolson с итерациями Якоби."""
    alpha = D * dt * 0.5
    lap_C = laplacian(C_in, dx_m, dy_m, boundary)
    RHS = C_in + alpha * lap_C

    Cnew = C_in.copy()

    ax = alpha / (dx_m * dx_m)
    ay = alpha / (dy_m * dy_m)
    denom = 1.0 + 2.0 * (ax + ay)
    is_dirichlet = boundary == "dirichlet"

    for _ in range(jacobi_iters):
        left, right, up, down = apply_boundary_shifts(Cnew, boundary)
        Cnew = (RHS + ax * (left + right) + ay * (up + down)) / denom

        if is_dirichlet:
            Cnew[0, :] = 0.0
            Cnew[-1, :] = 0.0
            Cnew[:, 0] = 0.0
            Cnew[:, -1] = 0.0

    return Cnew


def step_advdiff(
    C: np.ndarray,
    u_px: float,
    v_px: float,
    dx_px: float,
    dx_m: float,
    dy_m: float,
    dt: float,
    D: float,
    decay_multiplier: float,
    source_cell,
    source_rate: float,
    boundary: str,
    jacobi_iters: int,
) -> np.ndarray:
    """Один шаг: semi-Lagrangian адвекция, затем Crank–Nicolson диффузия,
    затем источник и распад."""
    # 1) адвекция
    C_adv = semi_lagrangian_advection(C, u_px, v_px, dx_px, dt, boundary)

    # 2) диффузия
    C_diff = crank_nicolson_diffusion(C_adv, D, dx_m, dy_m, dt, boundary, jacobi_iters)

    # 3) источник (распределяем по окрестности 3x3 вокруг source_cell)
    if source_cell is not None:
        iy0, ix0 = source_cell
        cell_area_m2 = dx_m * dy_m
        total_add = (source_rate * dt) / cell_area_m2
        w = total_add / 9.0
        ny, nx = C.shape
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                iy = iy0 + dy
                ix = ix0 + dx
                if 0 <= iy < ny and 0 <= ix < nx:
                    C_diff[iy, ix] += w

    # 4) распад
    C_diff *= decay_multiplier

    return C_diff

