from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class SimulationConfig:
    """Параметры симуляции утечки газа в эйлеровом приближении."""

    # геометрия окна / сетки (в пикселях)
    width_px: int = 1000
    height_px: int = 1000
    grid_cell_px: int = 4

    # физический масштаб
    meter_per_pixel: float = 0.01  # метров в одном пикселе

    # шаг по времени, с
    dt: float = 0.1

    # коэффициенты диффузии, м^2/с
    diffusion_molecular: float = 1e-7
    diffusion_turbulent: float = 1e-3

    # коэффициент экспоненциального распада, 1/с
    decay_rate: float = 1e-3

    # скорость ветра, м/с (vx, vy)
    wind_velocity_mps: np.ndarray = np.array([-0.2, -0.2], dtype=float)

    # граничные условия: 'dirichlet', 'neumann', 'periodic'
    boundary: str = "neumann"

    # итерации Якоби для схемы Crank–Nicolсон
    diffuse_jacobi_iters: int = 40

    # источник
    source_rate_kg_per_s: float = 1e-9
    source_pos_px: Optional[np.ndarray] = None

    # визуализация
    background_path: str = "background.png"
    fps: int = 60

    def __post_init__(self) -> None:
        # размеры сетки
        self.nx: int = self.width_px // self.grid_cell_px
        self.ny: int = self.height_px // self.grid_cell_px

        # масштаб
        self.pixel_per_meter: float = 1.0 / self.meter_per_pixel

        # шаг сетки в метрах
        self.dx_m: float = self.grid_cell_px * self.meter_per_pixel
        self.dy_m: float = self.grid_cell_px * self.meter_per_pixel

        # полный коэффициент диффузии
        self.diffusion_total: float = self.diffusion_molecular + self.diffusion_turbulent

        # скорость в пикселях/сек
        self.wind_velocity_pps: np.ndarray = self.wind_velocity_mps * self.pixel_per_meter

        # позиция источника по умолчанию – центр по высоте, фиксированный x
        if self.source_pos_px is None:
            self.source_pos_px = np.array([self.width_px / 2.0, self.height_px / 2.0])

        # индекс ячейки источника (iy, ix)
        self.source_cell: Optional[Tuple[int, int]] = (
            int(self.source_pos_px[1] // self.grid_cell_px),
            int(self.source_pos_px[0] // self.grid_cell_px),
        )

