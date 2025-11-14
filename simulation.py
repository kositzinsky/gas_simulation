import numpy as np

from simulation_config import SimulationConfig
from euler_defussion import step_advdiff


class GasLeakSimulation:
    """Инкапсулирует состояние и шаги модели утечки газа.

    Содержит поле концентрации C[y, x] и параметры среды из SimulationConfig.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

        # поле концентрации
        self.C = np.zeros((config.ny, config.nx), dtype=float)

        # предвычисления
        self.decay_multiplier: float = float(np.exp(-config.decay_rate * config.dt))

        # удобные сокращения
        self._dx_px: int = config.grid_cell_px
        self._dx_m: float = config.dx_m
        self._dy_m: float = config.dy_m

        self._u_px: float = float(config.wind_velocity_pps[0])
        self._v_px: float = float(config.wind_velocity_pps[1])

    def step(self) -> None:
        """Выполняет один шаг модели (адвекция + диффузия + источник + распад)."""
        self.C = step_advdiff(
            self.C,
            self._u_px,
            self._v_px,
            self._dx_px,
            self._dx_m,
            self._dy_m,
            self.config.dt,
            self.config.diffusion_total,
            self.decay_multiplier,
            self.config.source_cell,
            self.config.source_rate_kg_per_s,
            self.config.boundary,
            self.config.diffuse_jacobi_iters,
        )

    @property
    def field(self) -> np.ndarray:
        """Текущее поле концентрации C[y, x]."""
        return self.C

    def reset(self) -> None:
        """Сбрасывает концентрацию к нулю."""
        self.C.fill(0.0)

    def set_wind_mps(self, vx: float, vy: float) -> None:
        """Обновляет скорость ветра (в м/с) и внутренние пиксельные значения."""
        self.config.wind_velocity_mps = np.array([vx, vy], dtype=float)
        self.config.wind_velocity_pps = self.config.wind_velocity_mps * self.config.pixel_per_meter
        self._u_px = float(self.config.wind_velocity_pps[0])
        self._v_px = float(self.config.wind_velocity_pps[1])
