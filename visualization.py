import os
import time

import numpy as np
import pygame
from PIL import Image
from matplotlib import colormaps

from simulation_config import SimulationConfig


WHITE_COLOR = (255, 255, 255)
RED_COLOR = (255, 0, 0)
HUD_X = 8
CMAP = colormaps["viridis"]


class PygameRenderer:
    """Отвечает за инициализацию pygame и отрисовку кадра."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config

        # убедимся, что фон существует
        if not os.path.exists(config.background_path):
            Image.new("RGBA", (config.width_px, config.height_px), (125, 125, 125, 255)).save(
                config.background_path
            )

        pygame.init()
        self.screen = pygame.display.set_mode((config.width_px, config.height_px))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("Arial", 16)

        bg_img = Image.open(config.background_path).convert("RGBA").resize(
            (config.width_px, config.height_px)
        )
        self.bg_surf = pygame.image.fromstring(bg_img.tobytes(), bg_img.size, "RGBA")

        # маркер источника
        self.source_marker_x = int(config.source_pos_px[0])
        self.source_marker_y = int(config.source_pos_px[1])

    def _field_to_surface(self, Cfield: np.ndarray) -> pygame.Surface:
        """Конвертирует поле концентрации в поверхностный слой с colormap и альфой."""
        cmax = float(Cfield.max())
        if cmax <= 0.0:
            vmax = 1.0
        else:
            vmax = float(np.percentile(Cfield, 99.5))
        vmax = max(vmax, 1e-9)

        H_norm = np.clip(Cfield / vmax, 0.0, 1.0)
        rgba = (CMAP(H_norm) * 255).astype(np.uint8)
        pil_img = Image.fromarray(np.flipud(rgba))
        surf = pygame.image.fromstring(pil_img.tobytes(), pil_img.size, "RGBA").convert_alpha()
        surf = pygame.transform.smoothscale(surf, (self.config.width_px, self.config.height_px))
        surf.set_alpha(200)
        return surf

    def draw_frame(self, sim, frame_start_time: float) -> None:
        """Рисует один кадр, включая фон, поле, источник и HUD."""
        C = sim.field
        heat_surf = self._field_to_surface(C)

        self.screen.blit(self.bg_surf, (0, 0))
        self.screen.blit(heat_surf, (0, 0))

        # маркер источника
        pygame.draw.line(
            self.screen,
            RED_COLOR,
            (self.source_marker_x - 8, self.source_marker_y),
            (self.source_marker_x + 8, self.source_marker_y),
            2,
        )
        pygame.draw.line(
            self.screen,
            RED_COLOR,
            (self.source_marker_x, self.source_marker_y - 8),
            (self.source_marker_x, self.source_marker_y + 8),
            2,
        )

        # HUD
        frame_ms = (time.perf_counter() - frame_start_time) * 1000.0
        fps = self.clock.get_fps()
        cfg = self.config
        hud_lines = [
            f"Wind (m/s): {cfg.wind_velocity_mps[0]:.2f}, {cfg.wind_velocity_mps[1]:.2f}",
            f"D_total (m^2/s): {cfg.diffusion_total:.2e}",
            f"dx, dy (m): {cfg.dx_m:.4f}, {cfg.dy_m:.4f}",
            f"Source (kg/s): {cfg.source_rate_kg_per_s:.3e}",
            f"Decay (1/s): {cfg.decay_rate:.3e}",
            f"Frame ms: {frame_ms:.1f}, FPS(est): {fps:.1f}",
        ]
        for i, line in enumerate(hud_lines):
            surf = self.font.render(line, True, WHITE_COLOR)
            self.screen.blit(surf, (HUD_X, 8 + i * 18))

        pygame.display.flip()

    def tick(self) -> None:
        """Ограничивает FPS согласно конфигу."""
        self.clock.tick(self.config.fps)

    def quit(self) -> None:
        pygame.quit()
