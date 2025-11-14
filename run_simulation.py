import time

import pygame

from simulation_config import SimulationConfig
from simulation import GasLeakSimulation
from visualization import PygameRenderer


def main() -> None:
    config = SimulationConfig()
    sim = GasLeakSimulation(config)
    renderer = PygameRenderer(config)

    running = True
    print(
        "Запуск: semi-Lagrangian + Crank–Nicolson (итеративный) — "
        f"граничные условия через {SimulationConfig.boundary}."
    )

    while running:
        frame_start = time.perf_counter()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        sim.step()
        renderer.draw_frame(sim, frame_start)
        renderer.tick()

    renderer.quit()


if __name__ == "__main__":
    main()
