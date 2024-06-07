#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
if __name__ == "__main__":
    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
        rob.play_simulation()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")
    i = 0
    while i < 10_000:
        print('\t'.join([str(x) for x in rob.read_irs()]))

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()