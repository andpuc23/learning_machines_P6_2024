#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
if __name__ == "__main__":
    rob = HardwareRobobo()
    while True:
        print(rob.read_irs())