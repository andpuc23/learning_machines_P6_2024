#!/usr/bin/env python3
import sys

from robobo_interface import SimulationRobobo, HardwareRobobo
if __name__ == "__main__":
    if sys.argv[1] == "--hardware":
        rob = HardwareRobobo(camera=True)
    elif sys.argv[1] == "--simulation":
        rob = SimulationRobobo()
    else:
        raise ValueError(f"{sys.argv[1]} is not a valid argument.")

    threshold_val = 100 # 10 cm

    dist = rob.read_irs()[4] 
    
    while dist < threshold_val:
        rob.move(25, 25, 100)
        # time.sleep(0.01)
        dist = rob.read_irs()[4]
        print('proximity:', dist)
        
    rob.sleep(3)
    
    rob.block()
    rob.move(-25, 25, 1200) # rotate
    
    rob.block()
    rob.move(50, 50, 500) # move again
    rob.block()