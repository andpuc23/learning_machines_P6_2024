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

    threshold_val = 100 # 10 cm
      

    dist = rob.read_irs()[4] 
    
    # while dist < threshold_val:
    for i in range(1000):
        rob.move(25, 25, 150)
        # print('sleeping')
        # rob.sleep(5)
        dist = rob.read_irs()[4]
        print('proximity:', dist)
        if dist >= threshold_val:
            break
    if isinstance(rob, HardwareRobobo):
        rob.sleep(3)
    print('done')
    # rob.block()
    
    rob.move(-25, 25, 1200) # rotate
    
    # rob.block()

    rob.move(50, 50, 500) # move again
    # rob.block()

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
