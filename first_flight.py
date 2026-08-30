# Takes off
# flies 1 m back and forth
# lands

import logging
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M'

logging.basicConfig(level=logging.ERROR)

def main():
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        with MotionCommander(scf, default_height=1) as mc:

            print("Takeoff!")
            time.sleep(2)

            print("Forward...")
            mc.forward(1.0)
            time.sleep(2)

            print("Back...")
            mc.back(1.0)
            time.sleep(2)

            print("Landing!")

if __name__ == '__main__':
    main()
