# takes off
# evades objekt if they come to close
# ctrl+c = lands

import time
import math
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

URI = uri_helper.uri_from_env(default='radio://0/80/2M/E7E7E7E7E7')

FLIGHT_HEIGHT  = 1.0
LOOP_RATE      = 20

# avoidance distances
D_REACT        = 0.60
D_EMERGENCY    = 0.30
D_MIN_SIDE     = 0.30

MAX_SPEED      = 0.7
MIN_SPEED      = 0.1

TAKEOFF_SPEED  = 0.2
LAND_SPEED     = 0.2
HEIGHT_STEP    = TAKEOFF_SPEED / LOOP_RATE
LAND_STEP      = LAND_SPEED    / LOOP_RATE

# gently follows terrain changes (tables, steps etc.) during flight
HEIGHT_ADJUST_SPEED = 0.01
HEIGHT_ADJUST_STEP  = HEIGHT_ADJUST_SPEED / LOOP_RATE
HEIGHT_TOLERANCE    = 0.05

HOVER_DURATION = 2.0


def compute_speed(distance):
    if distance >= D_REACT:
        return 0.0
    if distance <= D_EMERGENCY:
        return MAX_SPEED
    factor = 1.0 - (distance / D_REACT)
    speed  = MIN_SPEED + factor * (MAX_SPEED - MIN_SPEED)
    return round(speed, 3)


class DroneController:

    def __init__(self, scf):
        self.scf = scf
        self.cf  = scf.cf

        self._raw = {
            'front':  2.0,
            'back':   2.0,
            'left':   2.0,
            'right':  2.0,
            'up':     2.0,
            'height': 0.0,
        }
        self.sensors = {
            'front':  2.0,
            'back':   2.0,
            'left':   2.0,
            'right':  2.0,
            'up':     2.0,
        }
        self.height = 0.0

        self.is_flying = False
        self.dt        = 1.0 / LOOP_RATE

        # ramped up during takeoff, adjusted during flight
        self.current_target_height = 0.0

        self.last_ground_height = None
        self.hover_start = None

    def _clamp_sensor(self, value_mm, max_m=3.0):
        if value_mm <= 0 or value_mm >= 32000:
            return max_m
        return min(value_mm / 1000.0, max_m)

    def setup_logging(self):
        lg = LogConfig(name='Sensors', period_in_ms=50)
        lg.add_variable('range.front',  'uint16_t')
        lg.add_variable('range.back',   'uint16_t')
        lg.add_variable('range.left',   'uint16_t')
        lg.add_variable('range.right',  'uint16_t')
        lg.add_variable('range.up',     'uint16_t')
        lg.add_variable('range.zrange', 'uint16_t')

        self.cf.log.add_config(lg)
        lg.data_received_cb.add_callback(self._sensor_callback)
        lg.start()
        print("[SETUP] Sensors started.")

    def _sensor_callback(self, timestamp, data, logconf):
        self._raw['front']  = self._clamp_sensor(
            data.get('range.front',  32767))
        self._raw['back']   = self._clamp_sensor(
            data.get('range.back',   32767))
        self._raw['left']   = self._clamp_sensor(
            data.get('range.left',   32767))
        self._raw['right']  = self._clamp_sensor(
            data.get('range.right',  32767))
        self._raw['up']     = self._clamp_sensor(
            data.get('range.up',     32767))
        self._raw['height'] = self._clamp_sensor(
            data.get('range.zrange', 32767), max_m=5.0)

    def update_sensors(self):
        for key in ['front', 'back', 'left', 'right', 'up']:
            self.sensors[key] = self._raw[key]
        self.height = self._raw['height']

    def compute_target_height(self):
        # flow deck measures distance to ground, compare against
        # FLIGHT_HEIGHT and nudge target up/down if it changed
        # (e.g. flying over a table)
        error = self.height - FLIGHT_HEIGHT

        if abs(error) <= HEIGHT_TOLERANCE:
            return self.current_target_height, False

        if error < 0:
            # ground got higher, we're too low now
            new_height = min(
                self.current_target_height + HEIGHT_ADJUST_STEP,
                FLIGHT_HEIGHT + 1.0
            )
        else:
            # ground dropped, we're too high now
            new_height = max(
                self.current_target_height - HEIGHT_ADJUST_STEP,
                FLIGHT_HEIGHT - 0.5
            )

        return new_height, True

    def compute_avoidance_vector(self):
        vx = 0.0
        vy = 0.0

        speed = compute_speed(self.sensors['front'])
        if speed > 0:
            vx -= speed
            print(f"[FRONT] {self.sensors['front']:.2f}m "
                  f"→ backing off {speed:.2f}m/s")

        speed = compute_speed(self.sensors['back'])
        if speed > 0:
            vx += speed
            print(f"[BACK]  {self.sensors['back']:.2f}m "
                  f"→ moving forward {speed:.2f}m/s")

        speed = compute_speed(self.sensors['left'])
        if speed > 0:
            vy -= speed
            print(f"[LEFT]  {self.sensors['left']:.2f}m "
                  f"→ moving right {speed:.2f}m/s")

        speed = compute_speed(self.sensors['right'])
        if speed > 0:
            vy += speed
            print(f"[RIGHT] {self.sensors['right']:.2f}m "
                  f"→ moving left {speed:.2f}m/s")

        speed_total = math.sqrt(vx**2 + vy**2)
        if speed_total > MAX_SPEED:
            vx = vx / speed_total * MAX_SPEED
            vy = vy / speed_total * MAX_SPEED

        return vx, vy

    def takeoff(self):
        print(f"[TAKEOFF] Climbing at {TAKEOFF_SPEED}m/s...")
        self.current_target_height = 0.0

        while self.current_target_height < FLIGHT_HEIGHT:
            t_start = time.time()
            self.update_sensors()

            self.current_target_height = min(
                self.current_target_height + HEIGHT_STEP,
                FLIGHT_HEIGHT
            )

            self.cf.commander.send_hover_setpoint(
                0.0, 0.0, 0.0,
                self.current_target_height
            )

            print(
                f"[TAKEOFF] Target={self.current_target_height:.2f}m "
                f"Actual={self.height:.2f}m"
            )

            elapsed   = time.time() - t_start
            sleeptime = self.dt - elapsed
            if sleeptime > 0:
                time.sleep(sleeptime)

        print(f"[TAKEOFF] Reached {FLIGHT_HEIGHT}m!")

        print(f"[HOVER] Stabilizing for {HOVER_DURATION}s...")
        hover_start = time.time()
        while time.time() - hover_start < HOVER_DURATION:
            t_start = time.time()
            self.update_sensors()
            self.cf.commander.send_hover_setpoint(
                0.0, 0.0, 0.0, FLIGHT_HEIGHT
            )
            remaining = HOVER_DURATION - (time.time() - hover_start)
            print(f"[HOVER] {remaining:.1f}s...")
            elapsed   = time.time() - t_start
            sleeptime = self.dt - elapsed
            if sleeptime > 0:
                time.sleep(sleeptime)

        self.current_target_height = FLIGHT_HEIGHT
        print("[HOVER] Stable! Starting avoidance.")

    def land(self):
        print(f"\n[LAND] Descending at {LAND_SPEED}m/s...")
        self.current_target_height = self.height

        while self.current_target_height > 0.10:
            t_start = time.time()
            self.update_sensors()

            self.current_target_height = max(
                self.current_target_height - LAND_STEP,
                0.0
            )

            self.cf.commander.send_hover_setpoint(
                0.0, 0.0, 0.0,
                self.current_target_height
            )

            print(
                f"[LAND] Target={self.current_target_height:.2f}m "
                f"Actual={self.height:.2f}m"
            )

            elapsed   = time.time() - t_start
            sleeptime = self.dt - elapsed
            if sleeptime > 0:
                time.sleep(sleeptime)

        print("[LAND] Landed!")
        for _ in range(10):
            self.cf.commander.send_stop_setpoint()
            time.sleep(0.05)

    def run(self):
        self.setup_logging()
        time.sleep(1.0)

        print("[RUN] Starting!")
        print(f"[RUN] Flight height:     {FLIGHT_HEIGHT}m")
        print(f"[RUN] Climb speed:       {TAKEOFF_SPEED}m/s")
        print(f"[RUN] Descent speed:     {LAND_SPEED}m/s")
        print(f"[RUN] Height adjustment: {HEIGHT_ADJUST_SPEED}m/s")
        print(f"[RUN] Height tolerance:  ±{HEIGHT_TOLERANCE*100:.0f}cm")
        print("[RUN] CTRL+C to land\n")

        self.cf.commander.send_setpoint(0, 0, 0, 0)
        time.sleep(0.1)

        try:
            self.takeoff()
            self.is_flying = True

            print("[RUN] Hovering! Try putting something under the drone.\n")

            while True:
                t_start = time.time()
                self.update_sensors()

                new_height, adjusted = self.compute_target_height()

                if adjusted:
                    direction = (
                        "↑ Climbing" if new_height > self.current_target_height
                        else "↓ Descending"
                    )
                    print(
                        f"[HEIGHT] {direction} slowly... "
                        f"Ground distance={self.height:.2f}m "
                        f"Target={FLIGHT_HEIGHT}m "
                        f"New height={new_height:.2f}m"
                    )
                    self.current_target_height = new_height

                vx, vy = self.compute_avoidance_vector()

                self.cf.commander.send_hover_setpoint(
                    vx, vy, 0.0,
                    self.current_target_height
                )

                if vx == 0.0 and vy == 0.0 and not adjusted:
                    print(
                        f"[HOVER] Idle | "
                        f"F={self.sensors['front']:.2f}m "
                        f"B={self.sensors['back']:.2f}m "
                        f"L={self.sensors['left']:.2f}m "
                        f"R={self.sensors['right']:.2f}m "
                        f"H={self.height:.2f}m "
                        f"Target={self.current_target_height:.2f}m"
                    )

                elapsed   = time.time() - t_start
                sleeptime = self.dt - elapsed
                if sleeptime > 0:
                    time.sleep(sleeptime)

        except KeyboardInterrupt:
            print("\n[RUN] CTRL+C detected!")
            self.land()

        finally:
            print("[RUN] Sending final stop.")
            for _ in range(10):
                self.cf.commander.send_stop_setpoint()
                time.sleep(0.05)


def main():
    cflib.crtp.init_drivers()
    print(f"[MAIN] Connecting to {URI}")

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        controller = DroneController(scf)
        controller.run()

    print("[MAIN] Done.")


if __name__ == '__main__':
    main()
