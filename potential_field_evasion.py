import logging
import signal
import time
from collections import deque
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

URI = 'radio://0/80/2M/E7E7E7E7E7'

FLIGHT_HEIGHT        = 0.4
AVOID_DISTANCE       = 0.55
CRITICAL_DISTANCE    = 0.38
FORWARD_SPEED        = 0.2
SIDE_AVOID_SPEED     = 0.2
MAX_SPEED            = 0.35
MAX_Z_SPEED          = 0.1
K_REPULSION          = 0.3
K_REPULSION_CRIT     = 0.75
MAX_FLIGHT_TIME      = 120
FILTER_SIZE          = 5
LOOP_RATE            = 20
LOOP_DT              = 1.0 / LOOP_RATE

K_HEIGHT             = 0.5
HEIGHT_TOLERANCE     = 0.05
CEILING_DISTANCE     = 0.3

# height gets averaged over a few seconds so it doesn't react
# to every single bump (flying over a table etc.)
LOG_PERIOD_MS        = 50
HEIGHT_AVERAGE_TIME  = 3.0
HEIGHT_AVERAGE_SAMPLES = int(
    (HEIGHT_AVERAGE_TIME * 1000) / LOG_PERIOD_MS
)

logging.basicConfig(level=logging.ERROR)

STATE_TAKEOFF = "TAKEOFF"
STATE_FLIGHT  = "FLIGHT"
STATE_LANDING = "LANDING"

AVOID_NONE  = "NONE"
AVOID_LEFT  = "LEFT"
AVOID_RIGHT = "RIGHT"

sensor_data = {
    'front': float('inf'),
    'back':  float('inf'),
    'left':  float('inf'),
    'right': float('inf'),
    'up':    float('inf'),
    'down':  float('inf'),
}

sensor_history = {
    'front': deque(maxlen=FILTER_SIZE),
    'back':  deque(maxlen=FILTER_SIZE),
    'left':  deque(maxlen=FILTER_SIZE),
    'right': deque(maxlen=FILTER_SIZE),
    'up':    deque(maxlen=FILTER_SIZE),
    'down':  deque(maxlen=FILTER_SIZE),
}

# separate buffer for height, uses a much longer window (~3s)
# than the regular sensor_history (~0.25s)
height_history = deque(maxlen=HEIGHT_AVERAGE_SAMPLES)

current_state       = STATE_TAKEOFF
shutdown_requested   = False
avoid_state          = AVOID_NONE

SENSOR_MAX_RANGE = 4.0


def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[INFO] STRG+C detected → initiating landing...")
    shutdown_requested = True

signal.signal(signal.SIGINT, signal_handler)


def sensor_callback(timestamp, data, logconf):
    global sensor_data

    def mm_to_m(val):
        if val >= 4000:
            return float('inf')
        return val / 1000.0

    sensor_data['front'] = mm_to_m(data.get('range.front',  4000))
    sensor_data['back']  = mm_to_m(data.get('range.back',   4000))
    sensor_data['left']  = mm_to_m(data.get('range.left',   4000))
    sensor_data['right'] = mm_to_m(data.get('range.right',  4000))
    sensor_data['up']    = mm_to_m(data.get('range.up',     4000))
    sensor_data['down']  = mm_to_m(data.get('range.zrange', 4000))

    for key in sensor_history:
        val = sensor_data[key]
        if val == float('inf'):
            sensor_history[key].append(SENSOR_MAX_RANGE)
        else:
            sensor_history[key].append(val)

    down_val = sensor_data['down']
    if down_val == float('inf'):
        height_history.append(SENSOR_MAX_RANGE)
    else:
        height_history.append(down_val)


def get_filtered(key):
    history = sensor_history[key]
    if len(history) == 0:
        return sensor_data[key]
    return sum(history) / len(history)


def get_average_height():
    if len(height_history) == 0:
        return sensor_data['down']
    return sum(height_history) / len(height_history)


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))


def compute_repulsion(distance, avoid_dist, k):
    if distance < avoid_dist and distance > 0:
        return k * (1.0 / distance - 1.0 / avoid_dist)
    return 0.0


def compute_forward_speed():
    front = get_filtered('front')

    if front >= AVOID_DISTANCE:
        return FORWARD_SPEED

    factor = (front - CRITICAL_DISTANCE) / (AVOID_DISTANCE - CRITICAL_DISTANCE)
    factor = clamp(factor, 0.0, 1.0)

    return FORWARD_SPEED * factor


def choose_avoid_direction():
    left  = get_filtered('left')
    right = get_filtered('right')

    if left >= right:
        print(f"[AVOID] Dodging LEFT  (L={left:.2f}m R={right:.2f}m)")
        return AVOID_LEFT
    else:
        print(f"[AVOID] Dodging RIGHT (L={left:.2f}m R={right:.2f}m)")
        return AVOID_RIGHT


def compute_height_correction():
    current_height = get_average_height()
    error = FLIGHT_HEIGHT - current_height

    if abs(error) < HEIGHT_TOLERANCE:
        vz_height = 0.0
    else:
        vz_height = clamp(error * K_HEIGHT, -MAX_Z_SPEED, MAX_Z_SPEED)

    # ceiling stays reactive, no averaging - needs to respond fast
    up = get_filtered('up')
    vz_ceiling = -compute_repulsion(up, CEILING_DISTANCE, K_REPULSION_CRIT)

    vz = vz_height + vz_ceiling
    return clamp(vz, -MAX_Z_SPEED, MAX_Z_SPEED)


def compute_velocity():
    global avoid_state

    front = get_filtered('front')
    back  = get_filtered('back')
    left  = get_filtered('left')
    right = get_filtered('right')

    vx = compute_forward_speed()
    vx -= compute_repulsion(front, AVOID_DISTANCE,    K_REPULSION)
    vx += compute_repulsion(back,  AVOID_DISTANCE,    K_REPULSION)
    vx -= compute_repulsion(front, CRITICAL_DISTANCE, K_REPULSION_CRIT)
    vx += compute_repulsion(back,  CRITICAL_DISTANCE, K_REPULSION_CRIT)

    vy = 0.0
    vy -= compute_repulsion(left,  AVOID_DISTANCE,    K_REPULSION)
    vy += compute_repulsion(right, AVOID_DISTANCE,    K_REPULSION)
    vy -= compute_repulsion(left,  CRITICAL_DISTANCE, K_REPULSION_CRIT)
    vy += compute_repulsion(right, CRITICAL_DISTANCE, K_REPULSION_CRIT)

    if front < AVOID_DISTANCE:
        if avoid_state == AVOID_NONE:
            avoid_state = choose_avoid_direction()
    else:
        avoid_state = AVOID_NONE

    if avoid_state == AVOID_LEFT:
        vy += SIDE_AVOID_SPEED
    elif avoid_state == AVOID_RIGHT:
        vy -= SIDE_AVOID_SPEED

    vz = compute_height_correction()

    vx = clamp(vx, -MAX_SPEED, MAX_SPEED)
    vy = clamp(vy, -MAX_SPEED, MAX_SPEED)

    return vx, vy, vz


def print_status(state, vx, vy, vz):
    print(
        f"[{state:10}] "
        f"F:{get_filtered('front'):5.2f}m "
        f"B:{get_filtered('back'):5.2f}m "
        f"L:{get_filtered('left'):5.2f}m "
        f"R:{get_filtered('right'):5.2f}m "
        f"U:{get_filtered('up'):5.2f}m "
        f"D:{get_filtered('down'):5.2f}m "
        f"D_avg:{get_average_height():5.2f}m "
        f"| vx:{vx:+.2f} vy:{vy:+.2f} vz:{vz:+.2f} "
        f"| Avoiding:{avoid_state}"
    )


def main():
    global current_state, shutdown_requested, avoid_state

    cflib.crtp.init_drivers()

    print("[INFO] Connecting to Crazyflie...")
    print("[INFO] Press STRG+C to land\n")

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        log_config = LogConfig(name='Ranger', period_in_ms=LOG_PERIOD_MS)
        log_config.add_variable('range.front',  'uint16_t')
        log_config.add_variable('range.back',   'uint16_t')
        log_config.add_variable('range.left',   'uint16_t')
        log_config.add_variable('range.right',  'uint16_t')
        log_config.add_variable('range.up',     'uint16_t')
        log_config.add_variable('range.zrange', 'uint16_t')

        scf.cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(sensor_callback)
        log_config.start()

        time.sleep(0.5)

        with MotionCommander(scf, default_height=FLIGHT_HEIGHT) as mc:

            current_state = STATE_TAKEOFF
            print(f"[{STATE_TAKEOFF}]")

            time.sleep(2.0)

            # prefill height buffer with target height so the
            # controller doesn't start from 0
            height_history.extend([FLIGHT_HEIGHT] * HEIGHT_AVERAGE_SAMPLES)

            avoid_state = AVOID_NONE

            current_state = STATE_FLIGHT
            print(f"[{STATE_FLIGHT}]\n")

            start_time = time.time()

            while current_state == STATE_FLIGHT:

                if time.time() - start_time > MAX_FLIGHT_TIME:
                    current_state = STATE_LANDING
                    break

                if shutdown_requested:
                    current_state = STATE_LANDING
                    break

                vx, vy, vz = compute_velocity()

                print_status(STATE_FLIGHT, vx, vy, vz)

                mc.start_linear_motion(vx, vy, vz)

                time.sleep(LOOP_DT)

            print(f"\n[{STATE_LANDING}]")

            mc.stop()
            time.sleep(0.3)
            mc.land()

            print(f"[{STATE_LANDING}] done")

        log_config.stop()

    print("[INFO] Connection closed.")


if __name__ == '__main__':
    main()