import logging
import math
import signal
import time
from collections import deque

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.syncLogger import SyncLogger
from cflib.crazyflie.log import LogConfig
from cflib.positioning.motion_commander import MotionCommander

# --- connection ---
URI = 'radio://0/80/2M/E7E7E7E7E7'

FLIGHT_HEIGHT        = 0.4
AVOID_DISTANCE       = 0.45
CRITICAL_DISTANCE    = 0.38
FORWARD_SPEED        = 0.15
SIDE_AVOID_SPEED     = 0.1
MAX_SPEED            = 0.2
MAX_Z_SPEED          = 0.1
K_REPULSION          = 0.25
K_REPULSION_CRIT     = 0.75
MAX_FLIGHT_TIME      = 120
FILTER_SIZE          = 3
LOOP_RATE            = 20
LOOP_DT              = 1.0 / LOOP_RATE

K_HEIGHT             = 1.2
HEIGHT_TOLERANCE     = 0.05
CEILING_DISTANCE     = 0.3

# height gets averaged over a few seconds so it doesn't jump around
# when flying over furniture etc.
LOG_PERIOD_MS           = 50
HEIGHT_AVERAGE_TIME     = 3.0
HEIGHT_AVERAGE_SAMPLES  = int((HEIGHT_AVERAGE_TIME * 1000) / LOG_PERIOD_MS)

# Waypoints are relative to takeoff position, in meters.
# x = forward, y = left, z = height (optional, defaults to FLIGHT_HEIGHT)
# Leave empty for explore mode (fly forward, avoid stuff).
WAYPOINTS = [
    (0.0, 1.0),
    (2.5, 1.0),
    (2.5, 0.5),
    (3.5, 0.5),
]

RETURN_TO_HOME       = False
WAYPOINT_TOLERANCE   = 0.05
WAYPOINT_TIMEOUT     = 25.0
GOAL_SPEED_MAX       = 0.2
K_ATTRACTION         = 0.8

ESTIMATOR_TIMEOUT       = 5.0
ESTIMATOR_THRESHOLD     = 0.05
ESTIMATOR_STATUS_EVERY  = 1.0
ABORT_ON_ESTIMATOR_FAIL = False

logging.basicConfig(level=logging.ERROR)

STATE_TAKEOFF = "TAKEOFF"
STATE_FLIGHT  = "FLIGHT"
STATE_LANDING = "LANDING"

AVOID_NONE  = "NONE"
AVOID_LEFT  = "LEFT"
AVOID_RIGHT = "RIGHT"
AVOID_FRONT = "FRONT"
AVOID_BACK  = "BACK"

sensor_data = {
    'front': float('inf'), 'back':  float('inf'),
    'left':  float('inf'), 'right': float('inf'),
    'up':    float('inf'), 'down':  float('inf'),
}

sensor_history = {
    'front': deque(maxlen=FILTER_SIZE), 'back':  deque(maxlen=FILTER_SIZE),
    'left':  deque(maxlen=FILTER_SIZE), 'right': deque(maxlen=FILTER_SIZE),
    'up':    deque(maxlen=FILTER_SIZE), 'down':  deque(maxlen=FILTER_SIZE),
}

height_history = deque(maxlen=HEIGHT_AVERAGE_SAMPLES)

current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}

current_state       = STATE_TAKEOFF
shutdown_requested   = False
avoid_state          = AVOID_NONE
target_height        = FLIGHT_HEIGHT

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
        sensor_history[key].append(SENSOR_MAX_RANGE if val == float('inf') else val)

    down_val = sensor_data['down']
    height_history.append(SENSOR_MAX_RANGE if down_val == float('inf') else down_val)


def position_callback(timestamp, data, logconf):
    global current_position
    current_position['x'] = data.get('stateEstimate.x', current_position['x'])
    current_position['y'] = data.get('stateEstimate.y', current_position['y'])
    current_position['z'] = data.get('stateEstimate.z', current_position['z'])


def get_position():
    return current_position['x'], current_position['y'], current_position['z']


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
    # only used in explore mode
    front = get_filtered('front')

    if front >= AVOID_DISTANCE:
        return FORWARD_SPEED

    factor = (front - CRITICAL_DISTANCE) / (AVOID_DISTANCE - CRITICAL_DISTANCE)
    factor = clamp(factor, 0.0, 1.0)
    return FORWARD_SPEED * factor


def compute_goal_velocity(dx, dy, dist):
    # simple P-controller with a speed cap, slows down near target
    if dist < 1e-6:
        return 0.0, 0.0
    speed = min(GOAL_SPEED_MAX, K_ATTRACTION * dist)
    return speed * dx / dist, speed * dy / dist


def compute_height_correction():
    current_height = get_average_height()
    error = target_height - current_height

    if abs(error) < HEIGHT_TOLERANCE:
        vz_height = 0.0
    else:
        vz_height = clamp(error * K_HEIGHT, -MAX_Z_SPEED, MAX_Z_SPEED)

    up = get_filtered('up')
    vz_ceiling = -compute_repulsion(up, CEILING_DISTANCE, K_REPULSION_CRIT)

    vz = vz_height + vz_ceiling
    return clamp(vz, -MAX_Z_SPEED, MAX_Z_SPEED)


def apply_bug_avoidance(vx_goal, vy_goal, vx, vy):
    # picks whichever axis isn't blocked to dodge around obstacles,
    # works no matter which direction we're trying to fly
    global avoid_state

    front = get_filtered('front')
    back  = get_filtered('back')
    left  = get_filtered('left')
    right = get_filtered('right')

    if abs(vx_goal) >= abs(vy_goal):
        blocked = (vx_goal >= 0 and front < AVOID_DISTANCE) or \
                  (vx_goal <  0 and back  < AVOID_DISTANCE)

        if blocked:
            if avoid_state not in (AVOID_LEFT, AVOID_RIGHT):
                avoid_state = AVOID_LEFT if left >= right else AVOID_RIGHT
                print(f"[AVOID] Blocked on X-axis → dodging {avoid_state} "
                      f"(L={left:.2f}m R={right:.2f}m)")
            vy += SIDE_AVOID_SPEED if avoid_state == AVOID_LEFT else -SIDE_AVOID_SPEED
        else:
            avoid_state = AVOID_NONE
    else:
        blocked = (vy_goal >= 0 and left  < AVOID_DISTANCE) or \
                  (vy_goal <  0 and right < AVOID_DISTANCE)

        if blocked:
            if avoid_state not in (AVOID_FRONT, AVOID_BACK):
                avoid_state = AVOID_FRONT if front >= back else AVOID_BACK
                print(f"[AVOID] Blocked on Y-axis → dodging {avoid_state} "
                      f"(F={front:.2f}m B={back:.2f}m)")
            vx += SIDE_AVOID_SPEED if avoid_state == AVOID_FRONT else -SIDE_AVOID_SPEED
        else:
            avoid_state = AVOID_NONE

    return vx, vy


def compute_velocity(dx=None, dy=None, dist=None):
    front = get_filtered('front')
    back  = get_filtered('back')
    left  = get_filtered('left')
    right = get_filtered('right')

    if dx is None:
        vx_goal = compute_forward_speed()
        vy_goal = 0.0
    else:
        vx_goal, vy_goal = compute_goal_velocity(dx, dy, dist)

    vx = vx_goal
    vx -= compute_repulsion(front, AVOID_DISTANCE,    K_REPULSION)
    vx += compute_repulsion(back,  AVOID_DISTANCE,    K_REPULSION)
    vx -= compute_repulsion(front, CRITICAL_DISTANCE, K_REPULSION_CRIT)
    vx += compute_repulsion(back,  CRITICAL_DISTANCE, K_REPULSION_CRIT)

    vy = vy_goal
    vy -= compute_repulsion(left,  AVOID_DISTANCE,    K_REPULSION)
    vy += compute_repulsion(right, AVOID_DISTANCE,    K_REPULSION)
    vy -= compute_repulsion(left,  CRITICAL_DISTANCE, K_REPULSION_CRIT)
    vy += compute_repulsion(right, CRITICAL_DISTANCE, K_REPULSION_CRIT)

    vx, vy = apply_bug_avoidance(vx_goal, vy_goal, vx, vy)

    vz = compute_height_correction()

    vx = clamp(vx, -MAX_SPEED, MAX_SPEED)
    vy = clamp(vy, -MAX_SPEED, MAX_SPEED)

    return vx, vy, vz


def print_status(vx, vy, vz, goal_info=""):
    x, y, z = get_position()
    print(
        f"[{current_state:10}] "
        f"Pos(x:{x:+.2f} y:{y:+.2f} z:{z:+.2f}) "
        f"F:{get_filtered('front'):5.2f}m B:{get_filtered('back'):5.2f}m "
        f"L:{get_filtered('left'):5.2f}m R:{get_filtered('right'):5.2f}m "
        f"D_avg:{get_average_height():5.2f}m "
        f"| vx:{vx:+.2f} vy:{vy:+.2f} vz:{vz:+.2f} "
        f"| Avoiding:{avoid_state} {goal_info}"
    )


def wait_for_position_estimator(scf, timeout=ESTIMATOR_TIMEOUT, threshold=ESTIMATOR_THRESHOLD):
    # bitcraze's example version has no timeout - it can hang forever
    # on a bad surface, so added one here plus some status output
    print("[INFO] Waiting for position estimator to converge...")

    log_config = LogConfig(name='KalmanVariance', period_in_ms=100)
    log_config.add_variable('kalman.varPX', 'float')
    log_config.add_variable('kalman.varPY', 'float')
    log_config.add_variable('kalman.varPZ', 'float')

    var_history = {
        'x': deque([1000] * 10, maxlen=10),
        'y': deque([1000] * 10, maxlen=10),
        'z': deque([1000] * 10, maxlen=10),
    }

    start_time = time.time()
    last_output = start_time

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_history['x'].append(data['kalman.varPX'])
            var_history['y'].append(data['kalman.varPY'])
            var_history['z'].append(data['kalman.varPZ'])

            spread_x = max(var_history['x']) - min(var_history['x'])
            spread_y = max(var_history['y']) - min(var_history['y'])
            spread_z = max(var_history['z']) - min(var_history['z'])

            now = time.time()

            if now - last_output > ESTIMATOR_STATUS_EVERY:
                print(f"[INFO] Kalman variance  ΔX:{spread_x:.5f} "
                      f"ΔY:{spread_y:.5f} ΔZ:{spread_z:.5f} "
                      f"(target < {threshold})")
                last_output = now

            if spread_x < threshold and spread_y < threshold and spread_z < threshold:
                print(f"[INFO] Position estimator converged "
                      f"(after {now - start_time:.1f}s).")
                return True

            if now - start_time > timeout:
                print(f"[WARN] Timeout ({timeout:.0f}s) while waiting for convergence!")
                print(f"[WARN] Last values: ΔX:{spread_x:.5f} "
                      f"ΔY:{spread_y:.5f} ΔZ:{spread_z:.5f}")
                print("[WARN] Position estimate may be inaccurate at the start!")
                return False

    return False


def reset_estimator(scf):
    scf.cf.param.set_value('stabilizer.estimator', '2')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    return wait_for_position_estimator(scf)


def normalize_waypoint(wp):
    if len(wp) == 2:
        return (wp[0], wp[1], FLIGHT_HEIGHT)
    return (wp[0], wp[1], wp[2])


def build_mission():
    mission = [normalize_waypoint(wp) for wp in WAYPOINTS]
    if mission and RETURN_TO_HOME:
        last = mission[-1]
        if abs(last[0]) > 1e-3 or abs(last[1]) > 1e-3:
            mission.append((0.0, 0.0, FLIGHT_HEIGHT))
    return mission


def fly_to_waypoint(mc, goal_x, goal_y, goal_z, index, total, mission_start_time):
    global avoid_state, target_height, shutdown_requested

    target_height = goal_z
    avoid_state = AVOID_NONE
    start_time = time.time()

    print(f"\n[WAYPOINT {index + 1}/{total}] Target: "
          f"x={goal_x:.2f} y={goal_y:.2f} z={goal_z:.2f}")

    while True:
        if shutdown_requested:
            return False

        if time.time() - mission_start_time > MAX_FLIGHT_TIME:
            print("[WARN] Maximum flight time reached → aborting mission.")
            shutdown_requested = True
            return False

        if time.time() - start_time > WAYPOINT_TIMEOUT:
            print(f"[WARN] Timeout at waypoint {index + 1} → skipping.")
            return True

        x, y, _ = get_position()
        dx = goal_x - x
        dy = goal_y - y
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_TOLERANCE:
            print(f"[WAYPOINT {index + 1}/{total}] reached (distance={dist:.2f}m)")
            return True

        vx, vy, vz = compute_velocity(dx, dy, dist)
        print_status(vx, vy, vz, goal_info=f"| WP{index + 1}/{total} dist:{dist:.2f}m")

        mc.start_linear_motion(vx, vy, vz)
        time.sleep(LOOP_DT)


def fly_explore_mode(mc):
    global shutdown_requested

    print("\n[INFO] No waypoints defined → Explore mode active.")
    start_time = time.time()

    while not shutdown_requested and (time.time() - start_time) < MAX_FLIGHT_TIME:
        vx, vy, vz = compute_velocity()
        print_status(vx, vy, vz)
        mc.start_linear_motion(vx, vy, vz)
        time.sleep(LOOP_DT)


def main():
    global current_state, shutdown_requested

    cflib.crtp.init_drivers()

    print("[INFO] Connecting to Crazyflie...")
    print("[INFO] Press STRG+C to land\n")

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        sensor_log = LogConfig(name='Ranger', period_in_ms=LOG_PERIOD_MS)
        sensor_log.add_variable('range.front',  'uint16_t')
        sensor_log.add_variable('range.back',   'uint16_t')
        sensor_log.add_variable('range.left',   'uint16_t')
        sensor_log.add_variable('range.right',  'uint16_t')
        sensor_log.add_variable('range.up',     'uint16_t')
        sensor_log.add_variable('range.zrange', 'uint16_t')

        scf.cf.log.add_config(sensor_log)
        sensor_log.data_received_cb.add_callback(sensor_callback)
        sensor_log.start()

        position_log = LogConfig(name='Position', period_in_ms=100)
        position_log.add_variable('stateEstimate.x', 'float')
        position_log.add_variable('stateEstimate.y', 'float')
        position_log.add_variable('stateEstimate.z', 'float')

        scf.cf.log.add_config(position_log)
        position_log.data_received_cb.add_callback(position_callback)
        position_log.start()

        time.sleep(0.5)

        converged = reset_estimator(scf)

        if not converged and ABORT_ON_ESTIMATOR_FAIL:
            print("[ERROR] Position estimator did not converge. "
                  "Please check surface/lighting. Aborting.")
            sensor_log.stop()
            position_log.stop()
            return

        with MotionCommander(scf, default_height=FLIGHT_HEIGHT) as mc:

            current_state = STATE_TAKEOFF
            print(f"[{STATE_TAKEOFF}]")
            time.sleep(2.0)

            height_history.extend([FLIGHT_HEIGHT] * HEIGHT_AVERAGE_SAMPLES)

            current_state = STATE_FLIGHT
            print(f"[{STATE_FLIGHT}]\n")

            mission = build_mission()
            mission_start_time = time.time()

            if mission:
                for i, (gx, gy, gz) in enumerate(mission):
                    keep_going = fly_to_waypoint(
                        mc, gx, gy, gz, i, len(mission), mission_start_time
                    )
                    if not keep_going and shutdown_requested:
                        break
            else:
                fly_explore_mode(mc)

            current_state = STATE_LANDING
            print(f"\n[{STATE_LANDING}]")

            mc.stop()
            time.sleep(0.3)
            mc.land()

            print(f"[{STATE_LANDING}] done")

        sensor_log.stop()
        position_log.stop()

    print("[INFO] Connection closed.")


if __name__ == '__main__':
    main()