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

# ============================================================
# PARAMETER
# ============================================================

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

# ── Höhenglättung über Zeitfenster ────────────────────
LOG_PERIOD_MS           = 50
HEIGHT_AVERAGE_TIME     = 3.0
HEIGHT_AVERAGE_SAMPLES  = int((HEIGHT_AVERAGE_TIME * 1000) / LOG_PERIOD_MS)

# ============================================================
# WEGPUNKT-NAVIGATION
# ============================================================
#
# Koordinaten relativ zum Startpunkt (Take-off-Position), in Metern:
#   x = vorwärts (Blickrichtung beim Start)
#   y = links
#   z = Höhe über Grund
#
# Format je Wegpunkt: (x, y)  ODER  (x, y, z)
# Fehlt z, wird FLIGHT_HEIGHT verwendet.
#
# Ist WAYPOINTS leer -> "Explore-Modus": die Drohne fliegt
# konstant vorwärts und weicht Hindernissen aus.
#
WAYPOINTS = [
    (0.0, 1.0),
    (2.5, 1.0),
    (2.5, 0.5),
    (3.5, 0.5),
]

RETURN_TO_HOME       = False    # nach letztem Wegpunkt zurück zu (0,0)
WAYPOINT_TOLERANCE   = 0.05    # m - ab wann gilt ein Wegpunkt als erreicht
WAYPOINT_TIMEOUT     = 25.0    # s - max. Zeit pro Wegpunkt, dann überspringen
GOAL_SPEED_MAX       = 0.2     # m/s - max. Geschwindigkeit Richtung Ziel
K_ATTRACTION         = 0.8     # P-Regler-Verstärkung Richtung Ziel

# ============================================================
# POSITIONSSCHÄTZER (KALMAN-FILTER) - KONVERGENZPRÜFUNG
# ============================================================

ESTIMATOR_TIMEOUT       = 5.0   # s - max. Wartezeit auf Konvergenz
ESTIMATOR_THRESHOLD     = 0.05  # Ziel-Schwellwert für Varianz-Spanne
ESTIMATOR_STATUS_EVERY  = 1.0    # s - Intervall für Diagnose-Ausgabe
ABORT_ON_ESTIMATOR_FAIL = False  # True = Mission abbrechen, falls Konvergenz scheitert

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

# ── Positionsschätzung (Flow Deck / Kalman-Filter) ──────
current_position = {'x': 0.0, 'y': 0.0, 'z': 0.0}

current_state        = STATE_TAKEOFF
shutdown_requested    = False
avoid_state           = AVOID_NONE
ziel_hoehe            = FLIGHT_HEIGHT   # aktuelle Soll-Höhe (pro Wegpunkt änderbar)

SENSOR_MAX_RANGE = 4.0


def signal_handler(sig, frame):
    global shutdown_requested
    print("\n[INFO] STRG+C erkannt → Landung wird eingeleitet...")
    shutdown_requested = True


signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# SENSOR- UND POSITIONS-CALLBACKS
# ============================================================

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


# ============================================================
# HILFSFUNKTIONEN
# ============================================================

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


def berechne_abstoßung(distanz, avoid_dist, k):
    if distanz < avoid_dist and distanz > 0:
        return k * (1.0 / distanz - 1.0 / avoid_dist)
    return 0.0


def berechne_forward_speed():
    """Nur für den Explore-Modus (kein Wegpunkt aktiv)."""
    front = get_filtered('front')

    if front >= AVOID_DISTANCE:
        return FORWARD_SPEED

    faktor = (front - CRITICAL_DISTANCE) / (AVOID_DISTANCE - CRITICAL_DISTANCE)
    faktor = clamp(faktor, 0.0, 1.0)
    return FORWARD_SPEED * faktor


def berechne_zielgeschwindigkeit(dx, dy, dist):
    """
    Anziehungs-Komponente Richtung Wegpunkt
    (P-Regler mit Geschwindigkeitsdeckel -> sanftes Abbremsen kurz vorm Ziel).
    """
    if dist < 1e-6:
        return 0.0, 0.0
    speed = min(GOAL_SPEED_MAX, K_ATTRACTION * dist)
    return speed * dx / dist, speed * dy / dist


def berechne_hoehenkorrektur():
    aktuelle_hoehe = get_average_height()
    fehler = ziel_hoehe - aktuelle_hoehe

    if abs(fehler) < HEIGHT_TOLERANCE:
        vz_height = 0.0
    else:
        vz_height = clamp(fehler * K_HEIGHT, -MAX_Z_SPEED, MAX_Z_SPEED)

    up = get_filtered('up')
    vz_ceiling = -berechne_abstoßung(up, CEILING_DISTANCE, K_REPULSION_CRIT)

    vz = vz_height + vz_ceiling
    return clamp(vz, -MAX_Z_SPEED, MAX_Z_SPEED)


def wende_bug_ausweichlogik_an(vx_goal, vy_goal, vx, vy):
    """
    Verallgemeinerte "Bug"-Ausweichlogik: erkennt, ob die Achse in
    Richtung des (Ziel-)Geschwindigkeitsvektors blockiert ist, und
    weicht auf der jeweils anderen Achse aus. Funktioniert unabhängig
    davon, ob die Drohne vorwärts, rückwärts oder seitwärts zum Ziel
    fliegen muss.
    """
    global avoid_state

    front = get_filtered('front')
    back  = get_filtered('back')
    left  = get_filtered('left')
    right = get_filtered('right')

    if abs(vx_goal) >= abs(vy_goal):
        blockiert = (vx_goal >= 0 and front < AVOID_DISTANCE) or \
                    (vx_goal <  0 and back  < AVOID_DISTANCE)

        if blockiert:
            if avoid_state not in (AVOID_LEFT, AVOID_RIGHT):
                avoid_state = AVOID_LEFT if left >= right else AVOID_RIGHT
                print(f"[AVOID] Blockiert auf X-Achse → weiche {avoid_state} aus "
                      f"(L={left:.2f}m R={right:.2f}m)")
            vy += SIDE_AVOID_SPEED if avoid_state == AVOID_LEFT else -SIDE_AVOID_SPEED
        else:
            avoid_state = AVOID_NONE
    else:
        blockiert = (vy_goal >= 0 and left  < AVOID_DISTANCE) or \
                    (vy_goal <  0 and right < AVOID_DISTANCE)

        if blockiert:
            if avoid_state not in (AVOID_FRONT, AVOID_BACK):
                avoid_state = AVOID_FRONT if front >= back else AVOID_BACK
                print(f"[AVOID] Blockiert auf Y-Achse → weiche {avoid_state} aus "
                      f"(F={front:.2f}m B={back:.2f}m)")
            vx += SIDE_AVOID_SPEED if avoid_state == AVOID_FRONT else -SIDE_AVOID_SPEED
        else:
            avoid_state = AVOID_NONE

    return vx, vy


def berechne_geschwindigkeit(dx=None, dy=None, dist=None):
    """
    Kombiniert Zielanziehung (falls dx/dy gesetzt, sonst Explore-Modus)
    mit sensorbasierter Abstoßung (Potentialfeld) + Bug-Ausweichlogik.
    """
    front = get_filtered('front')
    back  = get_filtered('back')
    left  = get_filtered('left')
    right = get_filtered('right')

    if dx is None:
        vx_goal = berechne_forward_speed()
        vy_goal = 0.0
    else:
        vx_goal, vy_goal = berechne_zielgeschwindigkeit(dx, dy, dist)

    vx = vx_goal
    vx -= berechne_abstoßung(front, AVOID_DISTANCE,    K_REPULSION)
    vx += berechne_abstoßung(back,  AVOID_DISTANCE,    K_REPULSION)
    vx -= berechne_abstoßung(front, CRITICAL_DISTANCE, K_REPULSION_CRIT)
    vx += berechne_abstoßung(back,  CRITICAL_DISTANCE, K_REPULSION_CRIT)

    vy = vy_goal
    vy -= berechne_abstoßung(left,  AVOID_DISTANCE,    K_REPULSION)
    vy += berechne_abstoßung(right, AVOID_DISTANCE,    K_REPULSION)
    vy -= berechne_abstoßung(left,  CRITICAL_DISTANCE, K_REPULSION_CRIT)
    vy += berechne_abstoßung(right, CRITICAL_DISTANCE, K_REPULSION_CRIT)

    vx, vy = wende_bug_ausweichlogik_an(vx_goal, vy_goal, vx, vy)

    vz = berechne_hoehenkorrektur()

    vx = clamp(vx, -MAX_SPEED, MAX_SPEED)
    vy = clamp(vy, -MAX_SPEED, MAX_SPEED)

    return vx, vy, vz


def print_status(vx, vy, vz, ziel_info=""):
    x, y, z = get_position()
    print(
        f"[{current_state:10}] "
        f"Pos(x:{x:+.2f} y:{y:+.2f} z:{z:+.2f}) "
        f"F:{get_filtered('front'):5.2f}m B:{get_filtered('back'):5.2f}m "
        f"L:{get_filtered('left'):5.2f}m R:{get_filtered('right'):5.2f}m "
        f"D_avg:{get_average_height():5.2f}m "
        f"| vx:{vx:+.2f} vy:{vy:+.2f} vz:{vz:+.2f} "
        f"| Ausweichen:{avoid_state} {ziel_info}"
    )


# ============================================================
# POSITIONS-SCHÄTZER ZURÜCKSETZEN & AUF KONVERGENZ WARTEN
# ============================================================

def wait_for_position_estimator(scf,
                                 timeout=ESTIMATOR_TIMEOUT,
                                 threshold=ESTIMATOR_THRESHOLD):
    """
    Wartet, bis die Kalman-Varianz für x/y/z stabil genug ist.

    Verbesserungen gegenüber der Standard-Bitcraze-Version:
      - Timeout, damit das Skript nicht ewig hängt (z. B. bei
        schlechter Beleuchtung oder unstrukturiertem Untergrund,
        wodurch der Schwellwert nie unterschritten wird)
      - Laufende Diagnose-Ausgabe, damit sichtbar ist, ob und wie
        schnell sich die Werte stabilisieren
      - Rückgabewert (True/False), damit main() bei Bedarf reagieren
        kann (z. B. Mission abbrechen)
    """
    print("[INFO] Warte auf Konvergenz des Positionsschätzers...")

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
    letzte_ausgabe = start_time

    with SyncLogger(scf, log_config) as logger:
        for log_entry in logger:
            data = log_entry[1]

            var_history['x'].append(data['kalman.varPX'])
            var_history['y'].append(data['kalman.varPY'])
            var_history['z'].append(data['kalman.varPZ'])

            spanne_x = max(var_history['x']) - min(var_history['x'])
            spanne_y = max(var_history['y']) - min(var_history['y'])
            spanne_z = max(var_history['z']) - min(var_history['z'])

            jetzt = time.time()

            if jetzt - letzte_ausgabe > ESTIMATOR_STATUS_EVERY:
                print(f"[INFO] Kalman-Varianz  ΔX:{spanne_x:.5f} "
                      f"ΔY:{spanne_y:.5f} ΔZ:{spanne_z:.5f} "
                      f"(Ziel < {threshold})")
                letzte_ausgabe = jetzt

            if spanne_x < threshold and spanne_y < threshold and spanne_z < threshold:
                print(f"[INFO] Positionsschätzer konvergiert "
                      f"(nach {jetzt - start_time:.1f}s).")
                return True

            if jetzt - start_time > timeout:
                print(f"[WARN] Timeout ({timeout:.0f}s) beim Warten auf Konvergenz!")
                print(f"[WARN] Letzte Werte: ΔX:{spanne_x:.5f} "
                      f"ΔY:{spanne_y:.5f} ΔZ:{spanne_z:.5f}")
                print("[WARN] Positionsschätzung könnte am Anfang ungenau sein!")
                return False

    return False


def reset_estimator(scf):
    scf.cf.param.set_value('stabilizer.estimator', '2')  # Kalman-Filter erzwingen
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '1')
    time.sleep(0.1)
    scf.cf.param.set_value('kalman.resetEstimation', '0')
    return wait_for_position_estimator(scf)


# ============================================================
# WEGPUNKTE NORMALISIEREN
# ============================================================

def normalisiere_wegpunkt(wp):
    if len(wp) == 2:
        return (wp[0], wp[1], FLIGHT_HEIGHT)
    return (wp[0], wp[1], wp[2])


def baue_mission():
    mission = [normalisiere_wegpunkt(wp) for wp in WAYPOINTS]
    if mission and RETURN_TO_HOME:
        letzter = mission[-1]
        if abs(letzter[0]) > 1e-3 or abs(letzter[1]) > 1e-3:
            mission.append((0.0, 0.0, FLIGHT_HEIGHT))
    return mission


# ============================================================
# WEGPUNKT ANFLIEGEN
# ============================================================

def fliege_zu_wegpunkt(mc, ziel_x, ziel_y, ziel_z, index, gesamt, mission_start_time):
    global avoid_state, ziel_hoehe, shutdown_requested

    ziel_hoehe = ziel_z
    avoid_state = AVOID_NONE
    start_time = time.time()

    print(f"\n[WEGPUNKT {index + 1}/{gesamt}] Ziel: "
          f"x={ziel_x:.2f} y={ziel_y:.2f} z={ziel_z:.2f}")

    while True:
        if shutdown_requested:
            return False

        if time.time() - mission_start_time > MAX_FLIGHT_TIME:
            print("[WARN] Maximale Flugzeit erreicht → Abbruch der Mission.")
            shutdown_requested = True
            return False

        if time.time() - start_time > WAYPOINT_TIMEOUT:
            print(f"[WARN] Timeout bei Wegpunkt {index + 1} → überspringe.")
            return True

        x, y, _ = get_position()
        dx = ziel_x - x
        dy = ziel_y - y
        dist = math.hypot(dx, dy)

        if dist < WAYPOINT_TOLERANCE:
            print(f"[WEGPUNKT {index + 1}/{gesamt}] erreicht (Abstand={dist:.2f}m)")
            return True

        vx, vy, vz = berechne_geschwindigkeit(dx, dy, dist)
        print_status(vx, vy, vz, ziel_info=f"| WP{index + 1}/{gesamt} dist:{dist:.2f}m")

        mc.start_linear_motion(vx, vy, vz)
        time.sleep(LOOP_DT)


def fliege_explore_modus(mc):
    global shutdown_requested

    print("\n[INFO] Keine Wegpunkte definiert → Explore-Modus aktiv.")
    start_time = time.time()

    while not shutdown_requested and (time.time() - start_time) < MAX_FLIGHT_TIME:
        vx, vy, vz = berechne_geschwindigkeit()
        print_status(vx, vy, vz)
        mc.start_linear_motion(vx, vy, vz)
        time.sleep(LOOP_DT)


# ============================================================
# HAUPTPROGRAMM
# ============================================================

def main():
    global current_state, shutdown_requested

    cflib.crtp.init_drivers()

    print("[INFO] Verbinde mit Crazyflie...")
    print("[INFO] STRG+C zum Landen drücken\n")

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        # ── Sensor-Log (Multiranger) ──────────────────
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

        # ── Positions-Log (Flow Deck / Kalman) ────────
        position_log = LogConfig(name='Position', period_in_ms=100)
        position_log.add_variable('stateEstimate.x', 'float')
        position_log.add_variable('stateEstimate.y', 'float')
        position_log.add_variable('stateEstimate.z', 'float')

        scf.cf.log.add_config(position_log)
        position_log.data_received_cb.add_callback(position_callback)
        position_log.start()

        time.sleep(0.5)

        # ── Kalman-Filter zurücksetzen & Konvergenz abwarten ──
        konvergiert = reset_estimator(scf)

        if not konvergiert and ABORT_ON_ESTIMATOR_FAIL:
            print("[FEHLER] Positionsschätzer konvergiert nicht. "
                  "Bitte Untergrund/Beleuchtung prüfen. Abbruch.")
            sensor_log.stop()
            position_log.stop()
            return

        with MotionCommander(scf, default_height=FLIGHT_HEIGHT) as mc:

            # ── TAKEOFF ──────────────────────────────
            current_state = STATE_TAKEOFF
            print(f"[{STATE_TAKEOFF}]")
            time.sleep(2.0)

            height_history.extend([FLIGHT_HEIGHT] * HEIGHT_AVERAGE_SAMPLES)

            current_state = STATE_FLIGHT
            print(f"[{STATE_FLIGHT}]\n")

            # ── FLIGHT / NAVIGATION ──────────────────
            mission = baue_mission()
            mission_start_time = time.time()

            if mission:
                for i, (zx, zy, zz) in enumerate(mission):
                    weiter = fliege_zu_wegpunkt(
                        mc, zx, zy, zz, i, len(mission), mission_start_time
                    )
                    if not weiter and shutdown_requested:
                        break
            else:
                fliege_explore_modus(mc)

            # ── LANDING ──────────────────────────────
            current_state = STATE_LANDING
            print(f"\n[{STATE_LANDING}]")

            mc.stop()
            time.sleep(0.3)
            mc.land()

            print(f"[{STATE_LANDING}] fertig")

        sensor_log.stop()
        position_log.stop()

    print("[INFO] Verbindung getrennt.")


if __name__ == '__main__':
    main()