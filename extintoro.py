import time
import numpy as np
import cv2
from mss import mss
from pynput.keyboard import Controller as KeyboardController, Key
from pynput.mouse import Controller as MouseController, Button
from ultralytics import YOLO

# Parametros de punteria y movimiento:
# - TOLERANCE: margen en pixeles para considerar el objetivo centrado.
# - TIME_PER_PIXEL: segundos de pulsacion por pixel (sensibilidad).
# - MIN_DURATION: duracion minima de pulsacion.
# - MAX_DURATION: duracion maxima de pulsacion.
TOLERANCE = 60
TIME_PER_PIXEL = 0.0001
MIN_DURATION = 0.01
MAX_DURATION = 0.15

# Parametros de disparo:
# - SHOOT_COOLDOWN: segundos entre disparos.
# - SHOTS_PER_BURST: disparos por rafaga antes de volver a AIMING.
SHOOT_COOLDOWN = 0.5
SHOTS_PER_BURST = 1

# Parametros de perdida de objetivo:
# - FIRE_LOST_GRACE: segundos de espera en AIMING tras perder deteccion.
FIRE_LOST_GRACE = 3.0

# Parametros de la ventana de previsualizacion:
# - DISPLAY_MONITOR_INDEX: monitor donde se muestra la ventana (1 = principal).
# - PREVIEW_FULLSCREEN: muestra la ventana a pantalla completa.
DISPLAY_MONITOR_INDEX = 1
PREVIEW_FULLSCREEN = True

# Parametros de captura:
# - CAPTURE_MONITOR_INDEX: monitor desde el que se capturan los frames.
CAPTURE_MONITOR_INDEX = 2

# Parametros de tiempo sin deteccion en SEARCHING:
# - SEARCHING_NO_BBOX_TIMEOUT: segundos sin detecciones antes de DESPLAZAMIENTO.
SEARCHING_NO_BBOX_TIMEOUT = 20.0
searching_no_bbox_start_time = None

# Parametros de desplazamiento:
# - DISPLACEMENT_DURATION: segundos totales de desplazamiento.
# - Q_PULSE: duracion de la pulsacion de la tecla "q".
DISPLACEMENT_DURATION = 10.0
Q_PULSE = 0.05
displacement_started = False
displacement_start_time = 0.0

# Parametros de barrido en SEARCHING:
# - SWEEP_H_KEY: tecla para barrido horizontal.
# - SWEEP_H_PULSE: duracion del pulso horizontal.
# - SWEEP_H_PAUSE: pausa extra tras el pulso horizontal.
# - SWEEP_V_PULSE: duracion del pulso vertical.
# - SWEEP_V_STEPS_PER_DIR: pulsos antes de invertir direccion vertical.
# - SWEEP_V_EVERY_N_FRAMES: aplica pulso vertical cada N frames.
SWEEP_H_KEY = Key.right
SWEEP_H_PULSE = 0.03
SWEEP_H_PAUSE = 0.2
SWEEP_V_PULSE = 0.02
SWEEP_V_STEPS_PER_DIR = 10
SWEEP_V_EVERY_N_FRAMES = 2

# Estado de disparo en ejecucion:
last_shot_time = 0
shots_in_burst = 0
last_fire_seen_time = 0

# Estado interno del barrido vertical:
sweep_v_dir = 1
sweep_v_steps = 0
sweep_frame_counter = 0


def capture_fullscreen_frame(sct, monitor_index=0):
    """
    Captura un frame completo del monitor indicado.

    Args:
        sct: instancia de mss() para captura de pantalla.
        monitor_index: indice en sct.monitors (0 = pantalla virtual, 1+ por monitor).

    Returns:
        Imagen BGR en un numpy array uint8 contiguo.
    """
    monitor = sct.monitors[monitor_index]
    sct_img = sct.grab(monitor)
    frame = np.array(sct_img, dtype=np.uint8)[:, :, :3].copy()
    return np.ascontiguousarray(frame)


def press_key_variable(keyboard, key, duration):
    """
    Pulsa una tecla durante una duracion variable.

    Args:
        keyboard: controlador de teclado de pynput.
        key: tecla o caracter a pulsar.
        duration: segundos que se mantiene pulsada la tecla.
    """
    keyboard.press(key)
    time.sleep(duration)
    keyboard.release(key)


def run_state_desplazamiento(keyboard):
    """
    Estado: DESPLAZAMIENTO.

    Flujo:
        - Al entrar, pulsa "q" para activar el desplazamiento.
        - Tras DISPLACEMENT_DURATION, pulsa "q" para desactivar y pasa a SEARCHING.

    Args:
        keyboard: controlador de teclado de pynput.

    Returns:
        Nombre del siguiente estado.
    """
    global displacement_started, displacement_start_time
    global searching_no_bbox_start_time

    # Resetea el temporizador de SEARCHING mientras se desplaza.
    searching_no_bbox_start_time = None

    if not displacement_started:
        print(">> Entrando en DESPLAZAMIENTO: pulsando 'q' (activar).")
        press_key_variable(keyboard, "q", Q_PULSE)
        displacement_start_time = time.time()
        displacement_started = True
        return "DESPLAZAMIENTO"

    elapsed = time.time() - displacement_start_time
    if elapsed >= DISPLACEMENT_DURATION:
        print(">> Fin de DESPLAZAMIENTO: pulsando 'q' (desactivar) y pasando a SEARCHING.")
        press_key_variable(keyboard, "q", Q_PULSE)
        displacement_started = False
        return "SEARCHING"

    time.sleep(0.01)
    return "DESPLAZAMIENTO"


def run_state_searching(results, keyboard):
    """
    Estado: SEARCHING.

    Flujo:
        - Si detecta objetivo, pasa a AIMING.
        - Si no detecta durante SEARCHING_NO_BBOX_TIMEOUT, pasa a DESPLAZAMIENTO.
        - Si no hay objetivo, realiza barrido horizontal y vertical.

    Args:
        results: lista de resultados de YOLO para el frame actual.
        keyboard: controlador de teclado de pynput.

    Returns:
        Nombre del siguiente estado.
    """
    global sweep_v_dir, sweep_v_steps, sweep_frame_counter
    global searching_no_bbox_start_time

    # La clase con id 0 se toma como objetivo.
    fire = results[0].boxes.cls == 0
    now = time.time()

    if fire.any():
        searching_no_bbox_start_time = None
        print(">> Objetivo detectado. Cambiando a AIMING.")
        return "AIMING"

    if searching_no_bbox_start_time is None:
        searching_no_bbox_start_time = now
    else:
        elapsed_no_bbox = now - searching_no_bbox_start_time
        if elapsed_no_bbox >= SEARCHING_NO_BBOX_TIMEOUT:
            print(">> Tiempo sin deteccion en SEARCHING. Pasando a DESPLAZAMIENTO.")
            searching_no_bbox_start_time = None
            return "DESPLAZAMIENTO"

    # Barrido horizontal continuo.
    press_key_variable(keyboard, SWEEP_H_KEY, SWEEP_H_PULSE)
    if SWEEP_H_PAUSE > 0:
        time.sleep(SWEEP_H_PAUSE)

    # Barrido vertical con cambio de direccion.
    sweep_frame_counter += 1
    if sweep_frame_counter % SWEEP_V_EVERY_N_FRAMES == 0:
        vkey = Key.up if sweep_v_dir > 0 else Key.down
        press_key_variable(keyboard, vkey, SWEEP_V_PULSE)

        sweep_v_steps += 1
        if sweep_v_steps >= SWEEP_V_STEPS_PER_DIR:
            sweep_v_steps = 0
            sweep_v_dir *= -1

    return "SEARCHING"


def run_state_aiming(results, frame, keyboard):
    """
    Estado: AIMING.

    Flujo:
        - Si pierde el objetivo mas de FIRE_LOST_GRACE, pasa a SEARCHING.
        - Si el objetivo esta centrado, pasa a SHOOTING.
        - Si no esta centrado, ajusta la mira y permanece en AIMING.

    Args:
        results: lista de resultados de YOLO para el frame actual.
        frame: frame BGR actual para dibujar overlays.
        keyboard: controlador de teclado de pynput.

    Returns:
        Nombre del siguiente estado.
    """
    global last_fire_seen_time

    # La clase con id 0 se toma como objetivo.
    fire = results[0].boxes.cls == 0
    now = time.time()

    if not fire.any():
        if last_fire_seen_time and (now - last_fire_seen_time) < FIRE_LOST_GRACE:
            print(">> Sin deteccion, esperando en AIMING.")
            return "AIMING"
        print(">> Objetivo perdido. Volviendo a SEARCHING.")
        return "SEARCHING"

    last_fire_seen_time = now
    fire_box = results[0].boxes.xyxy[fire][0]
    target_center_x = int((fire_box[0] + fire_box[2]) / 2)
    target_center_y = int((fire_box[1] + fire_box[3]) / 2)

    screen_center_x = frame.shape[1] // 2
    screen_center_y = frame.shape[0] // 2

    # Overlay de la caja y el centro de pantalla.
    cv2.rectangle(
        frame,
        (int(fire_box[0]), int(fire_box[1])),
        (int(fire_box[2]), int(fire_box[3])),
        (0, 255, 0),
        2,
    )
    cv2.drawMarker(
        frame,
        (screen_center_x, screen_center_y),
        (0, 0, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=20,
        thickness=2,
    )

    dx = target_center_x - screen_center_x
    dy = target_center_y - screen_center_y

    if abs(dx) <= TOLERANCE and abs(dy) <= TOLERANCE:
        print(f">> Objetivo centrado (dx:{dx}, dy:{dy}). Cambiando a SHOOTING.")
        return "SHOOTING"

    # Correccion horizontal.
    if abs(dx) > TOLERANCE:
        duration = abs(dx) * TIME_PER_PIXEL
        duration = max(MIN_DURATION, min(duration, MAX_DURATION))
        press_key_variable(keyboard, Key.right if dx > 0 else Key.left, duration)

    # Correccion vertical.
    if abs(dy) > TOLERANCE:
        duration = abs(dy) * TIME_PER_PIXEL
        duration = max(MIN_DURATION, min(duration, MAX_DURATION))
        press_key_variable(keyboard, Key.down if dy > 0 else Key.up, duration)

    return "AIMING"


def run_state_shooting(mouse):
    """
    Estado: SHOOTING.

    Flujo:
        - Dispara en rafaga con cooldown entre disparos.
        - Vuelve a AIMING cuando se completa la rafaga.

    Args:
        mouse: controlador de raton de pynput.

    Returns:
        Nombre del siguiente estado.
    """
    global last_shot_time, shots_in_burst
    current_time = time.time()

    if shots_in_burst >= SHOTS_PER_BURST:
        shots_in_burst = 0
        return "AIMING"

    if current_time - last_shot_time > SHOOT_COOLDOWN:
        print(f">> !! BANG !! (Click Izquierdo) [{shots_in_burst + 1}/{SHOTS_PER_BURST}]")
        mouse.click(Button.left, 1)
        last_shot_time = current_time
        shots_in_burst += 1
    else:
        print(".. Cargando disparo (cooldown) ..")

    return "SHOOTING"


def main():
    """
    Pipeline general:
        1) Carga el modelo y controladores de entrada.
        2) Inicializa el estado y espera el foco en el juego.
        3) Crea la ventana de previsualizacion.
        4) Bucle: captura, deteccion si aplica, FSM, overlay y display.
        5) Sale con ESC.
    """
    model = YOLO("fire_model.pt")
    keyboard = KeyboardController()
    mouse = MouseController()

    current_state = "DESPLAZAMIENTO"

    print("5 segundos para cambiar a la ventana del juego...")
    time.sleep(5)
    print("INICIANDO MAQUINA DE ESTADOS")

    with mss() as sct:
        window_name = "FSM Bot"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Ubica la ventana en el monitor indicado.
        if 0 <= DISPLAY_MONITOR_INDEX < len(sct.monitors):
            mon = sct.monitors[DISPLAY_MONITOR_INDEX]
            cv2.moveWindow(window_name, mon["left"], mon["top"])
            if PREVIEW_FULLSCREEN:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            # Captura del frame del juego.
            frame = capture_fullscreen_frame(sct, monitor_index=CAPTURE_MONITOR_INDEX)

            # Ejecuta YOLO solo en estados que lo necesitan.
            results = None
            if current_state in ("SEARCHING", "AIMING"):
                results = model(frame, verbose=False, conf=0.5)

            # Despacho de la maquina de estados.
            if current_state == "DESPLAZAMIENTO":
                current_state = run_state_desplazamiento(keyboard)
            elif current_state == "SEARCHING":
                current_state = run_state_searching(results, keyboard)
            elif current_state == "AIMING":
                current_state = run_state_aiming(results, frame, keyboard)
            elif current_state == "SHOOTING":
                current_state = run_state_shooting(mouse)

            # Overlay de estado actual.
            cv2.putText(
                frame,
                f"ESTADO: {current_state}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2,
            )

            cv2.imshow(window_name, frame)

            # Salir con ESC (27) para no interferir con la tecla "q".
            if cv2.waitKey(1) & 0xFF == 27:
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
