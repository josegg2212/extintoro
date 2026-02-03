# EXTINTORO

EXTINTORO es un proyecto para el robot **DJI RoboMaster S1** que integra **patrulla**, **detección de fuego** (visión artificial) y **actuación** mediante una **máquina de estados**.

Repositorio (rama de entrega): https://github.com/josegg2212/extintoro.git  *(branch: `develop`)*

---

## Estructura del repositorio

- **`extintoro.py`**  
  Script principal con **toda la lógica de la máquina de estados** (patrulla / searching / aiming / shooting) y el control por **emulación de teclado/ratón** para operar la app (ejecutar como administrador).

- **`fire_model.pt`**  
  Modelo YOLO entrenado para **detección de fuego** (pesos finales usados por `extintoro.py`).

- **`train_fuego/`**  
  Carpeta con los **resultados del entrenamiento** (métricas, gráficas y artefactos generados por Ultralytics, incluyendo pesos como `best.pt` y ficheros de resultados).

- **`yolo_train.py`**  
  Script para **entrenar** el modelo YOLO a partir de pesos preentrenados (pipeline típico de Ultralytics).

- **`yolo_test.py`**  
  Script para **test/validación** del modelo sobre imágenes de prueba (genera predicciones y guarda resultados).

- **`siguelineas_app.py`** y **`siguelineas_app.png`**  
  Implementación de la **habilidad personalizada** en la app de RoboMaster (modo patrulla / sigue-líneas).  
  El `.png` es la captura del diagrama por bloques y el `.py` es el export del programa.

---

## Requisitos

- **Python 3.9+** (recomendado 3.10)
- Sistema con permisos para **capturar pantalla** y **emular teclado/ratón** (según el sistema operativo).
- La **app de RoboMaster** debe estar abierta y visible para que la emulación funcione (EXTINTORO controla “desde fuera” la interfaz).

---

## Instalación

1) Clonar el repo y cambiar a la rama:

```bash
git clone https://github.com/josegg2212/extintoro.git
cd extintoro
git checkout develop
```

2) Crear entorno virtual e instalar dependencias:

```bash
python -m venv .venv

# Windows:
.venv\Scripts\activate

# Linux/Mac:
source .venv/bin/activate

pip install -U pip
pip install ultralytics opencv-python mss pynput numpy
```

> Nota: `ultralytics` gestiona las dependencias necesarias para ejecutar YOLO (incluyendo PyTorch según plataforma).

---

## Ejecución

### 1) Ejecutar EXTINTORO (máquina de estados)

Asegúrate de tener la app de RoboMaster abierta y lista para recibir entradas.

```bash
python extintoro.py
```

> Si tu configuración de monitores no coincide, revisa en el script los parámetros de captura (monitor / región) para adaptarlo a tu pantalla.

---

### 2) Entrenar el modelo (opcional)

`yolo_train.py` espera la ruta del dataset (por ejemplo `data.yaml`) en la variable de entorno `PATH_DATASET`.

Ejemplo:

```bash
# Linux/Mac
export PATH_DATASET="/ruta/a/data.yaml"
python yolo_train.py

# Windows (PowerShell)
$env:PATH_DATASET="C:\ruta\a\data.yaml"
python yolo_train.py
```

Los resultados del entrenamiento se guardan en el directorio configurado por Ultralytics (y/o en la carpeta del proyecto según el script).

---

### 3) Test/Predicción del modelo (opcional)

`yolo_test.py` permite lanzar predicciones sobre un directorio de test y guardar resultados.

```bash
python yolo_test.py
```

---

## Dependencias principales (resumen)

- `ultralytics` (YOLO)
- `opencv-python` (visualización y utilidades de visión)
- `mss` (captura de pantalla)
- `pynput` (emulación de teclado/ratón)
- `numpy`

---

## Notas

- `siguelineas_app.py/.png` no se ejecutan desde el PC: corresponden a la habilidad configurada en la app/robot para el modo patrulla.
- `train_fuego/` contiene las métricas y resultados del entrenamiento para documentar el rendimiento del detector.
