# GLNO para la ecuación de Schrödinger en superficies

Proyecto GLNO para aproximar la evolución temporal de la ecuación de Schrödinger en superficies.
El código se ejecuta desde este directorio, sin importar módulos de la carpeta padre.
El modelo aprende la evolución de paquetes gaussianos en superficies; Crank–Nicolson proporciona las soluciones de referencia.

## Instalación

Python 3.11 o posterior, desde esta carpeta:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test,notebook]'
jupyter notebook notebooks/GLNO_Shapes_Colab.ipynb
```

Se conserva únicamente `notebooks/GLNO_Shapes_Colab.ipynb`, el notebook autónomo utilizado
para las ejecuciones de múltiples formas. Incluye celdas que escriben los módulos Python
para ejecutarse en Colab sin clonar el repositorio. El código separado también está en `src/`.
`QUICK_RUN=True` requiere el dataset pequeño y realiza 2 épocas de entrenamiento.

## Datos del experimento

El experimento de múltiples formas utiliza esferas y toros deformados, con área normalizada a $4\pi$.
Las particiones se hacen por geometría. Cada forma tiene 5 paquetes gaussianos; las formas de test
incluyen otros 5 paquetes OOD por forma.

| Dataset | Formas train / val / test | Paquetes train / val / test | Paquetes OOD adicionales |
|---|---|---|---|
| `shapes` | 160 / 20 / 20 | 800 / 100 / 100 | 100 |
| `shapes_small` | 4 / 2 / 4 | 20 / 10 / 20 | 20 |

Los parámetros se copiaron de los `config.json` de los datasets locales a
[`configs/shapes.json`](configs/shapes.json) y [`configs/shapes_small.json`](configs/shapes_small.json).
Estos archivos contienen únicamente los campos que acepta `--config`.

| Parámetro | Valor guardado |
|---|---|
| Semilla | 0 |
| Tiempo final | 0.5 |
| Paso de Crank–Nicolson | 0.001 |
| Estados guardados | 11, incluido el inicial |
| Estados predichos por GLNO | 10, desde 0.05 hasta 0.5 |
| Modos espectrales | 700 |
| $k_0$ / $\sigma$ | [1, 6] / [0.15, 0.20] |
| $k_0$ / $\sigma$ OOD | [6, 8] / [0.18, 0.20] |

Para generar los datos, desde la raíz del repositorio instalado:

```bash
python scripts/generate_shapes_dataset.py --config configs/shapes_small.json --data-dir data/shapes_small
python scripts/generate_shapes_dataset.py --config configs/shapes.json --data-dir data/shapes
```

La generación requiere tiempo y espacio en disco. Las identidades esperadas por el notebook son
`8bab6ef70bbe73f4…` para `shapes_small` y `edba875cbb0ce818…` para `shapes`.
Usar la misma configuración no garantiza los mismos bytes con otras versiones de las bibliotecas
numéricas; un dataset regenerado requiere revisar identidades y huellas.

## Reproducir el experimento en Colab

El repositorio incluye el código para generar los datos CN; los datasets grandes no se
incluyen en GitHub. El recorrido es: generar en el computador, comprobar, empaquetar,
subir a Drive y ejecutar el notebook con GPU. Empiece con `shapes_small`.

### 1. Generar los datos

Después de instalar el proyecto, ejecute los comandos de la sección «Datos del experimento».
`configs/shapes_small.json` genera la prueba pequeña y `configs/shapes.json` el experimento
completo. `scripts/generate_shapes_dataset.py` construye las geometrías y resuelve CN usando
los módulos de `src/schrodinger_glno/`.

### 2. Comprobar datos y preparar las huellas

Desde la raíz del repositorio, con el entorno activado:

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/check_glno_shapes_data.py --data data/shapes_small --verify
python scripts/check_glno_shapes_data.py --data data/shapes --verify
```

Ejecute la segunda línea de comprobación solo cuando haya generado el dataset completo.
Cada comprobación debe terminar sin problemas; entonces escribe
`artifacts/glno_shapes_fingerprint_<dataset>.json`.

**Si está usando datos regenerados**, lea el campo `identity` de cada nueva huella y
actualice su entrada en `IDENTITIES` en la celda de configuración del notebook.
Reemplace también el JSON de la celda `%%writefile reference/glno_shapes_fingerprint_<dataset>.json`
por el contenido de `artifacts/glno_shapes_fingerprint_<dataset>.json`. Actualice la copia
correspondiente en `reference/` para mantener el registro del dataset utilizado.
Hágalo solo para los datasets que haya regenerado y cuya comprobación haya terminado sin problemas.

El notebook lleva sus propias copias de código y huellas: cambiar un archivo en `src/`
o `reference/` no cambia automáticamente esas celdas. Si modifica código, actualice también
su celda `%%writefile` correspondiente antes de ejecutar. Los scripts de construcción del
notebook no forman parte del entregable.

### 3. Empaquetar para Drive

```bash
python scripts/pack_shapes_dataset.py --data data/shapes_small --train-parts 1 --roundtrip --zip
python scripts/pack_shapes_dataset.py --data data/shapes --train-parts 4 --roundtrip --zip
```

Se crean `data_packed/glno_shapes_small.zip` y `data_packed/glno_shapes.zip`.
`--roundtrip` extrae una copia temporal para verificar la integridad y requiere espacio adicional.
Para probar el flujo solo hace falta el primer ZIP.

### 4. Subir a Google Drive

Suba los ZIP directamente a **Mi unidad**, manteniendo estos nombres:

- `glno_shapes_small.zip`: datos de la prueba rápida.
- `glno_shapes.zip`: datos del experimento completo.

Son ZIP de **datos**, distintos del ZIP de resultados de Colab incluido en este entregable.
El notebook los encuentra en Drive, verifica sus hashes y extrae los datos en el disco
local de la sesión. Guarda los checkpoints en `Mi unidad/glno_schrodinger/runs/<RUN_NAME>`.

### 5. Ejecutar el notebook

Suba `notebooks/GLNO_Shapes_Colab.ipynb` a Colab, seleccione un entorno con GPU y ejecute
las celdas en orden. No requiere `REPO_URL` ni clonar GitHub. Autorice el montaje de Drive
cuando se solicite. El notebook crea los archivos Python y luego carga los ZIP desde Drive.

| Objetivo | Configuración |
|---|---|
| Comprobar el flujo con entrenamiento breve | `QUICK_RUN=True`: `shapes_small`, 2 épocas |
| Repetir la ejecución periódica | `QUICK_RUN=False`, `VARIANT="periodic"`, `EPOCHS=300` |
| Repetir la ejecución completa | `QUICK_RUN=False`, `VARIANT="full"`, `EPOCHS=300` |

Use una carpeta de ejecución nueva para entrenar desde cero. Si ya existe un checkpoint,
el notebook intenta continuar la ejecución. Puede cambiar `DRIVE_ROOT` para aislar toda
la reproducción de las ejecuciones anteriores; así también evita que `QUICK_RUN` reutilice
su nombre automático `quick_periodic`. `EVALUATE_TEST=False` evita evaluar test durante la
selección del modelo; cámbielo a `True` cuando termine esa selección.

### Alcance de la reproducción

Regenerar datos con las configuraciones incluidas permite repetir el procedimiento, pero
no garantiza obtener los mismos archivos ni idénticas métricas con otras versiones de
NumPy, SciPy, BLAS, PyTorch o GPU. Para evaluar **los checkpoints originales**, se necesita
el dataset original con la identidad y la base espectral que esos checkpoints esperan.
Cambiar `IDENTITIES` no convierte un checkpoint antiguo en compatible con datos nuevos.
Los datos regenerados con otra identidad deben usarse para una ejecución nueva.

## Ejecuciones incluidas y resultados

`periodic` usa únicamente la rama periódica; `full` incorpora la rama aperiódica con
`gaussian_factor=50`. Los valores siguientes proceden de los `history.json` y `validation.json`
en [`results/glno_schrodinger/runs/`](results/glno_schrodinger/runs/).

| Ejecución | Dataset | Épocas registradas | Época de best.pt | Error validación select |
|---|---|---|---|---|
| `quick_periodic` | shapes_small | 2 | 2 | 0.89702228 |
| `shapes_glno_periodic_v1` | shapes | 300 | 283 | 0.00528561 |
| `shapes_glno_full_v1` | shapes | 300 | 293 | 0.02733158 |

La métrica es el error relativo $L^2$ ponderado por la masa. `select` excluye los toros de revolución:
la validación principal contiene 85 paquetes y reporta los 15 restantes por separado en `round`.
La prueba rápida sirve para comprobar el flujo y no representa el resultado científico.

El `test.json` incluido corresponde **solo a `shapes_glno_periodic_v1`**, con `best.pt` de la época 283:

| Evaluación | Error total select | Paquetes select | Paquetes round, reportados aparte |
|---|---|---|---|
| Test | 0.00469780 | 90 | 10 |
| OOD | 0.01724807 | 90 | 10 |

No se incluye un `test.json` de la variante `full`; no se atribuye a ella un resultado de test.
El notebook comienza con `QUICK_RUN=True` (2 épocas). Con `QUICK_RUN=False`, su valor inicial
`EPOCHS=50` corresponde al piloto, **no** a la duración de las ejecuciones guardadas, que llegaron a 300 épocas.

## Estructura

| Carpeta | Contenido |
|---|---|
| `src/schrodinger_glno/` | Geometría, solver, datos y modelo GLNO |
| `scripts/` | Generación, comprobación, empaquetado y evaluación |
| `notebooks/` | Único notebook de Colab usado para múltiples formas |
| `tests/` | Pruebas automáticas |
| `reference/` | Huellas de las bases espectrales |
| `configs/` | Configuraciones de los datasets utilizados |
| `results/glno_schrodinger/runs/` | Ejecuciones de Colab, métricas, gráficas y checkpoints |

Los resultados provienen de `glno_schrodinger-20261117T192006Z-1-001.zip`.
Los datasets grandes no se incluyen. Su generación y preparación en Drive se explican
arriba, en «Reproducir el experimento en Colab».

## Scripts incluidos

| Script | Uso |
|---|---|
| `generate_shapes_dataset.py` | Generar las formas y soluciones CN |
| `validate_shapes_dataset.py` | Comprobar la calidad del dataset |
| `check_glno_shapes_data.py` | Verificar datos y calcular las huellas espectrales |
| `pack_shapes_dataset.py` | Empaquetar datos para Drive |
| `visualize_shapes_dataset.py` | Visualizar las soluciones CN |
| `compare_glno_shapes.py` | Evaluar checkpoints sobre test y OOD |
| `generate_meshes_dataset.py` | Preparar datos de Bunny y Spot |
| `compare_glno_meshes.py` | Evaluar checkpoints en Bunny y Spot |

Los scripts de mallas se conservan porque hay resultados de Bunny y Spot en el entregable.
`generate_meshes_dataset.py` requiere el extra `meshes` y las mallas de entrada indicadas en su ayuda.

## Verificación sin entrenar

```bash
python -m pytest -q tests/test_notebook_delivery.py
```

Esta prueba comprueba la sintaxis del notebook y las importaciones de sus módulos, sin
entrenar. El resto de la suite puede incluir pruebas con entrenamiento pequeño.

## Procedencia

Las capas GLNO adaptan código de
https://github.com/JohnJoeZhu/Geometric-Neural-Operators-via-Aperiodic-Laplace-Learning-GLNO-
(commit `ca0fbb2`). La procedencia y modificaciones están documentadas en los archivos fuente.
Estos indican que el código original no tiene licencia y que la copia se preparó para trabajo universitario privado;
no se añade una licencia que conceda derechos sobre ese código.

## Figuras y resultados existentes

Las figuras se copiaron sin modificar los originales de `artifacts/` del proyecto
`quantum-tunneling`. No se recalcularon predicciones ni métricas para esta colección.

| Carpeta en `results/` | Contenido |
|---|---|
| `figures/cn/` | Muestras de las soluciones CN en esfera, toro y datasets de múltiples formas |
| `figures/geometry/` | Ejemplos de las geometrías generadas |
| `figures/comparisons/shapes/test/` | Cuatro paneles CN, GLNO y error en superficies de test |
| `figures/comparisons/shapes/ood/` | Un panel comparativo OOD |
| `figures/comparisons/meshes/test/` | Cuatro paneles comparativos en Bunny y Spot |
| `figures/errors/shapes/` | Gráfica agregada de errores en múltiples formas |
| `figures/errors/meshes/` | Gráfica agregada de errores en Bunny y Spot |
| `figures/diagnostics/` | Diagnósticos CN y del operador GLNO existentes |
| `figures/training/<run>/` | Copias exactas de las tres curvas originales de Colab |
| `metrics/shapes/`, `metrics/meshes/` | Resúmenes JSON originales de las evaluaciones locales |

Los paneles comparativos se conservan completos: incluyen la referencia CN, la predicción
GLNO de la rama periódica y su error espacial. No se recortaron ni se crearon imágenes
separadas de predicciones. Los resúmenes numéricos incluyen ambas variantes del modelo.

[`results/existing_figures_manifest.json`](results/existing_figures_manifest.json) registra
la ruta de origen y el SHA-256 de cada archivo copiado desde `artifacts/`.

Los 26 archivos del ZIP `glno_schrodinger-20261117T192006Z-1-001.zip` permanecen intactos
en `results/glno_schrodinger/runs/`. Sus métricas originales se distinguen de los resúmenes
de evaluación local en `results/metrics/`.
