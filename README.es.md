# BEV-UrbanMap: Lift-Splat-Shoot desde cero

🇬🇧 [Read in English](README.en.md)

Percepción en vista de pájaro (BEV, *bird's-eye view*) para conducción autónoma a partir de las
6 cámaras de un coche. Implementamos la arquitectura **Lift-Splat-Shoot** (Philion y Fidler,
ECCV 2020) en PyTorch y la entrenamos con **nuScenes** para segmentar vehículos en un mapa de
100 m × 100 m alrededor del coche.

Práctica final de *Inteligencia Artificial en las Organizaciones* (UC3M, 2026/27), grupo 14:
Sergio González Jiménez, Alfonso Gabriel Pineda Fernández, Daniel López-Antona Pesquera,
Edgar Antero Sánchez Elizondo.

## Estado (27 de noviembre de 2026)

| Parte | Estado |
|---|---|
| **Lift**: distribución de profundidad por píxel × contexto (producto exterior) + geometría del frustum | ✅ hecho y probado |
| **Splat**: *sum pooling* en pilares con el truco cumsum y su gradiente analítico | ✅ hecho y probado |
| **Codificador BEV** (bloques ResNet-18, sección 4.1 del paper) | ✅ hecho y probado |
| Etiquetas BEV de vehículos a partir de las cajas 3D de nuScenes | ✅ hecho y probado |
| Dataset con las divisiones oficiales y aumento de datos | ✅ hecho y probado |
| Entrenamiento y evaluación (IoU, precisión, recall, accuracy, IoU por distancia) | ✅ hecho |
| Líneas base del paper: CNN sin 3D y encoder congelado | ✅ hechas y entrenadas |
| **Shoot**: planificación de trayectorias con trayectorias plantilla | ⏳ pendiente (previsto para la entrega final) |
| Segmentación de la zona transitable | ⏳ pendiente |
| Entrenamiento con (parte de) nuScenes trainval | ⏳ pendiente, por ahora solo nuScenes mini |

## Cómo funciona

```
6 imágenes ──► CamEncode (Lift) ───────► qué hay en cada punto 3D    (B, 6, 41, 64, 8, 22)
               EfficientNet-B0 + cabeza de profundidad/contexto
calibración ──► geometría del frustum ──► dónde está cada punto      (B, 6, 41, 8, 22, 3)
                   K⁻¹, R, t
                         │
                         ▼
                 Splat (sum pooling) ───► mapa BEV de características (B, 64, 200, 200)
                         ▼
                 Codificador BEV ───────► logit de vehículo por casilla (B, 1, 200, 200)
```

- **Lift** (`models/cam_encode.py`, `models/geometry.py`). Para cada celda del mapa de
  características de la imagen, la red predice una probabilidad sobre 41 profundidades (4–44 m)
  y un vector de contexto de 64 valores; su producto exterior coloca el contexto a lo largo del
  rayo de la cámara. La calibración de la cámara da la posición (x, y, z) de cada uno de esos
  puntos alrededor del coche.
- **Splat** (`models/splat.py`). Cada punto cae en un pilar de 0,5 m × 0,5 m de una cuadrícula
  de 200 × 200; los puntos del mismo pilar se suman (el "truco cumsum" del paper).
- **Codificador BEV** (`models/bev_encode.py`). Rellena los huecos entre los rayos de las cámaras
  y predice el mapa.

**Fine-tuning frente a desarrollo propio.** El extractor de características de imagen
(EfficientNet-B0) parte de los pesos de ImageNet y se ajusta (*fine-tuning*). Todo lo demás
(neck, cabeza de profundidad/contexto, geometría del frustum, Splat, codificador BEV, etiquetas,
dataset y entrenamiento) es código propio y se entrena desde cero. El repositorio oficial de LSS
solo se ha usado como referencia.

## Estructura del proyecto

```
src/lss_bev/
  models/cam_encode.py    Lift: backbone de imagen + cabeza de profundidad/contexto + producto exterior
  models/geometry.py      Lift: rejilla del frustum, ajuste de intrínsecos, cámara -> coche
  models/splat.py         Splat: sum pooling en pilares (truco cumsum)
  models/bev_encode.py    Codificador BEV
  models/lift_splat.py    Modelo completo (Lift + Splat + codificador BEV), backbone congelable
  models/baseline.py      Línea base CNN sin estructura 3D
  models/build.py         Crear un modelo por nombre / cargar un checkpoint
  labels.py               Mapas BEV de vehículos reales a partir de las anotaciones de nuScenes
  dataset.py              Dataset de PyTorch (imágenes, calibración, etiqueta) + aumento de datos
  metrics.py              IoU, precisión, recall, accuracy, IoU por distancia
  nusc_calib.py           Lectura de la calibración de las cámaras desde los JSON de nuScenes
scripts/                  entrenamiento, evaluación y visualizaciones
tests/                    89 tests (pytest)
```

## Instalación

Requiere [uv](https://docs.astral.sh/uv/) y Python 3.12 (uv lo instala si no está).

```bash
uv sync --extra dev
```

En **Windows con GPU NVIDIA**, PyPI solo tiene versiones de PyTorch para CPU: instalad la versión
con CUDA (ver pytorch.org) o el entrenamiento se hará en la CPU sin avisar.

### Datos

Descargad **nuScenes mini** (v1.0) desde <https://www.nuscenes.org/nuscenes> y descomprimidlo en
`data/` para que quede así:

```
data/
  maps/  samples/  sweeps/  v1.0-mini/
```

Solo se usan las imágenes de las cámaras, la calibración y las anotaciones 3D (ni LiDAR ni radar).
Otras versiones (por ejemplo `v1.0-trainval`, incluso descargada en parte) funcionan con
`--version`; las muestras a las que les falte alguna imagen se omiten.

## Uso

Todos los comandos, desde la raíz del proyecto.

```bash
# Tests (~45 s)
uv run pytest tests -q

# Comprobaciones visuales
uv run python scripts/visualize_frustum.py       # puntos del frustum de las 6 cámaras, vistos desde arriba
uv run python scripts/visualize_splat.py         # qué cámara rellena cada casilla del mapa BEV
uv run python scripts/visualize_labels.py 0      # vehículos reales de una muestra (0-403)

# Prueba de sobreajuste: memorizar 4 muestras (el IoU debería superar ~0,8)
uv run python scripts/train.py --overfit 4 --epochs 150 --val-every 25 --name overfit

# Entrenamiento con nuScenes mini (~25 min cada uno en un Apple M4)
uv run python scripts/train.py --epochs 30 --name mini-lss
uv run python scripts/train.py --epochs 30 --model cnn --name mini-cnn
uv run python scripts/train.py --epochs 30 --model frozen --name mini-frozen

# Tabla de resultados, predicciones y curvas de entrenamiento
uv run python scripts/evaluate.py runs/mini-lss/best.pt runs/mini-cnn/best.pt runs/mini-frozen/best.pt
uv run python scripts/visualize_predictions.py runs/mini-lss/best.pt runs/mini-cnn/best.pt runs/mini-frozen/best.pt
uv run python scripts/plot_training.py runs/mini-lss runs/mini-frozen runs/mini-cnn
```

El script de entrenamiento usa automáticamente la GPU de Apple (MPS), una GPU NVIDIA (CUDA) o la
CPU. Cada ejecución guarda `runs/<nombre>/log.csv`, `last.pt` y `best.pt` (mejor IoU de
validación). Las figuras y tablas van a `outputs/`. Ninguna de las dos carpetas está en git.

## Resultados (nuScenes mini)

Segmentación de vehículos en la división oficial `mini_val` (81 muestras, 2 escenas), mejor
época de un entrenamiento de 30 épocas (una semilla), lotes de 4, Adam 1e-3, BCE con peso
positivo 1,0 (paper, sección 5).

| Modelo | IoU | IoU paper | Precisión | Recall | Accuracy | IoU 0–10 m | 10–20 m | 20–30 m | 30–50 m |
|---|---|---|---|---|---|---|---|---|---|
| CNN (sin 3D) | 0,16 | 24,25 | 16,2 | 0,2 | 97,34 | 1,4 | 0,3 | 0,0 | 0,0 |
| Encoder congelado | 9,25 | 26,83 | 37,0 | 11,0 | 97,15 | 37,9 | 17,4 | 2,7 | 0,7 |
| **Lift-Splat (nuestro)** | **9,74** | 32,07 | 42,5 | 11,2 | 97,25 | 38,8 | 18,9 | 2,0 | 0,7 |

Valores en %. Valores del paper: Tabla 1, nuScenes, "Vehicles".

- **Lo que importa es la estructura 3D.** Sin Lift ni Splat, la CNN casi no aprende con tan
  pocos datos, mientras que los dos modelos 3D colocan bien los vehículos cercanos.
- **La accuracy engaña**: el 97,36 % de las casillas están vacías, así que un modelo que nunca
  predice un vehículo alcanza un 97,36 % de accuracy, más que cualquiera de los modelos
  entrenados. El IoU es la métrica del paper.
- **Lift-Splat frente al encoder congelado**: con una sola semilla, la diferencia no es
  concluyente (el IoU de validación varía 2–3 puntos entre épocas).
- **Distancia**: funciona bien hasta 20 m y mal más allá (estimar la profundidad con una sola
  cámara es más difícil cuanto más lejos).
- **Distancia con el paper**: usamos ~1 % de sus datos (323 frente a ~28.000 muestras de
  entrenamiento) y 2.400 frente a 300.000 pasos; los modelos sobreajustan a partir de la época 8.

## Notas de implementación

- Características de imagen con stride 16 (8 × 22 celdas por imagen), como en el código oficial.
- Las casillas se calculan con `floor`, así que los puntos justo fuera del mapa se descartan
  (el código oficial usa `.long()`, que redondea hacia cero y los mete en las casillas del borde).
- Aumento de datos: escalado, recorte y volteo horizontal aleatorios; los intrínsecos cambian
  junto con la imagen, así que las etiquetas nunca cambian. La rotación de ±5,4° del código
  oficial no está implementada.
- Las etiquetas incluyen todos los vehículos anotados, sea cual sea su visibilidad, como el paper.

## Trabajo pendiente (hasta el 13 de diciembre)

1. **Shoot**: trayectorias plantilla a partir de las posiciones del coche (K-Means), cabeza de
   mapa de coste, la función de error de la ecuación 2 y la precisión top-k.
2. Segmentación de la **zona transitable** a partir de los mapas de nuScenes.
3. Más datos (parte de `v1.0-trainval`) y varias semillas por modelo.

## Referencias

1. J. Philion, S. Fidler. *Lift, Splat, Shoot: Encoding Images from Arbitrary Camera Rigs by
   Implicitly Unprojecting to 3D.* ECCV 2020.
2. H. Caesar et al. *nuScenes: A multimodal dataset for autonomous driving.* CVPR 2020.
3. K. He et al. *Deep residual learning for image recognition.* CVPR 2016.
4. M. Tan, Q. Le. *EfficientNet: Rethinking model scaling for convolutional neural networks.* ICML 2019.
