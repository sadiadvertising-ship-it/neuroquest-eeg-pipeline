# NeuroQuest EEG pipeline

Adquisición y clasificación de emociones (Happy, Neutral, Frustrated) con la diadema OpenBCI Ganglion (4 canales, 200 Hz). Es el componente de ingeniería del proyecto de grado NeuroQuest / NeuroUX y sigue el pipeline de la Figura Arq-5 de la tesis. Incluye un panel web local para ver las sesiones, el modelo y la señal en vivo.

## Pipeline

1. **Adquisición.** BrainFlow lee los canales EEG reales de la Ganglion. La fila 0 del paquete es el contador de paquetes y se guarda aparte para detectar paquetes perdidos.
2. **Filtrado.** Pasa-banda Butterworth 0,5-45 Hz de orden 4 más notch de 60 Hz (red eléctrica de Colombia) con Q = 30. El filtro es causal y corre sobre 3 s de señal previa, así no quedan transitorios en el borde más reciente de la ventana.
3. **Rechazo de artefactos.** La ventana se descarta si algún canal pasa de ±100 µV (parpadeo, movimiento) o si algún canal está plano o saturado (desviación menor a 0,5 µV).
4. **FFT.** Periodograma con ventana Hamming sobre ventanas de 2 s.
5. **Potencia por banda.** log10 de δ (1-4 Hz), θ (4-8), α (8-13), β (13-30) y γ (30-45) en Fp1, Fp2, T3 y T4: 20 features. El z-score lo hace el modelo.
6. **Modelo.** StandardScaler + SVM (o Random Forest).

Todo se ajusta en `config.py`.

## Instalación

Requiere Python 3.9-3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`setuptools` va fijado por debajo de 81 porque brainflow 5.12.1 todavía usa `pkg_resources`. En un Mac con Apple Silicon puede hacer falta otra versión de PyQt5; en un Mac Intel, la de `requirements.txt` funciona.

## Uso

```bash
# Grabar una sesión con el protocolo (con gráficas). El código de participante es obligatorio
python main.py --mode gui --participant P03

#   o en la terminal, sin gráficas
python main.py --mode record --participant P03

# Entrenar con las sesiones de data/sessions (por defecto pide al menos 20 sesiones útiles)
python main.py --mode train
python main.py --mode train --min-sessions 5 --permutations 100

# Clasificar en vivo
python main.py --mode realtime --participant P03

# Panel web local
python main.py --mode dashboard
```

- **Participante.** El código se escribe en la GUI o con `--participant` y solo acepta letras, números, `-` y `_` (por ejemplo P03). Nunca se usan nombres de personas.
- **Placa sintética.** `--synthetic` usa la placa sintética de BrainFlow para probar sin la diadema. Esas grabaciones van a `data/synthetic` y nunca se mezclan con las reales.
- **Puerto.** Se cambia con `--port`. Por defecto es `/dev/cu.usbmodem11`, el dongle BLED112 en macOS.

### Durante una grabación

1. Abre dos terminales. En una, `python main.py --mode dashboard`; en la otra, `python main.py --mode gui --participant P03`.
2. Antes de pulsar Start, revisa los electrodos en la GUI. La etiqueta "Signal" dice qué canal tiene un artefacto, está plano o tiene mucho más ruido de 60 Hz que los demás, lo que casi siempre es un electrodo con mal contacto. En la pestaña FFT, ese canal muestra un pico grande en 60 Hz. Ajusta hasta que los cuatro canales se vean bien. La GUI no guarda nada hasta Start, así que la página **En vivo** del panel muestra la señal desde ese momento.
3. Durante el protocolo, la GUI anuncia en el **Cue** (los últimos 3 s del reposo) qué emoción sigue, para tener listo el estímulo y presentarlo apenas empiece el bloque.
4. Si un canal se daña a mitad de la sesión, pulsa **Stop**: la sesión queda guardada como detenida y fuera del entrenamiento. Arregla el electrodo y empieza una sesión nueva.
5. Si deja de llegar señal, la GUI muestra "NO DATA" y el panel avisa cuántos segundos lleva sin muestras: revisa la batería de la diadema y el dongle.

### Protocolo

Un Cue de 3 s, que también deja estabilizar los filtros, y luego, en cada ronda (3 por defecto), cada emoción durante 15 s, separadas por 8 s de reposo y 3 s de Cue. El orden de las emociones cambia al azar en cada ronda (`SHUFFLE_ORDER`), para que la etiqueta no quede atada al momento de la sesión, por ejemplo al cansancio o al secado del gel. Los primeros 2 s de cada estímulo (`ONSET_MARGIN`) no entran al entrenamiento, porque el estímulo lo presenta una persona al ver la fase.

### Archivos

| Qué | Dónde | Formato |
|---|---|---|
| Sesiones con el protocolo | `data/sessions/session_<participante>_<fecha>.csv` | Señal cruda, una fila por muestra |
| Grabaciones de `realtime` | `data/live/live_<fecha>.csv` | Igual que las sesiones, con la etiqueta `Live` |
| Todo lo grabado con `--synthetic` | `data/synthetic/` | Igual |
| Predicciones en vivo (realtime y GUI) | `logs/<grabación>_predictions.csv` | Una fila por segundo: `ok`, `artifact` o `no_data` |
| Modelo y su evaluación | `models/<modelo>.pkl`, `.report.json`, `.cv.csv`, `.confusion_matrix.png` | |

**Formato de las grabaciones.**
- **Encabezado:** líneas `# clave=valor` con la frecuencia de muestreo, la placa, el participante, la hora de inicio y los pasos planeados del protocolo.
- **Columnas:** `timestamp,package,Fp1,Fp2,T3,T4,label,round,block`.
- **Cierre:** una línea final `# completed=...` si el protocolo terminó o `# status=stopped` si se detuvo. Sin esa línea, la sesión cuenta como incompleta (se cortó).
- **Etiquetas:** cada muestra lleva la fase que corresponde a su timestamp, no la del momento en que se leyó.
- **Features:** se calculan al entrenar, así que cambiar el pipeline no obliga a grabar de nuevo.

`data/`, `models/` y `logs/` están en `.gitignore`. No se deben subir datos de participantes al repositorio.

### Evaluación

- **Qué sesiones entran.** Solo las completas de `data/sessions`. Las detenidas, las incompletas, las sintéticas, las de formato viejo y los archivos vacíos quedan por fuera, con su motivo en la terminal y en el reporte. `--include-partial` incluye también las detenidas y las incompletas.
- **Grupos.** Las ventanas de un mismo participante, sesión o ronda se parecen entre sí, así que nunca quedan repartidas entre entrenamiento y prueba. Se usa la unidad más exigente disponible: participantes (con 3 o más), sesiones (con 3 o más) o rondas. Con rondas, el resultado solo vale dentro de una sesión ya vista.
- **Una sola regla.** La validación cruzada, la búsqueda de hiperparámetros y el modo en vivo deciden igual: la clase con mayor probabilidad. La confianza se recalibra con un escalado de temperatura ajustado en la validación agrupada, así que no promete más de lo que el modelo acierta.
- **Azar.** El azar se calcula con las clases presentes en cada pliegue y en cada sesión. Una sesión con una sola clase no tiene azar definido y no cuenta en esa comparación. Solo `--permutations N` muestra si el modelo supera al azar: repite toda la evaluación con las etiquetas permutadas por bloque y da un p-valor. El p más chico posible es 1/(N+1). Con 19 permutaciones llega justo a 0,05, y p ≤ 0,05 cuenta como superar al azar. Para la tesis conviene usar 100. Con 20 sesiones, 100 permutaciones tardan unos 45-60 min en un Mac de 4 núcleos.
- **Qué se guarda.** Todo queda en `models/<modelo>.report.json` y `models/<modelo>.cv.csv`, que el panel muestra.

## Panel

`python main.py --mode dashboard` abre una página en el navegador en `http://127.0.0.1:8501`. Solo lee `data/`, `logs/` y `models/` de este computador: nunca abre la diadema, escucha solo en este computador y tiene apagadas las estadísticas de uso de Streamlit.

- **Sesiones.** Cada grabación con su resumen (duración, muestras por segundo, paquetes perdidos y ventanas descartadas) y una línea de tiempo con las fases, lo que predijo el modelo en cada ventana, las ventanas descartadas y los 4 canales. Debajo, la calidad por canal y la duración real de cada bloque frente a la planeada.
- **En vivo.** Se actualiza cada segundo con lo que se está grabando. Muestra en qué bloque va el protocolo y qué emoción sigue, los últimos 10 s de señal, un aviso por canal, y en `realtime` las probabilidades de los últimos minutos.
- **Modelo.** La exactitud de cada sesión frente a su azar, la prueba de permutación, la matriz de confusión, la calibración, dónde falla (por segundo del bloque, por ronda y los peores bloques) y cuánto cambia la potencia de cada emoción frente al reposo.
- **Comparar.** La huella espectral y las ventanas descartadas por sesión, día o participante.

En Sesiones, Modelo y Comparar, **Preparar descarga** arma un HTML de la página que abre sin internet, para mandarlo al director o al jurado. Cada gráfica tiene además un botón de cámara para bajarla como PNG.

## Revisión con la diadema

Lo que falta comprobar con la Ganglion real, con el panel abierto:

1. **Montaje.** Al parpadear, se deben mover solo Fp1 y Fp2. Al tocar cada electrodo, debe moverse su canal.
2. **Muestras.** Graba una sesión completa. En Sesiones deben llegar cerca de 200 muestras por segundo, y la cuenta de paquetes perdidos debe tener sentido. La cabecera guarda el firmware que reporta BrainFlow (`# firmware=2` o `3`), y `analysis.lost_packets` cuenta según ese firmware: el 2 numera los paquetes 1-100 y 101-200 (el 0 es un paquete tras un reinicio) y el 3 los numera 0-99 y 100-199. Esa numeración se tomó del código de BrainFlow 5.12.1 y falta confirmarla con la diadema. BrainFlow 5.12.1 detecta solo el firmware; las versiones nuevas asumen el 3 si no se les dice nada, por eso la conexión pasa `other_info='fw:auto'`. Antes de actualizar BrainFlow, revisa que la cabecera siga diciendo el firmware correcto.
3. **Duración de los bloques.** Cada bloque debe durar lo planeado: 15 s de estímulo, 8 s de reposo y 3 s de Cue.
4. **Ventanas descartadas.** Revisa qué porcentaje se descarta. Si los parpadeos tumban más del 30 %, decide con los datos si subir `ARTIFACT_UV`, mover los electrodos o tratar el EOG. Los umbrales de ruido de 60 Hz del panel (`LINE_RATIO`, `LINE_MIN_UV` en `analysis.py`) también se ajustan con sesiones reales.
5. **Entrenamiento.** Con 3 o más sesiones, idealmente de 3 o más personas, entrena con `--permutations 100` y lee la página Modelo.
6. **Normalización.** Arq-5 dice "z-score" pero no si es global o por sesión. Con datos reales, compara el z-score global con uno por sesión (o restando la línea base del reposo) y quédate con el que generalice mejor.

## Pruebas

```bash
python test_pipeline.py    # pipeline, grabación, entrenamiento, tiempo real y GUI (sin pantalla)
python test_dashboard.py   # cada página del panel, la exportación y el arranque
```

**`test_pipeline.py`** genera sesiones sintéticas donde las emociones cambian la potencia alfa y beta, con ruido de 60 Hz y parpadeos. Comprueba:
- filtros, rechazo de artefactos y etiquetado por timestamp;
- qué sesiones quedan por fuera;
- que el modelo aprende cuando hay señal y da azar cuando no la hay;
- la regla de decisión única y la calibración;
- las permutaciones;
- lo que guarda el entrenamiento;
- el aviso cuando la diadema deja de mandar datos;
- la GUI completa en modo offscreen con la placa sintética.

**`test_dashboard.py`** corre cada página con y sin datos y con una grabación en curso. También revisa que el HTML exportado no dependa de internet y que el panel nunca importe brainflow.
