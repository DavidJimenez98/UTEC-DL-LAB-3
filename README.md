# Laboratorio 3: código fuente y resultados V3

**Modelo seleccionado:** `cem_only`, `round_04` (quinta ronda, 40 épocas acumuladas). Método adaptado de Zhang et al., *Confidence-guided outlier refinement and collaborative embedding for unsupervised person re-identification*, Scientific Reports (2026), DOI: https://doi.org/10.1038/s41598-026-56072-w.

## Contenido

- `reid_lab.py`: carga HDF5, transformaciones, ResNet-50/GeM, clustering, MLCOR, GEL, C-DTL, entrenamiento y evaluación.
- `workstation_v3.py`: compara baseline, cem_only y full, valida cada ronda y selecciona por validación. En Windows requiere GPU CUDA para el entrenamiento automatizado.
- `khipu/select_and_test.py`: selector local y reporte. La carpeta es histórica; no requiere Khipu ni Slurm.
- `reproducir_entrega.py`: comandos portables para revisar resultados, validar datos o reproducir evaluación/entrenamiento.
- `modelo/cem_only_round_04.pt`: checkpoint seleccionado, sin imágenes de entrenamiento.
- `resultados/`: métricas de los tres métodos en validación, test del ganador, configuración, hashes y ejemplos visuales.
- `INFORME_RESULTADOS.md`, `resultados/ANALISIS_ERRORES.md`: resultados y cinco aciertos/cinco fallos explicados.
- `tests/`: pruebas de las pérdidas, memorias, selección, orquestador y límite de tiempo. La prueba real de humo necesita los HDF5.

No se incluyen datasets, entornos Conda, caches, ejecuciones antiguas ni todos los checkpoints intermedios. Los JSON históricos conservan las rutas de la workstation como evidencia; para ejecutar use los comandos relativos siguientes.

## Revisar sin instalar PyTorch

Con Python 3.11 o superior, desde esta carpeta:

```powershell
python reproducir_entrega.py reporte
```

También puede abrir los informes Markdown y las figuras sin ejecutar entrenamiento.

## Preparar un entorno aislado

Desde una terminal con Conda/Miniforge, en la carpeta de la entrega:

```powershell
conda create --prefix .conda-reid python=3.11 -y
conda activate .\.conda-reid
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-entrega.txt
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Python 3.11 y PyTorch 2.7.1+cu118 corresponden al entorno registrado del experimento. Las restricciones de las demás dependencias son compatibles, no un `pip freeze` extraído de la workstation. No hace falta instalar un toolkit CUDA aparte para usar estas ruedas con un controlador compatible. Para evaluación sin GPU puede instalar las ruedas CPU de la misma versión desde el índice oficial CPU; será más lenta. El orquestador de entrenamiento exige CUDA.

## Ubicación de datos

Ubique `training.h5`, `validation.h5` y `testing.h5` oficiales en `Datos preparados/Market-1501/`, o pase `--data` con su directorio. No redivida ni modifique las particiones. Las consultas LR ya están en los archivos.

```powershell
python reproducir_entrega.py preflight --data "Datos preparados/Market-1501"
python reproducir_entrega.py evaluar --data "Datos preparados/Market-1501" --workers 0
```

La segunda orden utiliza el checkpoint incluido y escribe en `reproduccion_test/`, preservando las métricas entregadas. Es una reproducción del resultado congelado, no una nueva selección. No utilice sus resultados para ajustar parámetros y seguir llamando ciego al test. Pequeñas diferencias numéricas pueden aparecer entre entornos/hardware.

## Reentrenar baseline, ablación y full

```powershell
python reproducir_entrega.py entrenar --data "Datos preparados/Market-1501" --workers 8 --output runs_reproduccion
```

Entrena 8 rondas de 8 épocas por método, batch 64, extracción 256, semilla 42 y presupuesto máximo de 5 horas por invocación. Esta orden termina en validación. Para un experimento nuevo, una vez fijada la configuración por validación:

```powershell
python khipu/select_and_test.py --runs runs_reproduccion --common-rounds --testing "Datos preparados/Market-1501/testing.h5" --batch 256 --workers 8 --once
```

`--once` registra el hash del checkpoint y evita sustituir la evaluación por otro ganador dentro de la misma ejecución. No impide por sí solo contaminación metodológica entre directorios. La protección requiere respetar el protocolo.

Para reanudar use la misma orden de entrenamiento, parámetros, código y directorio. Se repite una ronda incompleta. Se guarda modelo, ronda y eps, sin estado del optimizador entre rondas. No cambie las huellas del código ni la configuración a mitad de experimento. En CMD use cada orden en una línea: el acento grave multilínea pertenece a PowerShell.

## Protocolo y adaptación

Training entrega 11043 imágenes y no se usan identidades reales. Validación: 100 consultas × 1793 imágenes de galería. Test: 3368 × 19732. Las consultas originales (128×64) y LR (32×16) entran a la misma red redimensionadas a 256×128. La galería permanece original. Se fusionan vectores L2 y se ordenan con coseno. Se excluye junk y la misma identidad en la misma cámara, sin retirar otras personas de esa cámara.

La comparación con baseline y la ablación se hacen en validación. Cada método selecciona la mejor ronda y el ganador global usa mayor mAP HR. Solo ese ganador tiene test V3. No afirmamos una tabla test de todos los métodos.

| Componente | Paper | Adaptación V3 |
|---|---|---|
| Representación | ResNet-50, GeM, global/superior/inferior | Conservada; BN solo en global, L2 por rama |
| Distancia de agrupamiento | Jaccard fusionado | Coseno fusionado 0.70/0.15/0.15 |
| DBSCAN | eps 0.6 en Market | eps candidatos 0.30/0.25/0.20, criterios de rechazo |
| Respaldo | Protocolo DBSCAN | K-means esférico, hasta 300 grupos, descarte de 15% en grupos con al menos 5 miembros |
| MLCOR | Reciprocidad y niveles, s=5, σ=0.5, ρ=0.6, B=40 | Mecanismo conservado sobre distancia coseno; la calibración cambia |
| CEM | GEL + C-DTL, memorias | Implementación propia de agregación y objetivo suave batch-hard |
| Memoria | Diccionarios global/local | Instancias agrupadas y centroides por rama, actualización 0.1/0.9 |
| Entrenamiento | 50 épocas × 200 iteraciones | 8 rondas × 8 épocas de longitud variable; Adam y clasificadores nuevos por ronda |
| Datos | Split benchmark del artículo | Archivos preparados del laboratorio, menos imágenes de train |

La BN congela estadísticas, conserva parámetros afines entrenables y el backbone tiene LR diez veces menor. Las pérdidas V3 son LBASE=0.8 CEg+0.2 CEup+0.2 CElow y LFINAL=LBASE+LC-DTL+0.6 LGEL. La ecuación 20 del paper ya define log Q y la 24 vuelve a escribir log Q: se interpreta con una única log-probabilidad finita. El objetivo online de C-DTL depende de distancias positiva y negativa, por lo que no afirmamos una garantía de gradiente exclusivamente negativo. Estas decisiones forman parte de la adaptación, no de una reproducción exacta.

## Pruebas

```powershell
python -m unittest discover -s tests -p test_v3_review.py -v
```

Las ocho pruebas se verificaron durante la revisión del código original V3. La entrega conserva esos archivos. La calidad del modelo se informa por separado en las métricas, no se deduce de pasar pruebas de software.
