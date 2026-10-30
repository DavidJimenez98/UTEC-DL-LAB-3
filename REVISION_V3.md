# Revisión de v3 para el Laboratorio 3

## Dictamen y fuentes

El código revisado cubre el protocolo del laboratorio y conserva MLCOR y CEM como propuesta central. Debe presentarse como una implementación adaptada, con diferencias explícitas respecto al artículo. La calidad final solo podrá juzgarse tras ejecutar v3 y revisar sus métricas y ejemplos.

Fuentes: [enunciado](Instruccciones/Laboratorio%203.docx) y [artículo original](https://www.nature.com/articles/s41598-026-56072-w). Esta revisión no sustituye la presentación y defensa individual del trabajo.

## Correspondencia con el enunciado

| Requisito | Evidencia en el código revisado |
|---|---|
| Entrenamiento sin identidad de persona | `train` recibe únicamente training.h5; el dataset devuelve imágenes, cámara e índice, sin leer person_ids |
| Implementación propia de la propuesta | MLCOR, GEL, C-DTL y actualización de memorias se implementan en reid_lab.py; se usan ResNet-50 y DBSCAN como componentes comunes |
| Base y ablación | baseline sin CEM/MLCOR; cem_only sin MLCOR; full con ambos |
| Configuración elegida por validación | Se maximiza mAP original entre rondas elegibles de validación; testing no interviene en la selección |
| Comparación con igual presupuesto nominal | Se alternan los métodos y solo se comparan rondas terminadas y validadas por los tres; los pasos reales por ronda también se registran |
| Dos condiciones, mismo modelo y galería | evaluate extrae una galería y evalúa las consultas originales y reducidas con el mismo checkpoint |
| L2 y coseno | Las tres ramas se normalizan; la concatenación ponderada final de 6144 dimensiones se normaliza antes del producto escalar |
| Métricas | mAP, Rank-1/5/10 y diferencia absoluta de mAP; se excluyen entradas junk y coincidencias persona/cámara |
| Cinco aciertos y cinco fallos | Se generan figura Top-10 y JSON; el reporte explica rangos. El análisis de causas visuales requiere revisar la figura del nuevo experimento |
| Código, instrucciones y resultados | README, scripts, manifiesto de configuración, logs y reportes Markdown |

La prueba de preflight abre testing.h5 únicamente para comprobar nombres de claves y tamaños. No lee las identidades ni calcula métricas para configurar el entrenamiento. La evaluación final usa el checkpoint elegido y queda registrada con su hash; una ejecución posterior reutiliza la prueba ya terminada.

## Cambios necesarios encontrados

1. **Peso de la pérdida base:** la ecuación 27 con m2=0.2 da 0.8 CE_global + 0.2 CE_superior + 0.2 CE_inferior. v2 usaba 0.6 para la global. v3 corrige ese coeficiente.
2. **Aprendizaje:** v3 añade un descenso de tasa cada 20 épocas acumuladas, con factor 0.1 elegido para esta adaptación. Las épocas se cuentan a través de las rondas y reanudaciones.
3. **Memorias:** v3 actualiza tanto instancias como centroides con momentum 0.1. Las entradas repetidas del muestreo PK se promedian antes de actualizar para evitar escrituras ambiguas.
4. **C-DTL:** se excluyen anclas sin positivo o negativo válido y se calculan las pérdidas en float32. La ecuación 20 ya define una log-probabilidad: se implementa -P_an * Q_an, evitando tomar otro logaritmo de un número negativo en la notación ambigua de la ecuación 24. Se conserva el objetivo operativo de esas ecuaciones; su texto no basta para garantizar una dirección de gradiente concreta ni una mejora empírica.
5. **Tiempo:** el ejecutor anterior comprobaba el presupuesto solo antes de una ronda. El nuevo controla los subprocesos y sus descendientes también durante una ronda, cuenta preflight y evaluación, y reserva tiempo para seleccionar y evaluar. Conserva las rondas completas; una ronda cortada se reinicia si se reanuda.
6. **Validez de la comparación:** se alternan baseline, cem_only y full; la selección usa la intersección de rondas validadas. Un método que terminó una ronda extra no obtiene una ventaja de selección.
7. **Trazabilidad:** se guardan configuración, hashes del código, semilla, pérdida, pasos, tiempo, memoria GPU, algoritmo de clustering y cantidad de outliers recuperados. Se rechaza mezclar checkpoints de protocolos distintos.

## Adaptaciones que deben explicarse

- **Clustering:** coseno fusionado y respaldo K-means esférico con hasta 300 grupos; las asignaciones del 15% inferior de confianza por grupo se excluyen cuando hay al menos cinco miembros. Esto sustituye Jaccard y puede introducir grupos artificiales. Los valores de eps no son intercambiables entre ambas distancias.
- **MLCOR:** conserva vecinos recíprocos, confianza gaussiana, niveles y presupuesto de recuperación, pero trabaja sobre la distancia coseno adaptada. Usar los mismos umbrales en otra escala no asegura la misma calibración ni el mismo efecto.
- **Representación y optimización:** se usa concatenación ponderada global/local para ranking, BatchNorm global con estadísticas congeladas, clasificador coseno con escala 16 y tasa del backbone diez veces menor que la de los clasificadores. Los optimizadores y pseudo-clasificadores se reinicializan al reclusterizar. Son decisiones de esta implementación.
- **Entrenamiento:** v2 ejecutó 20 épocas por método; v3 propone hasta 64, sujetas a cinco horas. El artículo usa 50 con 200 iteraciones por época. Aquí el número de pasos depende de las imágenes asignadas y el batch; los presupuestos no son equivalentes.
- **Memoria:** GEL almacena las instancias con pseudoetiqueta válida, sin asignar una identidad inventada a outliers no recuperados. Los centroides se actualizan con la media de los ejemplos de cada pseudoidentidad en el batch.

Por estas diferencias no se deben presentar las cifras como una reproducción exacta del artículo. Tampoco puede atribuirse una diferencia v2/v3 únicamente al hardware o al número de épocas: cambian batch, pérdida, tasa y actualización de memorias.

## Interpretación y límites

Una menor mAP de full frente a cem_only no demuestra que MLCOR haya asignado identidades incorrectas. Es una hipótesis compatible con pseudoetiquetas ruidosas y otras diferencias de optimización. No se utilizarán etiquetas reales de entrenamiento para validar esa hipótesis.

Cada experimento usa una semilla y solo 100 consultas de validación. Se reportarán diferencias descriptivas, sin afirmar significancia estadística o causalidad. Las métricas de testing de v2 ya se observaron; v3 es una ampliación exploratoria, cuyo ajuste y selección se harán con validación. No se elegirá una configuración por su resultado de testing.

## Archivos y ejecución

Copiar actualizados a D:\DAVID\lab3-reid: `reid_lab.py`, `workstation_v3.py`, `run_workstation_v3.ps1` y `khipu/select_and_test.py`, conservando la subcarpeta. Los datos y el entorno existentes sirven.

```cmd
powershell -ExecutionPolicy Bypass -File .\run_workstation_v3.ps1
```

Para separar primero la selección por validación de la prueba final:

```cmd
powershell -ExecutionPolicy Bypass -File .\run_workstation_v3.ps1 -ValidationOnly
```

Si ya existen checkpoints de la versión anterior de v3, ejecutar con `-Output runs_workstation_v3_reviewed`. Los datos antiguos no se borran. Una vez completada la prueba final, el experimento se congela. No se requiere actualizar driver, Python ni paquetes.

## Verificación

Se verifican sintaxis Python y PowerShell, equivalencia de GEL con su agregación explícita, gradientes finitos de C-DTL, actualización de entradas duplicadas, selección de rondas comunes, protección de configuración y cancelación real de subprocesos por plazo. Una prueba con 80 imágenes reales sin person_ids recorre full, checkpoint y reanudación. Estas pruebas no anticipan la mAP ni comprueban el batch 64 en la RTX A4000 remota; eso corresponde a la ejecución de la workstation.
