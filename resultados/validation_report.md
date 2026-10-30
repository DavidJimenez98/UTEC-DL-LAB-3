# Validacion del experimento

Seleccion exclusivamente por mAP original de validacion. Rondas elegibles: ['round_00', 'round_01', 'round_02', 'round_03', 'round_04', 'round_05', 'round_06', 'round_07']

| Metodo | Ronda (indice desde 0) | mAP | Rank-1 | Rank-5 | Rank-10 | mAP baja | Rank-1 baja | Rank-5 baja | Rank-10 baja | Caida mAP |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | round_07 | 0.2071 | 0.3600 | 0.4700 | 0.5400 | 0.0906 | 0.1200 | 0.2500 | 0.3500 | 0.1165 |
| cem_only | round_04 | 0.2893 | 0.4400 | 0.6700 | 0.7200 | 0.1178 | 0.2200 | 0.3200 | 0.4600 | 0.1715 |
| full | round_03 | 0.2606 | 0.4100 | 0.5500 | 0.6000 | 0.1125 | 0.1900 | 0.3700 | 0.4500 | 0.1481 |

Configuracion: hasta 8 rondas x 8 epocas, batch 64, protocolo paper_adapted_v3. Los resumenes por ronda contienen pasos de optimizacion, perdida, tiempo y memoria GPU.

## Comparacion exploratoria con v2

Cambian batch, duracion y detalles de optimizacion; las diferencias no se atribuyen solo a mas epocas.

| Metodo | mAP val v2 | mAP val actual | Diferencia |
|---|---:|---:|---:|
| baseline | 0.1885 | 0.2071 | +0.0186 |
| cem_only | 0.2114 | 0.2893 | +0.0779 |
| full | 0.1939 | 0.2606 | +0.0666 |

Una semilla y 100 consultas: diferencias descriptivas, sin prueba de significancia.
No se ha usado testing para elegir configuracion. La revision visual de errores se realiza tras la evaluacion final.
