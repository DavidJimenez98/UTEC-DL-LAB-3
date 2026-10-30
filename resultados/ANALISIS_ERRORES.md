# Cinco aciertos y cinco fallos de V3

La figura corresponde al checkpoint CEM `round_04`, consultas originales de testing. Se muestran los primeros cinco aciertos y primeros cinco fallos de Rank-1 al recorrer las consultas. No constituyen una muestra aleatoria. Borde verde significa misma identidad y rojo una identidad diferente, después del filtro Market.

![Consultas y diez candidatos](test_metrics_retrieval_examples.png)

## Aciertos

| Consulta (índice desde 0) | Identidad | Cámaras consulta / top-1 | Observación visual |
|---|---:|---|---|
| 3 | 1383 | 2 / 1 | Se mantienen bolso naranja, parte superior clara y pantalón azul pese al cambio de vista. |
| 6 | 811 | 3 / 5 | La camiseta amarilla y la mochila roja aparecen en varias coincidencias del ranking. |
| 9 | 1319 | 5 / 3 | La combinación de estampado gris y short turquesa se conserva entre cámaras. |
| 16 | 319 | 4 / 1 | Acierta top-1 aun con numerosos distractores también vestidos de negro. |
| 17 | 238 | 2 / 3 | Coincide la combinación de parte superior marrón, short azul y equipaje. |

## Fallos

| Consulta | ID correcto | ID top-1 | Primer relevante | Observación e hipótesis visual |
|---|---:|---:|---:|---|
| 0 | 1191 | 284 | 23 | La consulta muestra una prenda blanca amplia y verde interior; los candidatos enfatizan verde. El cambio de apariencia y punto de vista puede dificultar la coincidencia. |
| 1 | 873 | 728 | 3 | El primer candidato comparte camisa azul y pantalón oscuro. La identidad correcta sí aparece tercera: falla Rank-1, pero acierta Rank-5. |
| 2 | 360 | 812 | 670 | El amarillo y las mochilas se repiten entre identidades. El modelo recupera distractores visualmente muy parecidos. |
| 4 | 1271 | 1272 | 88 | Mochila grande y oclusión parcial alteran la silueta. Aparecen candidatos con mochilas y prendas de tonos parecidos. |
| 5 | 455 | 226 | 31 | Ropa oscura, mochila y short se repiten en la galería, con diferencias de postura y encuadre. |

Estas observaciones provienen de inspeccionar imágenes. Son hipótesis descriptivas: no demuestran qué región o atributo causó la decisión de la red. Para demostrarlo harían falta intervenciones controladas. Los ejemplos completos e IDs están en el JSON de recuperaciones. HR y LR se miden con el mismo modelo; esta figura ilustra únicamente HR.
