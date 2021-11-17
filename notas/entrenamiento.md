# Teoría

#### Entrenamiento base
1. El entrenamiento de DPT "closely follows" el de MiDas (es de los mismos autores). Es una perdida monocular (scale and shift invariant trimmed loss) que opera en una representación de profundidad inversa. A esta función de pérdida se le añade la gradient-matching loss propuesta en [Zhengqi](https://arxiv.org/abs/1804.00607).
2. Se usa multi-objective optimization con Adam (lr=1e-5 en el backbone y lr=1e-4 en el decoder). El encoder se inicializa con los pesos entrenados en Imagenet y el decoder se inicializa aleatoriamente.
3. Se desactiva el batch_normalization en el decoder.
4. Se resizea la imagen para que el lado más corto sean 384 píxeles y se hacen recortes aleatorios de 384x384.
5. Se entrena 60 epochs de 72.000 steps con bs=16.
6. Data augmentation son horizontal flips
7. Primero se entrenan 60 epochs en un curated subset (ReDWeb, HRWSI, BlendedMVS) antes de entrenar en el dataset completo. 

#### Finetuning

1. Se usa la loss propuesta por [Eigen](https://arxiv.org/abs/1406.2283) para el finetuning.
2. El gradient-matching loss se inhabilita para finetunear en KITTI ya que las anotaciones de este dataset no son densas.
3. La red se ha entrenado con una función de pérdida affine-invariant. Por lo tanto sus predicciones están escaladas y desplazadas arbitrariamente y pueden tener magnitudes altas. La falta de correspondencia en la magnitud de la predicción y el ground truth dominaría la función de pérdida, por lo que primero se alinean usando el robust alignment procedure descrito en 30. Las escalas y shifts del conjunto de entrenamiento se promedian y se obtiene la escala y shift que se aplica a las predicciones antes de pasarle los resultados a la función de pérdida. Como nosotros vamos a entrenar con KITTI, se emplean los valores de scale y shift que emplean ellos en su modelo preentrenado.

---

# Progreso

Se escribe un script de entrenamiento básico [Link](https://github.com/guillesanbri/DPT/blob/dpt_scriptable/train.py)

Comprobar si hay contaminación en el conjunto de evaluación del script de evaluación (no debería) -> Comprobado en notebooks/eval_with_pngs_split_check

Hay una explosión en la función de pérdida, se arregla.

Tenemos OOM en la GPU con batch_size == 1, probamos diferentes enfoques:
- [x] Imagen de KITTI con tamaño de inferencia tal y como especifica el paper (1216x352) -> OOM en el segundo batch (?).
- [x] Imagen reducida a 640x192 (bs=1) -> Aparentemente funciona, la función de perdida hay que aplicarle aún una máscara para que ignore los píxeles de las anotaciones que no tienen valor. Cuando esto esté hecho, poner a entrenar unas cuantas epochs y ver si mejora respecto del modelo base.
	- [x] Máscara ya aplicada
- [x] Imagen reducida a 640x192 (bs=2) -> No salta OOM, pero la pérdida se vuelve negativa. Esto muy probablemente sea un error en la implementación de la función de perdida, hay que mirarlo.
	- [x] La función de pérdidas se ha arreglado y debería funcionar bien, hay que probar a ejecutar esto. -> No explota la pérdida ya, aparentemente se podría usar un batch_size de 2.


Probamos con el modelo finetuneado en KITTI por nosotros (tamaño de evaluación != tamaño de entrenamiento). Este entrenamiento han sido 5 epochs con 1000 imágenes aleatorias del conjunto de entrenamiento, hay que modificar la función de pérdida, almacenar las métricas y poner a entrenar de nuevo.

El entrenamiento funciona!

Descargamos wandb para monitorizar el entrenamiento en web. 
conda install -c conda-forge wandb

Modificamos la función de pérdida para que no tenga en cuenta los píxeles sin anotación y añadimos métricas adicionales al entrenamiento, las logueamos en wandb.

Se ha ejecutado el entrenamiento durante 10 epochs con las 44k imágenes del conjunto de entrenamiento (~17h), el modelo no ha saturado y podía haber seguido aprendiendo, seguían bajando tanto la pérdida en el conjunto de entrenamiento como en el conjunto de validación. (dpt_hybrid_custom-kitti-lpuhqnwx.pt - wandb/test/twinkling-light-15)


---

# Resultados preliminares

Estas métricas son las resultantes de evaluar siguiendo la metodología (y scripts) proporcionados en el repositorio de DPT para reproducir las métricas que dan ellos en la publicación. Por lo tanto, los cambios introducidos en el entrenamiento de los modelos, por ejemplo, hacer las imágenes más pequeñas, no tienen lugar aqui ya que el tamaño de las imágenes en estas evaluaciones es idéntico al que usan en el paper original de DPT. Las imágenes que se emplean son las que están en la carpeta input que proporcionan en el repo de DPT. Se ha comprobado que ninguna de estas imágenes esté en el conjunto de entrenamiento empleado posteriormente.

---

## 1. Modelo base DPT-Hybrid entrenado en MIX6 sin finetuning

- **Pesos**: weights/dpt_hybrid-midas-d889a10e.pt

``` bash

python run_monodepth.py --model_type dpt_hybrid --kitti_crop --absolute_depth
python ./eval_with_pngs.py --pred_path ./output_monodepth/ --gt_path ./input/gt/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --garg_crop --do_kb_crop

```
##### Output (Full size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.166 | 0.326 | 0.461 |   0.590  |  9.326  | 16.758 |   1.574   | 123.437 |  0.501  |

---

## 2. Modelo base DPT-Hybrid entrenado en MIX6 finetuneado en KITTI por los autores

- **Pesos**: weights/dpt-hybrid-kitti-e7069aae.pt

``` bash

python run_monodepth.py --model_type dpt_hybrid_kitti --kitti_crop --absolute_depth
python ./eval_with_pngs.py --pred_path ./output_monodepth/ --gt_path ./input/gt/ --dataset kitti --min_depth_eval 1e-3 --max_depth_eval 80 --garg_crop --do_kb_crop

```

##### Output (Full size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.959 | 0.995 | 0.999 |   0.062  |  0.222  |  2.573 |   0.092   |  8.293  |  0.027  |

---

## 3. Modelo base DPT-Hybrid entrenado en MIX6 finetuneado en KITTI por mí (5 epochs, 1000 imágenes por epoch)

Este modelo era una prueba de concepto para ver si el entrenamiento funcionaba, mejoró los resultados del modelo proporcionado por los autores sin funetunear, por lo que se termino el script de entrenamiento y se lanzó el experimento siguiente (4.)

- **Pesos**: N/A

``` bash

./run_eval_with_pngs.sh

```

##### Output (Full size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.351 | 0.703 | 0.921 |   0.416  |  3.266  |  7.399 |   0.401   |  32.631 |  0.148  |

---

## 4. Modelo base DPT-Hybrid entrenado en MIX6 finetuneado en KITTI por mí (10 epochs, ~44k imágenes por epoch) ~17h

- **Pesos**: weight/dpt_hybrid_custom-kitti-lpuhqnwx.pt
- **Wandb**: test/twinkling-light-15
- **Notas**: El modelo no ha saturado y podía haber seguido aprendiendo, seguían bajando tanto la pérdida en el conjunto de entrenamiento como en el conjunto de validación. Sería interesante probar con 20 epochs ¿o más? Probablemente sea prioritario entrenar los modelos modificados y por lo tanto repetir este entrenamiento se podría dejar para más adelante. Cuando se repita este experimento con más epochs, el bs puede ser 2 en vez de 1.

``` bash

./run_eval_with_pngs.sh

```

##### Output (Full size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.723 | 0.906 | 0.973 |   0.159  |  1.198  |  6.607 |   0.251   |  22.173 |  0.078  |

##### Output (Reduced size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.917 | 0.985 | 0.997 |   0.102  |  0.407  |  3.448 |   0.140   |  11.229 |  0.046  |

---

## 5. Modelo base DPT-Hybrid con Performer Attention entrenado en MIX6 finetuneado en KITTI por mí (13 epochs, ~44k imágenes por epoch) ~20h

- **Pesos**: weight/dpt_hybrid_custom-kitti-ylqiuazs_006.pt -> A partir del 6 caen en picado los resultados (?)
- **Wandb**: test/performer-lr1e-4
- **Notas**: Se fue la luz. No había empezado a crecer la perdida en el conjunto de validación.

``` bash

./run_eval_with_pngs.sh

```

##### Output (Full size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.516 | 0.843 | 0.966 |   0.216  |  1.590  |  7.217 |   0.314   |  19.390 |  0.112  |

##### Output (Reduced size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.914 | 0.979 | 0.989 |   0.101  |  0.521  |  3.478 |   0.203   |  11.479 |  0.073  |

---

## 6. Modelo base DPT-Hybrid con Compressed Attention 2 entrenado en MIX6 finetuneado en KITTI por mí (20 epochs, ~44k imágenes por epoch) ~35.5h

- **Pesos**: weight/dpt_hybrid_custom-kitti-whdtqsbe.pt
- **Wandb**: test/mem-com-2-lr1e-5
- **Notas**: Se bajó el lr porque explotaba la función de pérdida. No había empezado a crecer la perdida en el conjunto de validación. Solo se podía usar batch_size 1. Com compression rate 3 explotaba (antes de bajar el learning rate), hay que probar con 3 y este lr más bajo.

``` bash

./run_eval_with_pngs.sh

```

##### Output (Full size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.578 | 0.901 | 0.979 |   0.236  |  1.452  |  6.367 |   0.273   |  24.511 |  0.097  |

##### Output (Reduced size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.930 | 0.987 | 0.997 |   0.091  |  0.365  |  3.311 |   0.129   |  10.645 |  0.041  |

---

## 7. Modelo base DPT-Hybrid con Compressed Attention 3 entrenado en MIX6 finetuneado en KITTI por mí (20 epochs, ~44k imágenes por epoch) ~35h

- **Pesos**: weight/dpt_hybrid_custom-kitti-pgjabahj.pt
- **Wandb**: test/mem-com-3-lr1e-5
- **Notas**: Se bajó el lr porque explotaba la función de pérdida. No había empezado a crecer la perdida en el conjunto de validación. Solo se podía usar batch_size 1.

``` bash

./run_eval_with_pngs.sh

```

##### Output (Full size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.535 | 0.821 | 0.951 |   0.287  |  2.105  |  6.810 |   0.313   |  25.608 |  0.112  |

##### Output (Reduced size)

|  d1 ↑ |  d2 ↑ |  d3 ↑ | AbsRel ↓ | SqRel ↓ | RMSE ↓ | RMSElog ↓ | SILog ↓ | log10 ↓ |
|:-----:|:-----:|:-----:|:--------:|:-------:|:------:|:---------:|:-------:|:-------:|
| 0.927 | 0.987 | 0.997 |   0.093  |  0.373  |  3.332 |   0.130   |  10.739 |  0.042  |