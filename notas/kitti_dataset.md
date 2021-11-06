# Raw y anotaciones

#### Anotaciones

Descargamos las imágenes anotadas desde [la página oficial de KITTI.](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction)

Las anotaciones son aproximádamente 14 GB tanto comprimidas como descomprimidas.

Dentro de las anotaciones hay train y val. 

Hay una descarga adicional de 2 GB que pone que son validation y test seleccionados manualmente.

- El conjunto de test no tiene las anotaciones, es para el benchmark oficial. Son 500 imágenes (365 MB)
- El conjunto de validación tiene 1000 elementos (672 MB) de distinto tamaño a los elementos del fichero de 14 GB (1216x352 en este conjunto de validación frente a 1242x375 en el conjunto de val del fichero de 14 GB). Aparentemente el tamaño de imagen empleado es 1242x375 así que ignoramos este fichero de 2 GB.

Las anotaciones descargadas corresponden a los ficheros synced+rectified data de los datos raw. Hay un script para descargar todo de forma automática.

``` bash
# Para contar los ficheros
find . -type f | wc -l

```

Volviendo a las anotaciones del fichero de 14GB:

- El conjunto de entrenamiento tiene 85898 imágenes (13.2 GB).
- El conjunto de validación tiene 6852 imágenes (1.1 GB).


#### Raw

Descargamos el raw data con ayuda del script que proporcionan en KITTI. Son 192760 imágenes (\~196 GB). Dentro de cada "vídeo" hay cuatro carpetas, image_00, image_01, image_02 e image_03. Las dos primeras corresponden a imágenes en escala de grises (izq-der) y las dos siguientes a imágenes en color (izq-der). Las imágenes en escala de grises tienen ligeramente mayor resolución y menos ruido. No obstante, para el entrenamiento se emplean las de color.

Hay imágenes raw que no tienen anotación, en concreto: 192760/2 = 96430 = 6852 (validación) + 85898 (entrenamiento) + 3680 (?)

---

# Splits

Pese a que el dataset está ya dividido en train y val, en las publicaciones de estimación de profundidades monocular no se usan estos conjuntos, si no que predominante se emplea el ["Eigen split"](https://stackoverflow.com/questions/63512296/kitti-eigen-split). De este split hay distintas variantes con escasa información en general.

En el [repositorio de Monodepth2](https://github.com/nianticlabs/monodepth2/tree/master/splits) hay varios directorios:

1. eigen: Solo especifica ficheros de test (697 imágenes).
2. eigen_benchmark: Solo especifica ficheros de test (652 imágenes).
3. eigen_full: Especifica ficheros de entrenamiento (45200 imágenes) y de validación (1776 imágenes).
4. eigen_zhou: Especifica ficheros de entrenamiento (39810 imágenes) y de validación (4424 imágenes).

En estos ficheros también se especifica si coger la imágen de la cámara derecha o izquierda.

Para entrenar, vamos a usar el eigen full (entrenamiento y validación).

Para la evaluación final de los modelos, se sigue el protocolo establecido por los autores de DPT y se emplean sus mismos scripts con su mismo conjunto de test. Este conjunto de test son 697 imágenes que coinciden una por una con el fichero eigen_test (1.) del repositorio Monodepth2.

Se adapta el dataloader para que lea imágenes en función de la información de este tipo de ficheros.

---

# Arreglando las listas de entrenamiento y validación


Las listas de ficheros descargadas del repositorio de Monodepth2 hacen referencia a ficheros cuyas etiquetas no están en los datos descargados. Tanto en eigen_zhou como en eigen_full. Se crea un script que busque y elimine los archivos que dan problemas. Como no es una cantidad significativa se eliminan y se obtienen nuevos ficheros con el sufijo fix. En el notebook dataset_split_file_checker.ipynb hay datos concretos para cada fichero probado. Resumen:

|                    | train_eigen_full | val_eigen_full | train_eigen_zhou | val_eigen_zhou |
|:------------------:|:----------------:|:--------------:|:----------------:|:--------------:|
| Num. files in list |       45200      |      1776      |       39810      |      4424      |
|     Missing raw    |      0 (0%)      |     0 (0%)     |      0 (0%)      |     0 (0%)     |
|    Missing depth   |    630 (1.39%)   |   30 (1.69%)   |    395 (0.99%)   |   33 (0.75%)   |


Para evitar disgustos posteriormente, se ha comprobado que las imágenes del conjunto de imágenes de test (recordemos, descargadas desde el repositorio de DPT) no se encuentran ni en el conjunto de entrenamiento ni el de test de eigen full.

```python

import os

files = os.listdir("../DPT/input/")
files = [f for f in files if f.endswith("png")]

train_file = "train_files_eigen_full_fix.txt"
with open("../data/KITTI/" + train_file) as ftrain:
    lines = ftrain.readlines()
    lines = [l.split("/")[1] for l in lines]
    lines = [l.split(" ")[:2] for l in lines]
    lines = [f"{file}_{n.zfill(10)}.png" for file, n in lines]

set(lines) & set(files)

>> set()

```
