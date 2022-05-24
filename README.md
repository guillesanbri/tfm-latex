# Estimación de Profundidad Monocular Online con Transformers Eficientes

## Links de interes

- [Repositorio del proyecto / Project repository](https://github.com/guillesanbri/DPT/tree/v1.0.0-tfm)
- [Repositorio de modelos / Model repository](https://zenodo.org/record/6574941)

## Abstract

La estimación de profundidad monocular consiste en recuperar automáticamente una aproximación de la dimensión perdida al proyectar una escena tridimensional en una imagen bidimensional. Este problema tiene infinitas soluciones geométricas, por lo que es prácticamente imposible resolverlo con técnicas de visión artificial tradicional. Sin embargo, las técnicas de Deep Learning son capaces de extraer distintas características de las imágenes que permiten aproximar una solución. En este trabajo se estudia este problema y las soluciones existentes, especialmente aquellas basadas en Transformers y aprendizaje supervisado. En una de estas soluciones, se llevan a cabo una serie de modificaciones y desarrollos que permiten reducir el tamaño del modelo original y multiplicar por cerca de cinco su velocidad de inferencia.
Además, se incluye un estudio exhaustivo, tanto cuantitativo como cualitativo, de la influencia de las modificaciones evaluando los modelos en el conjunto de datos KITTI, orientado a conducción autónoma.

Monocular depth estimation deals with the automatic recovery of an approximation of the dimension that is lost when projecting a three-dimensional scene into a two-dimensional image. This problem has an infinite number of geometric solutions, which makes it practically impossible to solve using traditional computer vision techniques. Nonetheless, Deep Learning techniques are capable of extracting different characteristics from the images that make it possible to approximate a solution. In this work this problem and the existing solutions are studied, especially those based on Transformers and supervised learning. In one of these solutions, a series of modifications and developments are carried out to reduce the size of the original model and multiply its inference speed by nearly five. Furthermore, an exhaustive
study, both quantitative and qualitative, of the influence of the different modifications is included, evaluating the models in the KITTI dataset, oriented to autonomous driving.

## Memoria Trabajo Fin de Máster

Este trabajo Fin de Máster está organizado de la siguiente manera: Primero, se han expuesto en la sección de Introducción tanto la motivación detrás del proyecto como los objetivos planteados; a continuación, en el segundo capítulo, se contextualiza el trabajo repasando las bases teóricas sobre las que se apoya su desarrollo y el Estado del Arte de estos campos; en el tercer capítulo, se presentan y justifican los materiales empleados para el desarrollo del trabajo y la metodología que se ha seguido; después, en el cuarto capítulo, se definen y explican aquellos desarrollos especialmente significativos dentro del trabajo para continuar en el quinto capítulo con los resultados que se han obtenido, haciendo especial hincapié en la influencia en estos resultados de cada uno de los desarrollos llevados a cabo. En el capítulo seis, se incluye una discusión de los resultados, y en el capítulo siete, las conclusiones del documento junto a una serie de líneas de investigación futuras que podrían explorarse para continuar trabajando en el contexto de este proyecto. Al final de este documento, se encuentra una sección de Anexos con información y material complementario.