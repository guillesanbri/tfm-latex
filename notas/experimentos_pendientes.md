# Base
- [ ] Entrenar con 4/2 imágenes por batch
- [ ] Entrenar 20 epochs
- [ ] Entrenar con un lr más bajo (1e-5)

# Performer
- [ ] Retomar el entrenamiento que se fue la luz
- [ ] Probar con 4 imágenes por batch
- [ ] Probar con 4 cabezas en las capas de atención

# Memory-compressed
- [ ] Probar con el learning rate más bajo y 3 de compresíon para ver si explota la pérdida
- [ ] Probar con 4 cabezas en las capas de atención

# General
- [ ] Jugar con el número de capas que se ponen eficientes y cuales no, puede que sea interesante poner p.ej. local attention al principio y global al final [link](https://github.com/lucidrains/local-attention)
- [x] Probar con el test reducido de tamaño
- [ ] Probar entrenando con los recortes
- [ ] Probar entrenando partiendo de los pesos finetuneados en kitti y no los base para ver si hace mucho overfitting a kitti.