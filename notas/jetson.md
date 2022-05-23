# Configuración
- Descargamos el devkit desde [aquí](https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit#write)
- Descargamos balenaEtcher

# Documentación
- En Jetson Nano Developer Kit User Guide no hay nada interesante.
- [Docs de Jetson SDK](https://docs.nvidia.com/jetson/jetpack/index.html) overview de los paquetes instalados.
- [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html)
	+ Multi-instance GPU: Permite dividir la GPU en múltiples GPUs más pequeñas, puede acelerar la inferencia mucho si el uso de GPU es bajo. (Ampere o posterior).
	+ DALI son primitivas para trabajar con imagen, audio y video.
	+ Principal medio para importar modelos entrenados: ONNX