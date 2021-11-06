# Pasos a seguir (Todas las tareas se duplican para NYUDepthV2):

**Datos**:
- [x] Hacer un Dataset y Dataloader de KITTI
- [ ] Hacer un Dataset y Dataloader de NYUDepthV2

[**Evaluación**](https://github.com/isl-org/DPT/blob/main/EVALUATION.md)
- Sample code [KITTI](https://gist.github.com/ranftlr/45f4c7ddeb1bbb88d606bc600cab6c8d), [NYUv2](https://gist.github.com/ranftlr/a1c7a24ebb24ce0e2f2ace5bce917022)
- Métricas sobre las imágenes ya predichas [Link](https://raw.githubusercontent.com/cogaplex-bts/bts/5a55542ebbe849eb85b5ce9592365225b93d8b28/utils/eval_with_pngs.py)
- [x] Asegurarse de que la evaluación de KITTI se ha hecho con el split que tengo.
- [x] Script de evaluación de DPT-Hybrid MIX6 en KITTI + Resultados
- [x] Script de evaluación de DPT-Hybrid KITTI en KITTI + Resultados
- [x] Aprender sobre cada una de las métricas.
- [ ] Scripts de velocidad de inferencia.
	- [ ] Pytorch
	- [ ] ONNX
	- [ ] TFLite
	- [ ] TensorRT

**Entrenamiento** 
- [(Info que puede ser interesante)](https://github.com/isl-org/DPT/issues/3), parece que es para segmentación así que a lo mejor no tanto. Probablemente la mejor opción sea partir del sample code de evaluación que hay arriba.
- [x] Finetunear DPT-Hybrid MIX6 con KITTI
- [x] Repetir la evaluación del modelo finetuneado para comparar con el proporcionado por los autores.
- [ ] Repetir bucle siguiente tantas veces como modificaciones haya
	- [ ] Modificar modelo
	- [ ] Finetunear modelo modificado con los pesos base (strict = False) en KITTI
	- [ ] Evaluar y comparar con el del repo y con el mio

**Deployment**
- [ ] Pasar a ONNX los modelos generados
- [ ] Pasar a TensorRT/TFLite los modelos generados
	- [ ] Probar el contenedor que proporciona NVIDIA [Link](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
	- Links de interes [1](https://forums.developer.nvidia.com/t/bash-trtexec-command-not-found/127302), 