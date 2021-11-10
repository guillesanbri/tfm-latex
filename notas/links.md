# Transformers eficientes
- ResT [(GitHub)](https://github.com/wofmanaf/ResT) [(Paper)](https://arxiv.org/pdf/2105.13677v3.pdf) ⇒ Habría que cambiar el backbone entero, pueden complicarse los hooks
- Routing Transformer [(GitHub)](https://github.com/lucidrains/routing-transformer) [(Paper)](https://arxiv.org/pdf/2003.05997.pdf) ⇒ Puede complicarse
- Reformer [(GitHub)](https://github.com/lucidrains/reformer-pytorch) [(Paper)](https://openreview.net/pdf?id=rkgNKkHtvB) ⇒ Parece buena opción para las primeras pruebas
- Linformer [(GitHub)](https://github.com/lucidrains/linformer) [(Paper)](https://arxiv.org/pdf/2006.04768.pdf) ⇒ Fixed sequence length, habría que fijar tamaño de las imágenes
- Memory Compressed Attention [(GitHub)](https://github.com/lucidrains/memory-compressed-attention) [(Paper)](https://arxiv.org/pdf/1801.10198.pdf) ⇒ Parece buena opción para las primeras pruebas, se puede configurar el grado de compresión (interesante)
- Performer [(GitHub)](https://github.com/lucidrains/performer-pytorch) [(Paper)](https://arxiv.org/pdf/2009.14794.pdf) ⇒ **Muy buena opción para las primeras pruebas**
- Fast Transformer [(GitHub)](https://github.com/lucidrains/fast-transformer-pytorch) [(Paper)](https://arxiv.org/pdf/2108.09084.pdf) ⇒ Fixed sequence length, habría que fijar tamaño de las imágenes
- Longformer [(GitHub)](https://github.com/allenai/longformer) [(Paper)](https://arxiv.org/pdf/2004.05150.pdf) ⇒ No es de lucidrains, puede complicarse
- Sparse Attention [(GitHub)](https://github.com/openai/sparse_attention) [(Paper)](https://arxiv.org/pdf/1904.10509.pdf) ⇒ TF
- Local windowed attention [(GitHub)](https://github.com/lucidrains/local-attention) ⇒ Hay que meter q, k y v a la llamada de la capa, habrá que modificar más, pero **servirían los pesos de las matrices Wt, Wk y Wv ya entrenadas**.
- DeiT [(GitHub)](https://github.com/facebookresearch/deit) [(Paper)](https://arxiv.org/pdf/2012.12877.pdf) ⇒ Leer paper bien antes, habría que cambiar el backbone entero y los hooks. Puede que no merezca la pena.
- Swin Transformers [(GitHub)](https://github.com/microsoft/Swin-Transformer) [(Paper)](https://arxiv.org/pdf/2103.14030.pdf) ⇒ Leer paper bien antes, habría que cambiar el backbone entero y los hooks. Puede que no merezca la pena.
- 

# Tutoriales implementación Transformers
- [ViT](https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632)
- [ViT](https://nn.labml.ai/transformers/vit/index.html)
- 

# TensorRT
- [Onnx-TensorRT GitHub](https://github.com/onnx/onnx-tensorrt)
- [Protobuf installation gist](https://gist.github.com/diegopacheco/cd795d36e6ebcd2537cd18174865887b)
- [Protobuf installation guide](https://github.com/protocolbuffers/protobuf/blob/master/src/README.md)
- [TensorRT Repo](https://github.com/NVIDIA/TensorRT#setting-up-the-build-environment)
- [TensorRT NVidia Documentation](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html)
- [TensorRT Developer NVidia blog post](https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorrt-updated/)
- [Info about the TensorRT container](https://forums.developer.nvidia.com/t/bash-trtexec-command-not-found/127302)
- [TensorRT NGC Docker container](https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt)
- 

# Información sobre métricas
- [Paper Measuring the Performance of Single Image Depth Estimation Methods](https://ylatif.github.io/papers/IROS2016_ccadena.pdf)
- [Paper DPT](https://arxiv.org/pdf/2103.13413.pdf)
- [Paper BtS (sección 4.4)](https://arxiv.org/pdf/1907.10326.pdf)
- [Paper Evaluation of CNN-based Single-Image Depth Estimation Methods](https://arxiv.org/pdf/1805.01328.pdf)
- [MACs](https://en.wikipedia.org/wiki/Multiply%E2%80%93accumulate_operation)
- [Medir tiempos correctamente](https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f)
- [FLOPS vs FLOPs](https://stackoverflow.com/questions/58498651/what-is-flops-in-field-of-deep-learning)

# Información sobre funciones de pérdida
- [Paper Eigen Scale Invariant Error](https://arxiv.org/pdf/1406.2283.pdf)
- [Paper DPT](https://arxiv.org/pdf/2103.13413.pdf)
- [Paper MiDas](https://arxiv.org/pdf/1907.01341.pdf)
- [Gradient matching loss de MegaDepth](https://arxiv.org/pdf/1804.00607.pdf)
- [Repo de GitHub con implementaciones](https://github.com/imran3180/depth-map-prediction/blob/master/main.py)

# Otros
- [Train on Imagenet in 18 minutes](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/)
- [Plantilla TFM](https://gitlab.inf.uva.es/valecar/tfm-mii-template/-/tree/master/Latex)
- [Generador de tablas](https://www.tablesgenerator.com/)