Es necesario instalar tensorflow, tensorflow addons y onnx antes de instalar onnx-tf.

[Tutorial](https://github.com/onnx/tutorials/blob/master/tutorials/OnnxTensorflowImport.ipynb)

1.- Hacemos un docker con tensorflow 2.5.1-gpu-jupyter

Dentro del docker (habrá que meter esto en el Dockerfile)

2.- pip install tensorflow-addons
3.- pip install onnx
4.- git clone git@github.com:onnx/onnx-tensorflow.git && cd onnx-tensorflow
5.- pip install -e .

6.- onnx-tf convert -i dpt.onnx -o ./tfmodel-dpt

Salta un RuntimeError: Resize coordinate_transformation_mode=pytorch_half_pixel is not supported in Tensorflow.

Parece que el culpable es un parámetro align_corners en las funciones de escalado de pytorch (si está en False revienta).
Para solucionarlo en teoría hay que cambiarlo a True y **volver a entrenar el modelo**. Esto sería catastrófico ya que entrenar todo es practicamente inviable (datasets + computo).
Probamos a cambiarlo a piñon a ver que diferencia hay en los resultados.

[Related](https://github.com/onnx/onnx-tensorflow/issues/632)

grep -rnw . -e align_corners

Efectivamente hay varias ocurrencias

```

./dpt/blocks.py:140:    def __init__(self, scale_factor, mode, align_corners=False):
./dpt/blocks.py:151:        self.align_corners = align_corners
./dpt/blocks.py:167:            align_corners=self.align_corners,
./dpt/blocks.py:239:            output, scale_factor=2.0, mode="bilinear", align_corners=True
./dpt/blocks.py:322:        align_corners=True,
./dpt/blocks.py:332:        self.align_corners = align_corners
./dpt/blocks.py:367:            output, scale_factor=2.0, mode="bilinear", align_corners=self.align_corners
./dpt/models.py:21:        align_corners=True,
./dpt/models.py:104:            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
./dpt/models.py:138:            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
./dpt/vit.py:41:        attn, size=shape[2:], mode="bicubic", align_corners=False
./dpt/vit.py:248:            posemb_grid, size=[gs_h, gs_w], mode="bilinear", align_corners=False
Coincidencia en el archivo binario ./dpt/__pycache__/blocks.cpython-39.pyc
Coincidencia en el archivo binario ./dpt/__pycache__/models.cpython-39.pyc
Coincidencia en el archivo binario ./dpt.onnx
./run_monodepth.py:159:                    align_corners=False,
./run_segmentation.py:106:                out, size=img.shape[:2], mode="bicubic", align_corners=False

```

Se cambia el init de block.py:140: ; vit.py:41: ; vit.py:248: ; monodepth.py:159: (este último no debería influir en el modelo exportado a onnx ya que es parte del postprocesado).

Los cambios han sido un exito, por lo menos la inferencia, no hay diferencias visibles en el funcionamiento del modelo. Pasamos a exportar el modelo a onnx a ver si se ha solucionado en tf.

Hay un error distinto, parece que hemos pasado el align_corners con exito (!)

```

tensorflow.python.autograph.pyct.error_utils.KeyError: in user code:

    /workspace/onnx-tensorflow/onnx_tf/backend_tf_module.py:99 __call__  *
        output_ops = self.backend._onnx_node_to_tensorflow_op(onnx_node,
    /workspace/onnx-tensorflow/onnx_tf/backend.py:347 _onnx_node_to_tensorflow_op  *
        return handler.handle(node, tensor_dict=tensor_dict, strict=strict)
    /workspace/onnx-tensorflow/onnx_tf/handlers/handler.py:59 handle  *
        return ver_handle(node, **kwargs)
    /workspace/onnx-tensorflow/onnx_tf/handlers/backend/resize.py:281 version_13  *
        return cls.version_11(node, **kwargs)
    /workspace/onnx-tensorflow/onnx_tf/handlers/backend/resize.py:201 version_11  *
        roi = tensor_dict[node.inputs[1]]

    KeyError: ''

```

Cambiamos a tf 2.6.0 y onnx 1.9.0
El error persiste.

Probamos con TensorRT y queda pendiente escribir un issue en [el repo de onnx-tf](https://github.com/onnx/onnx-tensorflow)