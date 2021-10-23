La implementación original de DPT no se puede pasar a onnx porque no es scriptable. La gente se queja en [1 (repo de DPT)](https://github.com/isl-org/DPT/issues/42) y [2 (repo de MiDas)](https://github.com/isl-org/MiDaS/issues/122). El autor le da prioridad y publica una rama dpt_scriptable donde se modifica el modelo para que pueda emplearse a través de torchscript (jit).

El error que nos encontramos si intentamos convertir a onnx usando esta rama es este:

```
Traceback (most recent call last):
  File "/home/gsb/Documentos/MasterIngInf/TFM/DPT/monodepth_to_onnx.py", line 232, in <module>
    run(
  File "/home/gsb/Documentos/MasterIngInf/TFM/DPT/monodepth_to_onnx.py", line 114, in run
    torch.onnx.export(model, dummy_input, "dpt.onnx", verbose=True, export_params=True, opset_version=13)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/onnx/__init__.py", line 317, in export
    return utils.export(model, args, f, export_params, verbose, training,
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/onnx/utils.py", line 107, in export
    _export(model, args, f, export_params, verbose, training, input_names, output_names,
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/onnx/utils.py", line 719, in _export
    _model_to_graph(model, args, verbose, input_names,
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/onnx/utils.py", line 493, in _model_to_graph
    graph, params, torch_out, module = _create_jit_graph(model, args)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/onnx/utils.py", line 437, in _create_jit_graph
    graph, torch_out = _trace_and_get_graph_from_model(model, args)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/onnx/utils.py", line 388, in _trace_and_get_graph_from_model
    torch.jit._get_trace_graph(model, args, strict=False, _force_outplace=False, _return_inputs_states=True)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/jit/_trace.py", line 1166, in _get_trace_graph
    outs = ONNXTracedModule(f, strict, _force_outplace, return_inputs, _return_inputs_states)(*args, **kwargs)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1106, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/jit/_trace.py", line 127, in forward
    graph, out = torch._C._create_graph_by_tracing(
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/jit/_trace.py", line 118, in wrapper
    outs.append(self.inner(*trace_inputs))
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1106, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1094, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "/home/gsb/Documentos/MasterIngInf/TFM/DPT/dpt/models.py", line 114, in forward
    inv_depth = self.forward_features(x).squeeze(dim=1)
  File "/home/gsb/Documentos/MasterIngInf/TFM/DPT/dpt/models.py", line 75, in forward_features
    layer_1, layer_2, layer_3, layer_4 = self.pretrained(x)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1106, in _call_impl
    return forward_call(*input, **kwargs)
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/nn/modules/module.py", line 1094, in _slow_forward
    result = self.forward(*input, **kwargs)
  File "/home/gsb/Documentos/MasterIngInf/TFM/DPT/dpt/vit.py", line 324, in forward
    layer_3 = self.act_postprocess3(layer_3.unflatten(2, out_size))
  File "/home/gsb/miniconda3/envs/tfm/lib/python3.9/site-packages/torch/_tensor.py", line 958, in unflatten
    return super(Tensor, self).unflatten(dim, sizes, names)
RuntimeError: NYI: Named tensors are not supported with the tracer

Process finished with exit code 1
```

Aparentemente con esto se encuentra la gente [(DPT issue)](https://github.com/isl-org/DPT/issues/42). Es un problema con la operación unflatten [(Pytorch issue)](https://github.com/pytorch/pytorch/issues/49538). Se soluciona tal y como sugieren en el issue de pytorch, empleando [view](https://pytorch.org/docs/stable/generated/torch.Tensor.view.html) en vez de unflatten.

```
(vit.py line ~320)

layer_3 = self.act_postprocess3(layer_3.unflatten(2, out_size))
layer_4 = self.act_postprocess4(layer_4.unflatten(2, out_size))

```

Se cambia por

```

x3, y3, z3 = layer_3.shape
layer_3 = self.act_postprocess3(layer_3.view(x3, y3, *out_size))
x4, y4, z4 = layer_4.shape
layer_4 = self.act_postprocess4(layer_4.view(x4, y4, *out_size))

```

Este cambio se ha dejado comentado en el [issue correspondiente](https://github.com/isl-org/DPT/issues/42) y se está añadiendo a la rama dpt_scriptable en [mi fork](https://github.com/guillesanbri/DPT/tree/dpt_scriptable).


Hay problemas con la inferencia del tamaño de la lista de forma dinámica al llamar al modelo con torch.jit.script (creo que es por esto, es la única diferencia que hay entre mi script monodepth_to_onnx.py, donde no hay error, y run_monodepth.py, donde si que hay error). Se soluciona cambiando a:

```

x3, y3, z3 = layer_3.shape
layer_3 = self.act_postprocess3(layer_3.view(x3, y3, out_size[0], out_size[1]))
x4, y4, z4 = layer_4.shape
layer_4 = self.act_postprocess4(layer_4.view(x4, y4, out_size[0], out_size[1]))

```