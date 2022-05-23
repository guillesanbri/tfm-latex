# Estructura de los scripts de DPT

1. Para instanciar DPT hay que llamar a DPTDepthModel, que está dentro de dpt/models.py
2. En models.py está DPTDepthModel, que hereda de DPT (también en este fichero), que a su vez hereda de BaseModel, que está en dpt/base_model.py
	1. DPTDepthModel inicializa la cabeza del modelo y la pasa al constructor de DPT. La cabeza no hay que tocarla, así que en principio esta clase no hay que cambiarla.
	2. DPTDepthModel también llama al método forward_features (no sobreescrito) y añade una operación al final del forward. Hay que ver la función forward_features.
3. Clase DPT
	1. Aquí se definen los hooks de donde se cogen las representaciones intermedias, como vamos a usar el modelo hybrid **habrá que prestar atención a las capas que se especifican para el modelo vitb_rn50_384** que son la 0, la 1, la 8 y la 11.
	2. Se llama a la función make_encoder (**devuelve dos cosas**) que pertenece al script dpt/block.py y se añaden las capas de refinenet. Habrá que tocar make_encoder, la parte de refinenet se queda como está.
4. make_encoder en blocks.py
	1. make_encoder en la condicion de backbone vitb_rn50_384 que es la que nos interesa carga dos variables, pretrained y scratch. La primera es la respuesta de la funcion make_pretrained_vitb_rn50_384 en dpt/vit.py y la segunda la respuesta de make_scratch.
	2. Nos interesa make_pretrained_vitb_rn50_384. Los hooks a las representaciones internas de la red son un muy buen punto de desacoplamiento, mientras se puedan mantener esos hooks todo debería salir medianamente bien en la arquitectura de la red.
5. make_pretrained_vitb_rn50_384 en vit.py
	1. Carga el modelo vit_base_resnet50_384 llamando a timm.create_model.
	2. Devuelve una instancia de BackboneWrapper con el modelo cargado (el vit) como argumento.
6. BackboneWrapper en vit.py
	1. Entra al if hybrid_backbone, añade los dos primeros feature hooks a la resnet50. El preprocess y el readout_oper 1 y 2 los pone como nn.Identity ya que no hacen falta (los dos primeros hooks van a la resnet).
	2. enable attention hooks es False por lo que no se entra. Hay un comentario que dice que las features del transformer se cogen directamente en self.forward_features.
	3. Los hooks al transformers se hacen a mano almacenando la salida en vit.py:293 iterando a través de los bloques en el modelo self.model (el que se ha cargado de timm).
7. Mientras los hooks sigan bien no debería haber problema. Lo que hay que cambiar es el transformer de timm, habrá que ver la organización de bloques para ver que se cambia exactamente dentro de su implementación. **Si todo va bien (poco probable), no habría que cambiar nada del repositorio de DPT**.
8. Si creamos un modelo con `model = timm.create_model("vit_base_resnet50_384` 


# TODO

- [ ] Ver como modificamos timm