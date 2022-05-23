Seguimos las instrucciones de instalación del [repositorio de onnx-tensorrt](https://github.com/onnx/onnx-tensorrt), que nos lleva al repo de TensorRT para instalar este y TensorRT OSS

Primero clonamos el repositorio

```bash

git clone https://github.com/NVIDIA/TensorRT
cd TensorRT
git submodule update --init --recursive

```

Construimos la imagen de docker recomendada en [el repo de TensorRT](https://github.com/NVIDIA/TensorRT#setting-up-the-build-environment) de forma que coincida con las especificaciones del equipo.
En este caso ubuntu 20.04 y cuda 11.2

```bash

./docker/build.sh --file docker/ubuntu-20.04.Dockerfile --tag tensorrt --cuda 11.2.0

```

Despues de probar varias combinaciones, la única que no me ha dado problemas es:

```bash

./docker/build.sh --file docker/ubuntu-18.04.Dockerfile --tag tensorrt --cuda 10.2

```

Entramos al docker (desde TFM/TensorRT) y hacemos el build:
(Al salir del contenedor se borra todo, se puede cambiar en el fichero launch dentro de TFM/TensorRT/docker)

./docker/launch.sh --tag tensorrt --gpus all


```bash

cd $TRT_OSSPATH
mkdir -p build && cd build
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=10.2
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=10.2 -DPROTOBUF_VERSION=3.0.0  # Probamos pidiendo que instale protobuf 3
make -j$(nproc)

```

Volvemos al repo de onnx-tensorrt

```

git clone https://github.com/onnx/onnx-tensorrt
cd onnx-tensorrt
mkdir build && cd build
cmake .. -DTENSORRT_ROOT=$TRT_OSSPATH && make -j$(nproc)  # No tengo nada claro que el directorio que haya que pasar sea ese la verdad.
# Salta este error: https://github.com/onnx/onnx-tensorrt/issues/598

# Parece que hay que instalar protobuf antes y por lo tanto no funciona. Puede que haya que volver a hacer el build de arriba cuando esté instalado protobuf?
https://github.com/protocolbuffers/protobuf/blob/master/src/README.md

# También puede que haya que hacer un make clean en el directorio de tensorrt porque he ejecutado dos veces el make borrando a piñon lo de la carpeta.

```

sudo apt-get install autoconf automake libtool curl make g++ unzip
wget https://github.com/protocolbuffers/protobuf/releases/download/v3.0.0/protobuf-cpp-3.0.0.tar.gz
tar -xzvf protobuf-cpp-3.0.0.tar.gz
cd protobuf-3.0.0/
./autogen.sh
./configure --prefix=/usr
make
make check
sudo make install
sudo ldconfig


git clone https://github.com/onnx/onnx-tensorrt
cd onnx-tensorrt
mkdir build && cd build
# https://githubmemory.com/repo/onnx/onnx-tensorrt/issues/611
git submodule init
git submodule update
cmake .. -DTENSORRT_ROOT=$TRT_OSSPATH && make -j$(nproc)  # No tengo nada claro que el directorio que haya que pasar sea ese la verdad.
export LD_LIBRARY_PATH=$PWD:$LD_LIBRARY_PATH


Parece que así va, solamente desde build llamando con ./onnx2trt, habrá que hacer toda la instalación en el Dockerfile o que sea permanente y volver a entrar en el contenedor.

(en el host)
docker cp dpt.onnx kind_albattani:/workspace/
(en el contenedor)
./onnx2trt ../../dpt.onnx -o ../../dpt.trt

Errores:

# https://github.com/onnx/tensorflow-onnx/issues/883
# Parece que no es algo crítico, puede ser por infinitos, habrá que mirarlo de todas formas.
[2021-10-24 10:13:04 WARNING] onnx2trt_utils.cpp:366: Your ONNX model has been generated with INT64 weights, while TensorRT does not natively support INT64. Attempting to cast down to INT32.
[2021-10-24 10:13:05 WARNING] onnx2trt_utils.cpp:392: One or more weights outside the range of INT32 was clamped

# https://stackoverflow.com/questions/59762694/ishufflelayer-applied-to-shape-tensor-must-have-0-or-1-reshape-dimensions-dimen (Nada)
# https://github.com/onnx/onnx-tensorrt/issues/264 (Nada)
[2021-10-24 10:13:05   ERROR] [shuffleNode.cpp::symbolicExecute::387] Error Code 4: Internal Error (Reshape_68: IShuffleLayer applied to shape tensor must have 0 or 1 reshape dimensions: dimensions were [-1,2])


While parsing node number 79 [Pad -> "476"]:
ERROR: ModelImporter.cpp:179 In function parseGraph:
[6] Invalid Node - Pad_79
[shuffleNode.cpp::symbolicExecute::387] Error Code 4: Internal Error (Reshape_68: IShuffleLayer applied to shape tensor must have 0 or 1 reshape dimensions: dimensions were [-1,2])




Hay error pasando tanto el dpt como un modelo del zoo de pytorch. Probamos a instalar una versión superior de protobuf para ver si no saltan los errores al hacer el make check.
https://github.com/protocolbuffers/protobuf/releases/download/v3.19.0/protobuf-cpp-3.19.0.tar.gz

¿Es posible que no esté instalado tensorrt? Probamos a instalar lo primero el protobuf de arriba. Se supone que -DTENSORRT_ROOT tiene que ser /opt/tensorrt y en el contenedor ahora mismo no hay nada ahí.
Esta versión de protobuf se instala sin problemas.

Se vuelve a instalar TRTOSS (?)
cmake .. -DTRT_LIB_DIR=$TRT_LIBPATH -DTRT_OUT_DIR=`pwd`/out -DCUDA_VERSION=10.2












Volvemos a hacerlo todo desde cero

El mismo docker

Hacemos el make de TRTOSS, sin especificar protobuf porque revienta aparentemente. Si primero se tira el make sin incluir lo de protobuf y luego incluyendolo si que va (??????)

Parece que aunque no especifiquemos el protobuf se compila igual, hay que mirarlo

¿Están?
- Protobuf
- TensorRT
- TensorRTOSS



He actualizado los drivers de la gráfica para poder usar cuda 11.4, volvemos a instalar todo desde cero con los paquetes más actualizados a ver qué tal.
Otro frente abierto es probar con el container que proporciona nvidia en https://ngc.nvidia.com/catalog/containers/nvidia:tensorrt