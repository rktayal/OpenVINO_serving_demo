# OpenVINO_serving_demo
Demonstrates on how to get started on serving an OpenVINO model

## Getting Started with OpenVINO model server
Inference model server implementation with gRPC interface, compatible with TensorFlow serving API and OpenVINO™ as the execution backend.


“OpenVINO™ model server” is a flexible, high-performance inference serving component for artificial intelligence models.
The software makes it easy to deploy new algorithms and AI experiments, while keeping the same server architecture and APIs like in TensorFlow Serving.

The repo illustrates step by step on how to serve a OpenVINO detection model. (ssd_mobilenet_v2_coco_2018_03_29 for this example)

### Requirements
```
Tested on CentOS 7 (Linux)
Docker
Python3
Python Packages
- futures
- grpcio
- grpcio-tools
- numpy
- opencv-python
- protobuf
- tensorflow
- tensorflow-serving-api
```
You can use the command `pip install <package_name>` to resolve the above Python Dependencies.

### Preparing the models
Before starting to serve the model server container, you should prepare the models to be served. You can convert your model using [model_optimizer](https://software.intel.com/en-us/articles/OpenVINO-ModelOptimizer) tool provided in [OpenVINO toolkit](https://software.intel.com/en-us/openvino-toolkit/choose-download). For this illustration, I am using the [ssd_mobilenet_v2_coco](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) converted model.
The directory structure should be created as depicted below.

```
models/
|---model1/
|---|-----1/
|---|-----|---- frozen_inference_graph.bin
|---|-----|---- frozen_inference_graph.xml

```
The above `model` directory is checked in consisting the required structure and models.
### Pulling up the docker image
You can pull the publicly available docker image from [dockerhub](https://hub.docker.com/r/intelaipg/openvino-model-server/)
```
docker pull intelaipg/openvino-model-server
```
After running this command, you should be able to view the image under `docker images` with the name `docker.io/intelaipg/openvino-model-server:latest`

### Starting Docker container
Once the models are ready, you can start the Docker container with the OpenVINO™ model server. To enable just a single model, you do not need any extra configuration file, so this process can be completed with just one command like below:
```
docker run --rm -d  -v "$(pwd)"/models/:/opt/ml:Z -p 9001:9001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/ml/model1 --model_name first_model --port 9001
```
- option `-v` defines how the models folder should be mounted inside the docker container.
- option `-p` exposes the model serving port outside the docker container.
- `ie-serving-py:latest` represent the image name which can be different depending the tagging and building process.
- `start_server.sh` script activates the python virtual environment inside the docker container.
- `--model_name` value can be anything. NOTE: The same name should be passed in client script.
- `ie_serving` command starts the model server which has the following parameters:
