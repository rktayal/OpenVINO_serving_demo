# OpenVINO_Serving_Demo
Object detection models are some of the most sophisticated deep learning models. They’re capable of localizing and classifying objects in real time both in images and videos. But what good is a model if it cannot be used for production?
Thanks to the intel guys, we have OpenVINO serving, that is capable of serving our models in production.
The repo can be used as a plug and play to serve OpenVINO model (ssd_mobilenet_v2) & to get you started on OpenVINO serving.

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
The above `models` directory is checked in consisting the required structure and models.
### Pulling up the docker image
You can pull the publicly available docker image from [dockerhub](https://hub.docker.com/r/intelaipg/openvino-model-server/)
```
docker pull intelaipg/openvino-model-server
```
After running this command, you should be able to view the image under `docker images` with the name `docker.io/intelaipg/openvino-model-server:latest`

### Starting Docker container
Once the models are ready, you can start the Docker container with the OpenVINO™ model server. To enable just a single model, you do not need any extra configuration file, so this process can be completed with just one command like below:
```
docker run --rm -d  -v "$(pwd)"/models/:/opt/ml:z -p 9001:9001 ie-serving-py:latest \
/ie-serving-py/start_server.sh ie_serving model --model_path /opt/ml/model1 --model_name first_model --port 9001
```
- option `-v` defines how the models folder should be mounted inside the docker container.
- option `-p` exposes the model serving port outside the docker container.
- `ie-serving-py:latest` represent the image name which can be different depending the tagging and building process.
- `start_server.sh` script activates the python virtual environment inside the docker container.
- `--model_name` value can be anything. NOTE: The same name should be passed in client script.
- `ie_serving` command starts the model server which has the following parameters:

Now the container should be up and running. You can view the logs using the command
`docker logs <container_id>`
You should be able to view something like below:
```
2019-04-14 11:57:05,907 - ie_serving.main - INFO - Log level set: INFO
2019-04-14 11:57:05,907 - ie_serving.models.model - INFO - Server start loading model: first_model
2019-04-14 11:57:06,481 - ie_serving.models.model - INFO - List of available versions for my_model model: [1]
2019-04-14 11:57:06,481 - ie_serving.models.model - INFO - Default version for my_model model is 1
2019-04-14 11:57:06,495 - ie_serving.server.start - INFO - Server listens on port 9001 and will be serving models: ['first_model']
```

Get the metadata for the model using the below command
```
python get_serving_meta.py --grpc_address 0.0.0.0 --grpc_port 9001 --model_name first_model --model_version 1
```
NOTE: `--model_name` should be same as mentioned while starting the container
The output should look something like
```
Getting model metadata for model: first_model
Inputs metadata:
        Input name: image_tensor; shape: [1, 3, 300, 300]; dtype: DT_FLOAT
Outputs metadata:
        Output name: DetectionOutput; shape: [1, 1, 100, 7]; dtype: DT_FLOAT
```
NOTE: Each model in IR format defines input and output tensors in the AI graph. By default OpenVINO™ model server is using tensors names as the input and output dictionary keys. The client is passing the input values to the gRPC request and reads the results by referring to the correspondent tensor names.

Therefore the name `image_tensor` and `DetectionOutput` has to be passed while making a request for inference.

You can perform and inference using the following command:
```
python grpc_client_vino.py --grpc_address 0.0.0.0 --grpc_port 9001 --input_name image_tensor \
--output_name DetectionOutput --model_name first_model
```
You should view the output like this:
```
output tensor shape(1, 1, 100, 7)
obj->info : 1 346 95 475 458    # class_id, xmin, ymin, xmax, ymax
obj->info : 1 146 146 348 478
duration: 39.32 msec
```
**Et. Voila!! Detection done!!!**

### What we achieved?
We just performed object detection use-case to demonstrate the power of OpenVINO serving. We exported our trained model to a format expected by OpenVINO serving, and used a client script that could request the model server for inference.

### References
https://github.com/IntelAI/OpenVINO-model-server/tree/master/example_client
