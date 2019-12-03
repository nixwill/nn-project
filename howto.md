```shell script
docker run --gpus all --rm -it -w /workspace -v "$PWD":/workspace -u "$(id -u)":"$(id -g)" tensorflow/tensorflow:2.0.0-gpu-py3 python runner.py
```
