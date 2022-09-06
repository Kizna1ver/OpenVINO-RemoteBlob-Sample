# OpenVINO-RemoteBlobSample
A sample of [Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API.html#examples)  with ffmpeg.
This project follows [OpenVINO Cpp sample](https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp).


## System information (version)
Refer to this [repo](https://github.com/Kizna1ver/FFmpeg-OpenVINO-VAAPI/tree/main) to install prerequisite software environment.

## Run classification on VGG16 model
Prepare the IR Model vgg16 by omz_downloader.
```
omz_downloader --name vgg16
omz_converter --name vgg16
```
Build
```
mkdir build && cd build
cmake ..
make -j8
```
Run
```
./intel64/hello_remote_blob /workspace/input.mp4 /workspace/output.mp4
```