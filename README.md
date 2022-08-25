# OpenVINO-RemoteBlobSample
A sample of [Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API.html#examples)  with ffmpeg.
This project follows [OpenVINO Cpp sample](https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp).


## System information (version)
- openvino: recent version in my forked [repo](https://github.com/Kizna1ver/openvino/tree/feature/CAPI-VA-Remoteblob) from branch master.  
- Operating System / Platform : ubuntu20.04
- Intel Media Driver=> [21.3.5](https://github.com/intel/media-driver/releases/tag/intel-media-21.3.5)
- ffmpeg: commit [81ebf40](https://github.com/Kizna1ver/FFmpeg) in branch master.
- ffmpeg install configuration =>
```
--extra-cflags=-I/opt/intel/openvino_2022/runtime/include/ie/ --extra-ldflags=-L/opt/intel/openvino_2022/runtime/lib/intel64/ --extra-ldflags=-L/opt/intel/openvino_2022/runtime/3rdparty/tbb/lib/ --enable-libopenvino --disable-lzma --enable-pic --enable-nonfree --disable-stripping --enable-hwaccel=h264_vaapi --enable-shared
```

## Run
Prepare the IR Model ssd300
```
omz_downloader --name ssd300
omz_converter --name ssd300
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