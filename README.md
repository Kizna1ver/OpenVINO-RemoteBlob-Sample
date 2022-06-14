# OpenVINO-RemoteBlobSample
A sample of [Direct Consuming of the NV12 VAAPI Video Decoder Surface on Linux](https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_GPU_RemoteTensor_API.html#examples)  with ffmpeg.
This project can be add to [OpenVINO Cpp sample directory](https://github.com/openvinotoolkit/openvino/tree/master/samples/cpp) and run. 


## System information (version)
- OpenVINO=> 2022.1
- Operating System / Platform => ubuntu20.04
- Intel Media Driver=> [21.3.5](https://github.com/intel/media-driver/releases/tag/intel-media-21.3.5)
- ffmpeg=> version:git-2022-05-19-94968db 
- ffmpeg install configuration =>
```
--extra-cflags=-I/opt/intel/openvino_2022/runtime/include/ie/ --extra-ldflags=-L/opt/intel/openvino_2022/runtime/lib/intel64/ --extra-ldflags=-L/opt/intel/openvino_2022/runtime/3rdparty/tbb/lib/ --enable-libopenvino --disable-lzma --enable-pic --enable-nonfree --disable-stripping --enable-hwaccel=h264_vaapi --enable-shared
```

