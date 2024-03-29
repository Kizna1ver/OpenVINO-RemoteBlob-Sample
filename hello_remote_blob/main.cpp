#include <unistd.h>
#include <stdlib.h>

// #include <gflags/gflags.h>
#include <iostream>

#include <vector>
#include <memory>
#include <string>
#include <ctime>
#include <algorithm>
#include <fstream>

#include "../classification_results.h"
#include "openvino/openvino.hpp"
#include "openvino/runtime/intel_gpu/properties.hpp"
#include "openvino/runtime/intel_gpu/ocl/va.hpp"
#include "openvino/core/preprocess/pre_post_process.hpp"

#include <condition_variable>

#include <stdio.h>

extern "C" {
#include "config.h"
#include "libavformat/avformat.h"
#include "libavformat/avio.h"

#include "libavcodec/avcodec.h"

#include "libavutil/buffer.h"
#include "libavutil/error.h"
#include "libavutil/hwcontext.h"
// #include "libavutil/hwcontext_qsv.h" // no mf
#include <libavutil/hwcontext_vaapi.h>
#include "libavutil/mem.h"
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libavutil/avassert.h>
#include <libavutil/imgutils.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavfilter/buffersink.h>
#include <libavfilter/buffersrc.h>
#include "va/va.h"
}



AVFilterContext *buffersink_ctx;
AVFilterContext *buffersrc_ctx;
AVFilterGraph *filter_graph;
AVFormatContext *input_ctx = NULL;
AVCodecContext *decoder_ctx = NULL;
VASurfaceID surface_id;

using namespace std;
using namespace ov::preprocess;
// using namespace InferenceEngine;
clock_t start1,end1,start2,end2;
bool buildnetwork= 1 ;

#if defined(ENABLE_UNICODE_PATH_SUPPORT) && defined(_WIN32)
#define tcout std::wcout
#define file_name_t std::wstring
#define WEIGHTS_EXT L".bin"
#define imread_t imreadW
#define ClassificationResult_t ClassificationResultW
#else
#define tcout std::cout
#define file_name_t std::string
#define WEIGHTS_EXT ".bin"
#define imread_t cv::imread
#define ClassificationResult_t ClassificationResult
#endif


// In ov2.0, resize preprocess performs equally as filter scale_vaapi.
// const char *filter_descr = "scale_vaapi=684:372";
const char *filter_descr = "scale_vaapi=iw:ih"; // keep origin size
static int init_filters(const char *filters_descr,AVBufferRef  *hw_frames_ctx);
static AVBufferRef *hw_device_ctx = NULL;
static enum AVPixelFormat hw_pix_fmt;
static FILE *output_file = NULL;
int flag_filter=0;
int iflag=0;
int i,j=0; //judge frame is empty or not
static int hw_decoder_init(AVCodecContext *ctx, const enum AVHWDeviceType type)
{
    int err = 0;

    if ((err = av_hwdevice_ctx_create(&hw_device_ctx, type,
                                      NULL, NULL, 0)) < 0) {
        fprintf(stderr, "Failed to create specified HW device.\n");
        return err;
    }
    ctx->hw_device_ctx = av_buffer_ref(hw_device_ctx);

    return err;
}

static enum AVPixelFormat get_hw_format(AVCodecContext *ctx,
                                        const enum AVPixelFormat *pix_fmts)
{
    const enum AVPixelFormat *p;

    for (p = pix_fmts; *p != -1; p++) {
        if (*p == hw_pix_fmt)
            return *p;
    }

    fprintf(stderr, "Failed to get HW surface format.\n");
    return AV_PIX_FMT_NONE;
}


static int decode_write(AVCodecContext *avctx, AVPacket *packet,AVFormatContext *input_ctx,VADisplay va_display)
{
    AVFrame *frame = NULL, *sw_frame = NULL, *filt_frame=NULL;
    AVFrame *tmp_frame = NULL;
    uint8_t *buffer = NULL;
    int size;
    int ret = 0;

  
    // initialize  filter
    if(flag_filter==0&&(avctx->hw_frames_ctx!=NULL)){
        if ((ret = init_filters(filter_descr,avctx->hw_frames_ctx)) < 0)                 
            fprintf(stderr, "The filter initilion fails !!!!!!!");
        flag_filter = flag_filter +1;
    }

    ret = avcodec_send_packet(avctx, packet);
    if (ret < 0) {
        fprintf(stderr, "Error during decoding\n");
        return ret;
    }

    while (1) {
        if (!(frame = av_frame_alloc()) || !(sw_frame = av_frame_alloc())||!(filt_frame = av_frame_alloc())) {
            fprintf(stderr, "Can not alloc frame\n");
            ret = AVERROR(ENOMEM);
           // goto fail;
        }

        ret = avcodec_receive_frame(avctx, frame);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
            av_frame_free(&frame);
            av_frame_free(&sw_frame);
            return 0;
        } else if (ret < 0) {
            fprintf(stderr, "Error while decoding\n");
            //goto fail;
        }
        end1=clock();
        surface_id=(VASurfaceID)(uintptr_t)frame->data[3];
        std::cout<<"sur_id before scale vaapi:"<<surface_id<<std::endl;
        fprintf(stderr, "No.%d frame \n", i);
        //goto filter
        if(flag_filter==1){
            if (av_buffersrc_add_frame_flags(buffersrc_ctx, frame, AV_BUFFERSRC_FLAG_KEEP_REF) < 0){
                av_log(NULL, AV_LOG_ERROR, "Error while feeding the filtergraph\n");
                break;
            }
            ret = av_buffersink_get_frame(buffersink_ctx, filt_frame);
            if(ret == AVERROR(EAGAIN) || ret == AVERROR_EOF)
                break;        

        }

        surface_id=(VASurfaceID)(uintptr_t)filt_frame->data[3];
        std::cout<<"sur_id:"<<surface_id<<std::endl;
        fprintf(stderr, "No.%d frame \n", i);
        i++;

    fail:
        av_frame_free(&filt_frame);
        av_frame_free(&frame);
        av_frame_free(&sw_frame);
        av_freep(&buffer);
        if (ret < 0)
            return ret;

    }
    return 0;
}

static int init_filters(const char *filters_descr,AVBufferRef *hw_frames_ctx)
{
    char args[512];
    int ret = 0;
    const AVFilter *buffersrc  = avfilter_get_by_name("buffer");
    const AVFilter *buffersink = avfilter_get_by_name("buffersink");
    AVFilterInOut *outputs = avfilter_inout_alloc();
    AVFilterInOut *inputs  = avfilter_inout_alloc();
    AVRational time_base = input_ctx->streams[0]->time_base;
    enum AVPixelFormat pix_fmts[] = { AV_PIX_FMT_VAAPI, AV_PIX_FMT_NV12, AV_PIX_FMT_NONE };
    AVBufferSrcParameters *par = av_buffersrc_parameters_alloc();
    filter_graph = avfilter_graph_alloc();
    if (!outputs || !inputs || !filter_graph) {
        ret = AVERROR(ENOMEM);
        goto end;
    }
 
    /* buffer video source: the decoded frames from the decoder will be inserted here. */
    snprintf(args, sizeof(args),
            "video_size=%dx%d:pix_fmt=%d:time_base=%d/%d:pixel_aspect=%d/%d",
            decoder_ctx->width, decoder_ctx->height, AV_PIX_FMT_VAAPI,
            time_base.num, time_base.den,
            decoder_ctx->sample_aspect_ratio.num, decoder_ctx->sample_aspect_ratio.den);


    ret = avfilter_graph_create_filter(&buffersrc_ctx, buffersrc, "in",
                                       args, NULL, filter_graph);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot create buffer source\n");
        goto end;
    }

    par->hw_frames_ctx =  hw_frames_ctx;   
    ret = av_buffersrc_parameters_set(buffersrc_ctx,par);
    
    if (ret< 0)
        goto end;

    /* buffer video sink: to terminate the filter chain. */
    ret = avfilter_graph_create_filter(&buffersink_ctx, buffersink, "out",
                                       NULL, NULL, filter_graph);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot create buffer sink\n");
        goto end;
    }

    ret = av_opt_set_int_list(buffersink_ctx, "pix_fmts", pix_fmts,
                              AV_PIX_FMT_NONE, AV_OPT_SEARCH_CHILDREN);
    if (ret < 0) {
        av_log(NULL, AV_LOG_ERROR, "Cannot set output pixel format\n");
        goto end;
    }

    /*
     * Set the endpoints for the filter graph. The filter_graph will
     * be linked to the graph described by filters_descr.
     */

    /*
     * The buffer source output must be connected to the input pad of
     * the first filter described by filters_descr; since the first
     * filter input label is not specified, it is set to "in" by
     * default.
     */
    outputs->name       = av_strdup("in");
    outputs->filter_ctx = buffersrc_ctx;
    outputs->pad_idx    = 0;
    outputs->next       = NULL;

    /*
     * The buffer sink input must be connected to the output pad of
     * the last filter described by filters_descr; since the last
     * filter output label is not specified, it is set to "out" by
     * default.
     */
    inputs->name       = av_strdup("out");
    inputs->filter_ctx = buffersink_ctx;
    inputs->pad_idx    = 0;
    inputs->next       = NULL;

    if ((ret = avfilter_graph_parse_ptr(filter_graph, filters_descr,
                                    &inputs, &outputs, NULL)) < 0)
        goto end;

    if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0)
        goto end;

end:
    av_freep(&par);
    avfilter_inout_free(&inputs);
    avfilter_inout_free(&outputs);
    return ret;
}

int main(int argc, char **argv)
{
    // Core ie; //initialize Core API 
    int video_stream, ret;
    AVStream *video = NULL;
    VADisplay display=NULL;
    const AVCodec *decoder = NULL;
    AVPacket packet;
    AVVAAPIDeviceContext * va_device_data;
    AVHWDeviceContext * hw_device_data;
    enum AVHWDeviceType type;
    int currentFrame=0,prevFrame=1,transfer=NULL;

    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input file> <output file>\n", argv[0]);
        return -1;
    }

    type = av_hwdevice_find_type_by_name("vaapi");
    if (type == AV_HWDEVICE_TYPE_NONE) {
        //fprintf(stderr, "Device type %s is not supported.\n", argv[1]);
        fprintf(stderr, "Available device types:");
        while((type = av_hwdevice_iterate_types(type)) != AV_HWDEVICE_TYPE_NONE)
            fprintf(stderr, " %s", av_hwdevice_get_type_name(type));
        fprintf(stderr, "\n");
        return -1;
    }
    /* open the input file */
    if (avformat_open_input(&input_ctx, argv[1], NULL, NULL) != 0) {
        fprintf(stderr, "Cannot open input file '%s',please input .mp4 file\n", argv[1]);
        return -1;
    }

    if (avformat_find_stream_info(input_ctx, NULL) < 0) {
        fprintf(stderr, "Cannot find input stream information.\n");
        return -1;
    }

    /* find the video stream information */
    ret = av_find_best_stream(input_ctx, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (ret < 0) {
        fprintf(stderr, "Cannot find a video stream in the input file\n");
        return -1;
    }
    video_stream = ret;

    for (i = 0;; i++) {
        const AVCodecHWConfig *config = avcodec_get_hw_config(decoder, i);
        if (!config) {
            fprintf(stderr, "Decoder %s does not support device type %s.\n",
                    decoder->name, av_hwdevice_get_type_name(type));
            return -1;
        }
        if (config->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX &&
            config->device_type == type) {
            hw_pix_fmt = config->pix_fmt;
            break;
        }
    }

    if (!(decoder_ctx = avcodec_alloc_context3(decoder)))
        return AVERROR(ENOMEM);

    video = input_ctx->streams[video_stream];
    if (avcodec_parameters_to_context(decoder_ctx, video->codecpar) < 0)
        return -1;

    decoder_ctx->get_format  = get_hw_format;

    if (hw_decoder_init(decoder_ctx, type) < 0)
        return -1;

    if ((ret = avcodec_open2(decoder_ctx, decoder, NULL)) < 0) {
        fprintf(stderr, "Failed to open codec for stream #%u\n", video_stream);
        return -1;
    }

    hw_device_data = (AVHWDeviceContext *)hw_device_ctx ->data;
    va_device_data = (AVVAAPIDeviceContext *)hw_device_data ->hwctx;
    display= va_device_data -> display;
    
    /* open the file to dump raw data */
    output_file = fopen(argv[2], "w+");

    // --------------------------------OpenVINO Part-----------------------------------------------

    // --------------------------- 1. Load inference engine instance -------------------------------------

    size_t input_height = 224;
    size_t input_width = 224;
    // initialize the core and load the network
    ov::Core core;
    auto model = core.read_model("../public/vgg16/FP32/vgg16.xml");
    std::string input_tensor_name = model->input().get_any_name();
    std::string output_tensor_name = model->output().get_any_name();
    std::cout << "input name: " << input_tensor_name << std::endl;
    std::cout << "Output name: " << output_tensor_name << std::endl;
    // printf("Output name:%s \n", output_tensor_name);  

    printf("Preprocessing...\n");  
    auto p = PrePostProcessor(model);
    p.input().tensor().set_element_type(ov::element::u8)
                    .set_color_format(ov::preprocess::ColorFormat::NV12_TWO_PLANES, {"y", "uv"})
                    .set_memory_type(ov::intel_gpu::memory_type::surface)
                    .set_spatial_static_shape(input_height, input_width);
            
    p.input().preprocess()
                    // .convert_element_type(ov::element::f32)
                    // see https://github.com/openvinotoolkit/openvino/pull/7508/files#diff-977744dce511c9709f290043976db52b438c0aa4a8b4017d9bb261296b9885e6R675
                    // convert to BGR, default element type is f32
                    .convert_color(ov::preprocess::ColorFormat::BGR)
                    .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR);
    p.input().model().set_layout("NCHW");
    model = p.build();
    printf("Build finished\n");  

    // create the shared context object
    auto shared_va_context = ov::intel_gpu::ocl::VAContext(core, display);
    // compile model within a shared context
    auto compiled_model = core.compile_model(model, shared_va_context);
    // auto compiled_model = core.compile_model(model, "GPU");

    printf("Compile model finished\n");  
    // input_tensor_name = model->input().get_any_name();
    output_tensor_name = model->output().get_any_name();
    // std::cout << "input name" << input_tensor_name << std::endl;
    std::cout << "Output name" << output_tensor_name << std::endl;
    std::string labelFileName = "../public/vgg16/FP32/vgg16.label";
    std::vector<std::string> labels;
    std::ifstream inputFile;
    inputFile.open(labelFileName, std::ios::in);
    if (inputFile.is_open()) {
        std::string strLine;
        while (std::getline(inputFile, strLine)) {
            // trim(strLine);
            labels.push_back(strLine);
        }
    }
    auto input = model->get_parameters().at(0);
    auto infer_request = compiled_model.create_infer_request();
    printf("Ready to inference\n");

    vector<string> input_names;
    auto input_nodes = model->inputs();
    for (auto input_node:input_nodes){
        std::string name = input_node.get_any_name();
        input_names.push_back(name);
        std::cout << "input node name:" << input_node.get_any_name() << std::endl;
    }

    auto context = compiled_model.get_context(); // default is clcontext
    // ov::RemoteContext context = core.get_default_context(contexts[0]);
    auto output_port = compiled_model.output();
    // codes for output tensor in GPU.
    // auto out_tensor = context.create_tensor(output_port.get_element_type(), output_port.get_shape());
    // std::cout << out_tensor.get_device_name() << std::endl;
    // infer_request.set_tensor(output_port, out_tensor);

    // auto va_out_blob = shared_va_context.create_tensor

    /* actual decoding and dump the raw data */
    while (ret >= 0) {
        if ((ret = av_read_frame(input_ctx, &packet)) < 0)
            break;
        j = i;
        if (video_stream == packet.stream_index)
            ret = decode_write(decoder_ctx, &packet, input_ctx, display); // decode one frame

        auto nv12_blob = shared_va_context.create_tensor_nv12(input_height, input_width, surface_id);

        // --------------------------- 6. Prepare input --------------------------------------------------------
        infer_request.set_tensor(input_names[0], nv12_blob.first);
        infer_request.set_tensor(input_names[1], nv12_blob.second);
        // ov::RemoteTensor remote_tensor = context.create_tensor(output_port.get_element_type(), output_port.get_shape());
        // infer_request.set_tensor(output_port, remote_tensor);
        printf("y surface_id: %d, uv sid: %d", VASurfaceID(nv12_blob.first), VASurfaceID(nv12_blob.second));

        // --------------------------- 7. Do inference --------------------------------------------------------
        if(j != i){  //if this frame is not empty and the inference complete
            // infer_request.start_async();
            // infer_request.wait();
            infer_request.infer();
        }
        ov::Tensor out_tensor = infer_request.get_tensor(output_tensor_name);
        // printf("output surface_id: %d", VASurfaceID(output));

        ClassificationResult classification_result(out_tensor, {"Noooo"}, 1, 10, labels);
        classification_result.show();

        av_packet_unref(&packet);
    }
    end1=clock();

    
    /* flush the decoder */
    packet.data = NULL;
    packet.size = 0;
    ret = decode_write(decoder_ctx, &packet, input_ctx, display);
    av_packet_unref(&packet);
    // // print the FPS
    // double endtime=(double)(end1-start1)/CLOCKS_PER_SEC;
    // std::cout<<"Total FPS:"<<1/(endtime/1500)<<"FPS"<< std::endl;
    if (output_file)
        fclose(output_file);
    avcodec_free_context(&decoder_ctx);
    avformat_close_input(&input_ctx);
    av_buffer_unref(&hw_device_ctx);

    return 0;
}

