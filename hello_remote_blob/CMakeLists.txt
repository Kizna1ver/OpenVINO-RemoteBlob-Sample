# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

ie_add_sample(NAME hello_remote_blob
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES OpenCL avcodec avformat avutil swresample swscale swscale avfilter)

