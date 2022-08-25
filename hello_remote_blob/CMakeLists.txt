# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

find_package(PkgConfig REQUIRED)
pkg_check_modules(ffmpeg REQUIRED IMPORTED_TARGET libavcodec libavformat libavutil libavfilter libswresample libswscale)
ie_add_sample(NAME hello_remote_blob
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp"
              DEPENDENCIES OpenCL)
ie_add_sample(NAME hello_remote_blob_y
              SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/va_y_channel.cpp"
              DEPENDENCIES OpenCL)
target_link_libraries(hello_remote_blob PRIVATE PkgConfig::ffmpeg)
target_link_libraries(hello_remote_blob_y PRIVATE PkgConfig::ffmpeg)



