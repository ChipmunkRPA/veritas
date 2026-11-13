/* SPDX-License-Identifier: Apache-2.0 */
/* Write one line per decoded picture, followed by its exported vector blocks. */
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/motion_vector.h>

static int receive(AVCodecContext *codec, AVFrame *frame)
{
    int status;
    while ((status = avcodec_receive_frame(codec, frame)) >= 0) {
        if (frame->best_effort_timestamp == AV_NOPTS_VALUE || frame->decode_error_flags) {
            return AVERROR_INVALIDDATA;
        }
        AVFrameSideData *data = av_frame_get_side_data(frame, AV_FRAME_DATA_MOTION_VECTORS);
        const size_t count = data ? data->size / sizeof(AVMotionVector) : 0;
        printf("F %" PRId64 " %c %d %d %zu\n", frame->best_effort_timestamp,
               av_get_picture_type_char(frame->pict_type), frame->width, frame->height, count);
        const AVMotionVector *vectors = data ? (const AVMotionVector *)data->data : NULL;
        for (size_t i = 0; i < count; i++) {
            const AVMotionVector *v = &vectors[i];
            printf("V %d %d %u %u %" PRId32 " %" PRId32 " %u %" PRId32 "\n",
                   v->dst_x, v->dst_y, v->w, v->h, v->motion_x, v->motion_y, v->motion_scale, v->source);
        }
        av_frame_unref(frame);
    }
    return status == AVERROR(EAGAIN) || status == AVERROR_EOF ? 0 : status;
}

int main(int argc, char **argv)
{
    if (argc != 2) {
        return EXIT_FAILURE;
    }
    AVFormatContext *input = NULL;
    AVCodecContext *codec = NULL;
    AVPacket *packet = NULL;
    AVFrame *frame = NULL;
    AVDictionary *options = NULL;
    int result = EXIT_FAILURE;
    av_dict_set(&options, "protocol_whitelist", "file", 0);
    if (avformat_open_input(&input, argv[1], NULL, &options) < 0 || avformat_find_stream_info(input, NULL) < 0) {
        goto cleanup;
    }
    const AVCodec *decoder = NULL;
    const int index = av_find_best_stream(input, AVMEDIA_TYPE_VIDEO, -1, -1, &decoder, 0);
    if (index < 0 || input->streams[index]->codecpar->codec_id != AV_CODEC_ID_H264) {
        goto cleanup;
    }
    codec = avcodec_alloc_context3(decoder);
    if (!codec || avcodec_parameters_to_context(codec, input->streams[index]->codecpar) < 0) {
        goto cleanup;
    }
    codec->thread_count = 1;
    codec->flags2 |= AV_CODEC_FLAG2_EXPORT_MVS;
    packet = av_packet_alloc();
    frame = av_frame_alloc();
    if (!packet || !frame || avcodec_open2(codec, decoder, NULL) < 0) {
        goto cleanup;
    }
    const AVRational clock = input->streams[index]->time_base;
    printf("T %d %d\n", clock.num, clock.den);
    int status;
    while ((status = av_read_frame(input, packet)) >= 0) {
        if (packet->stream_index == index &&
            (avcodec_send_packet(codec, packet) < 0 || receive(codec, frame) < 0)) {
            goto cleanup;
        }
        av_packet_unref(packet);
    }
    if (status == AVERROR_EOF && avcodec_send_packet(codec, NULL) >= 0 && receive(codec, frame) >= 0) {
        result = EXIT_SUCCESS;
    }
cleanup:
    if (result != EXIT_SUCCESS) {
        fprintf(stderr, "Could not decode the complete H.264 motion-vector stream.\n");
    }
    av_dict_free(&options);
    av_frame_free(&frame);
    av_packet_free(&packet);
    avcodec_free_context(&codec);
    avformat_close_input(&input);
    return result;
}
