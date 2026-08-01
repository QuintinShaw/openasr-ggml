#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#include <stdint.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of MTLCommandBuffer used to submit a graph for processing
#define GGML_METAL_MAX_COMMAND_BUFFERS 8

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    ggml_metal_event_t ev_cpy; // for async copies

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_fusion;
    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;
    int debug_fusion;

    // how many times a given op was fused
    uint64_t fuse_cnt[GGML_OP_COUNT];

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers
    int n_nodes_0;      // number of nodes submitted by the main thread
    int n_nodes_1;      // remaining number of nodes submitted by the n_cb threads
    int n_nodes_per_cb;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // error state - set when a command buffer fails during completion
    // once set, new submissions fail until the backend is recreated
    bool has_error;

    // Compute-scoped cooperative cancellation. The abort flag is host-visible;
    // each command buffer owns a separate indirect-dispatch argument buffer so
    // the backend can still encode its command buffers on parallel host tasks.
    ggml_abort_callback abort_callback;
    void * abort_callback_data;
    id<MTLBuffer> abort_flag;
    id<MTLBuffer> abort_indirect_args[GGML_METAL_MAX_COMMAND_BUFFERS + 1];
    dispatch_semaphore_t abort_monitor_stop;
    dispatch_semaphore_t abort_monitor_done;
    bool abort_monitor_active;
};

static bool ggml_metal_abort_monitor_stop(ggml_metal_t ctx);

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
    // Show all the Metal device instances in the system
    NSArray * devices = MTLCopyAllDevices();
    for (id<MTLDevice> device in devices) {
        GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
    }
    [devices release]; // since it was created by a *Copy* C method
#endif

    // init context
    ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

    // TODO: would it be better to have one queue for the backend and one queue for the device?
    //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
    //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
    id<MTLCommandQueue> queue = ggml_metal_device_get_queue(dev);
    if (queue == nil) {
        GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
        return NULL;
    }

    res->dev = dev;
    res->lib = ggml_metal_device_get_library(dev);
    if (res->lib == NULL) {
        GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
        GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

        res->lib = ggml_metal_library_init(dev);
        if (res->lib == NULL) {
            GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

            free(res);

            return NULL;
        }
    }

    res->ev_cpy = ggml_metal_device_event_init(dev);

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

    res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

    res->use_fusion      = getenv("GGML_METAL_FUSION_DISABLE") == nil;
    res->use_concurrency = getenv("GGML_METAL_CONCURRENCY_DISABLE") == nil;

    {
        const char * val = getenv("GGML_METAL_GRAPH_DEBUG");
        res->debug_graph = val ? atoi(val) : 0;
    }

    {
        const char * val = getenv("GGML_METAL_FUSION_DEBUG");
        res->debug_fusion = val ? atoi(val) : 0;
    }

    res->use_graph_optimize = true;

    if (getenv("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
        res->use_graph_optimize = false;
    }

    memset(res->fuse_cnt, 0, sizeof(res->fuse_cnt));

    GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, res->use_fusion         ? "true" : "false");
    GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
    GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

    res->capture_compute = 0;
    res->capture_started = false;
    res->capture_scope = nil;

    {
        const char * val = getenv("GGML_METAL_CAPTURE_COMPUTE");
        if (val) {
            res->capture_compute = atoi(val);
        }
    }

    res->has_error = false;

    res->gf = nil;
    res->encode_async = nil;
    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->cmd_bufs[i].obj = nil;
    }

    res->cmd_bufs_ext = [[NSMutableArray alloc] init];

    res->cmd_buf_last = nil;

    res->pipelines_ext = ggml_metal_pipelines_init();

    res->abort_flag = [device newBufferWithLength:sizeof(uint32_t)
                                          options:MTLResourceStorageModeShared];
    for (int i = 0; i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        res->abort_indirect_args[i] = [device newBufferWithLength:sizeof(struct ggml_metal_cancel_dispatch_args)
                                                           options:MTLResourceStorageModePrivate];
    }
    if (res->abort_flag == nil) {
        GGML_LOG_ERROR("%s: failed to allocate Metal cancellation flag\n", __func__);
        ggml_metal_free(res);
        return NULL;
    }
    for (int i = 0; i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (res->abort_indirect_args[i] == nil) {
            GGML_LOG_ERROR("%s: failed to allocate Metal cancellation indirect buffer\n", __func__);
            ggml_metal_free(res);
            return NULL;
        }
    }

    return res;
}

void ggml_metal_free(ggml_metal_t ctx) {
    GGML_LOG_INFO("%s: deallocating\n", __func__);

    if (ctx->abort_monitor_active) {
        dispatch_semaphore_signal(ctx->abort_monitor_stop);
        dispatch_semaphore_wait(ctx->abort_monitor_done, DISPATCH_TIME_FOREVER);
        dispatch_release(ctx->abort_monitor_stop);
        dispatch_release(ctx->abort_monitor_done);
        ctx->abort_monitor_active = false;
    }

    [ctx->abort_flag release];
    for (int i = 0; i <= GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        [ctx->abort_indirect_args[i] release];
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ctx->debug_fusion > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);
        for (int i = 0; i < GGML_OP_COUNT; i++) {
            if (ctx->fuse_cnt[i] == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_op_name((enum ggml_op) i), ctx->fuse_cnt[i]);
        }
    }

    Block_release(ctx->encode_async);

    //[ctx->queue release]; // [TAG_QUEUE_PER_BACKEND]

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

enum ggml_status ggml_metal_synchronize(ggml_metal_t ctx) {
    if (ctx->has_error) {
        ggml_metal_abort_monitor_stop(ctx);
        return GGML_STATUS_BACKEND_POISONED;
    }

    // wait for any backend operations to finish
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }

    const bool aborted = ggml_metal_abort_monitor_stop(ctx);

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }
                ctx->has_error = true;
                return GGML_STATUS_EXECUTION_FAILED;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    GGML_LOG_ERROR("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return GGML_STATUS_EXECUTION_FAILED;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    return aborted ? GGML_STATUS_ABORTED : GGML_STATUS_SUCCESS;
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

static enum ggml_status ggml_metal_submit_blit(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    if (cmd_buf == nil) { ctx->has_error = true; return GGML_STATUS_EXECUTION_FAILED; }
    [cmd_buf commit];
    [ctx->cmd_bufs_ext addObject:cmd_buf];
    ctx->cmd_buf_last = cmd_buf;
    [cmd_buf retain];
    return GGML_STATUS_SUCCESS;
}

enum ggml_status ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (ctx->has_error) return GGML_STATUS_BACKEND_POISONED;
    @autoreleasepool {
        id<MTLBuffer> src = [(id<MTLDevice>) ggml_metal_device_get_obj(ctx->dev) newBufferWithBytes:data length:size options:MTLResourceStorageModeShared];
        struct ggml_metal_buffer_id dst = ggml_metal_get_buffer_id(tensor);
        if (src == nil || dst.metal == nil) { [src release]; ctx->has_error = true; return GGML_STATUS_EXECUTION_FAILED; }
        id<MTLCommandBuffer> cmd = [(id<MTLCommandQueue>) ggml_metal_device_get_queue(ctx->dev) commandBuffer];
        id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
        if (enc == nil) { [src release]; ctx->has_error = true; return GGML_STATUS_EXECUTION_FAILED; }
        [enc copyFromBuffer:src sourceOffset:0 toBuffer:dst.metal destinationOffset:dst.offs + offset size:size];
        [enc endEncoding]; [src release]; return ggml_metal_submit_blit(ctx, cmd);
    }
}

enum ggml_status ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (ctx->has_error) return GGML_STATUS_BACKEND_POISONED;
    @autoreleasepool {
        id<MTLBuffer> dst = [(id<MTLDevice>) ggml_metal_device_get_obj(ctx->dev) newBufferWithBytesNoCopy:data length:size options:MTLResourceStorageModeShared deallocator:nil];
        struct ggml_metal_buffer_id src = ggml_metal_get_buffer_id(tensor);
        if (dst == nil || src.metal == nil) { [dst release]; ctx->has_error = true; return GGML_STATUS_EXECUTION_FAILED; }
        id<MTLCommandBuffer> cmd = [(id<MTLCommandQueue>) ggml_metal_device_get_queue(ctx->dev) commandBuffer]; id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
        if (enc == nil) { [dst release]; ctx->has_error = true; return GGML_STATUS_EXECUTION_FAILED; }
        [enc copyFromBuffer:src.metal sourceOffset:src.offs + offset toBuffer:dst destinationOffset:0 size:size];
        [enc endEncoding]; [dst release]; return ggml_metal_submit_blit(ctx, cmd);
    }
}

enum ggml_status ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ctx_src->has_error || ctx_dst->has_error) return GGML_STATUS_BACKEND_POISONED;
    @autoreleasepool {
        struct ggml_metal_buffer_id src_id = ggml_metal_get_buffer_id(src), dst_id = ggml_metal_get_buffer_id(dst);
        if (src_id.metal == nil || dst_id.metal == nil) { ctx_src->has_error = true; return GGML_STATUS_EXECUTION_FAILED; }
        id<MTLCommandBuffer> cmd = [(id<MTLCommandQueue>) ggml_metal_device_get_queue(ctx_src->dev) commandBuffer]; id<MTLBlitCommandEncoder> enc = [cmd blitCommandEncoder];
        if (enc == nil) { ctx_src->has_error = true; return GGML_STATUS_EXECUTION_FAILED; }
        [enc copyFromBuffer:src_id.metal sourceOffset:src_id.offs toBuffer:dst_id.metal destinationOffset:dst_id.offs size:ggml_nbytes(src)];
        [enc endEncoding]; ggml_metal_event_encode_signal(ggml_metal_get_ev_cpy(ctx_src), cmd);
        enum ggml_status status = ggml_metal_submit_blit(ctx_src, cmd);
        return status == GGML_STATUS_SUCCESS ? ggml_metal_event_wait(ctx_dst, ggml_metal_get_ev_cpy(ctx_src)) : status;
    }
}

static void ggml_metal_abort_monitor_run(void * opaque) {
    ggml_metal_t ctx = opaque;
    volatile uint32_t * device_flag = (volatile uint32_t *) [ctx->abort_flag contents];
    while (true) {
        if (ctx->abort_callback(ctx->abort_callback_data)) {
            __atomic_store_n(device_flag, 1, __ATOMIC_RELEASE);
            break;
        }
        const dispatch_time_t deadline = dispatch_time(DISPATCH_TIME_NOW, NSEC_PER_MSEC);
        if (dispatch_semaphore_wait(ctx->abort_monitor_stop, deadline) == 0) {
            break;
        }
    }
    dispatch_semaphore_signal(ctx->abort_monitor_done);
}

static void ggml_metal_abort_monitor_start(ggml_metal_t ctx) {
    GGML_ASSERT(!ctx->abort_monitor_active);
    GGML_ASSERT(ctx->abort_callback != NULL);
    volatile uint32_t * device_flag = (volatile uint32_t *) [ctx->abort_flag contents];
    __atomic_store_n(device_flag, 0, __ATOMIC_RELEASE);
    ctx->abort_monitor_stop = dispatch_semaphore_create(0);
    ctx->abort_monitor_done = dispatch_semaphore_create(0);
    ctx->abort_monitor_active = true;
    dispatch_async_f(
        dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0),
        ctx,
        ggml_metal_abort_monitor_run);
}

static bool ggml_metal_abort_monitor_stop(ggml_metal_t ctx) {
    if (!ctx->abort_monitor_active) {
        return false;
    }
    dispatch_semaphore_signal(ctx->abort_monitor_stop);
    dispatch_semaphore_wait(ctx->abort_monitor_done, DISPATCH_TIME_FOREVER);
    volatile uint32_t * device_flag = (volatile uint32_t *) [ctx->abort_flag contents];
    const bool aborted = __atomic_load_n(device_flag, __ATOMIC_ACQUIRE) != 0
        || ctx->abort_callback(ctx->abort_callback_data);
    dispatch_release(ctx->abort_monitor_stop);
    dispatch_release(ctx->abort_monitor_done);
    ctx->abort_monitor_stop = nil;
    ctx->abort_monitor_done = nil;
    ctx->abort_monitor_active = false;
    return aborted;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_BACKEND_POISONED;
    }

    if (ctx->abort_callback != NULL) {
        // Metal cannot cancel a committed command buffer. A tiny gate kernel
        // before every normal dispatch samples a host-visible abort flag and
        // turns the remaining work into zero-sized indirect dispatches. This
        // preserves prompt cancellation without fragmenting a graph into
        // hundreds of command buffers and host synchronizations.
        ggml_metal_abort_monitor_start(ctx);
    }

    // number of nodes encoded by the main thread (empirically determined)
    const int n_main = MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    const int n_cb = ctx->n_cb;

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_nodes_0 are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    @autoreleasepool {
        ctx->gf = gf;

        ctx->n_nodes_0 = MIN(n_main, gf->n_nodes);
        ctx->n_nodes_1 = gf->n_nodes - ctx->n_nodes_0;

        ctx->n_nodes_per_cb = (ctx->n_nodes_1 + ctx->n_cb - 1) / ctx->n_cb;

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            ctx->capture_compute = -1;

            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                if (![[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error]) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s'\n", __func__, [[error localizedDescription] UTF8String]);
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            [cmd_buf enqueue];

            ctx->encode_async(n_cb);
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            [cmd_buf enqueue];

            // update the pointer to the last queued command buffer
            // this is needed to implement synchronize()
            ctx->cmd_buf_last = cmd_buf;
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    return GGML_STATUS_FAILED;
                }

            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            ctx->capture_started = false;
        }
    }

    // Encoding rejection is the first observed terminal error. Do not turn it
    // into BACKEND_POISONED until the caller has seen this result once.
    return ctx->has_error ? GGML_STATUS_EXECUTION_FAILED : GGML_STATUS_SUCCESS;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        ggml_graph_optimize(gf);
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_set_abort_callback(
        ggml_metal_t ctx, ggml_abort_callback abort_callback, void * abort_callback_data) {
    GGML_ASSERT(!ctx->abort_monitor_active);
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = abort_callback_data;
}

enum ggml_status ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    if (ctx->has_error) {
        return GGML_STATUS_BACKEND_POISONED;
    }
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        if (cmd_buf == nil) {
            ctx->has_error = true;
            return GGML_STATUS_EXECUTION_FAILED;
        }

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
        return GGML_STATUS_SUCCESS;
    }
}

enum ggml_status ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    if (ctx->has_error) {
        return GGML_STATUS_BACKEND_POISONED;
    }
    @autoreleasepool {
        id<MTLCommandQueue> queue = ggml_metal_device_get_queue(ctx->dev);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        if (cmd_buf == nil) {
            ctx->has_error = true;
            return GGML_STATUS_EXECUTION_FAILED;
        }

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
        return GGML_STATUS_SUCCESS;
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        if (ctx->n_cb > 2) {
            GGML_LOG_WARN("%s: n_cb = %d, using n_cb > 2 is not recommended and can degrade the performance in some cases\n", __func__, n_cb);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb;

        const int n_nodes_0 = ctx->n_nodes_0;
        const int n_nodes_1 = ctx->n_nodes_1;

        const int n_nodes_per_cb = ctx->n_nodes_per_cb;

        int idx_start = 0;
        int idx_end   = n_nodes_0;

        if (cb_idx < n_cb_l) {
            idx_start = n_nodes_0 + (                                         (cb_idx + 0) * n_nodes_per_cb);
            idx_end   = n_nodes_0 + (MIN((cb_idx == n_cb_l - 1) ? n_nodes_1 : (cb_idx + 1) * n_nodes_per_cb, n_nodes_1));
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            idx_start,
            idx_end,
            ctx->use_fusion,
            ctx->use_concurrency && !ctx->abort_monitor_active,
            ctx->capture_started,
            ctx->debug_graph,
            ctx->debug_fusion);

        if (ctx->abort_monitor_active) {
            ggml_metal_op_set_cancel_buffers(
                ctx_op,
                (struct ggml_metal_buffer_id) { ctx->abort_flag, 0 },
                (struct ggml_metal_buffer_id) { ctx->abort_indirect_args[cb_idx], 0 });
        }

        bool encoded = true;
        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                encoded = false;
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);

        if (encoded) {
            [cmd_buf commit];
        } else {
            ctx->has_error = true;
        }
    });
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
