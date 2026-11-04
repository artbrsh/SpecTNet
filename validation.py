"""
Classification evaluation and GPU performance metrics.
"""

from contextlib import nullcontext
import math
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

from models import prepare_for_inference


@torch.no_grad()
def validate(args, config, data_loader, model, logger):
    """Return sample-weighted top-1, top-5 (%) and cross-entropy loss."""
    model.eval()
    # Never mutate the architecture during validation within a training run.
    if getattr(config, 'EVAL_MODE', False):
        model = prepare_for_inference(model, config)
    device = next(model.parameters()).device
    # Sum once across ranks, including ranks with an empty validation partition.
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    for images, targets in data_loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=config.AMP_ENABLE and device.type == 'cuda'):
            outputs = model(images)
        loss = F.cross_entropy(outputs.float(), targets, reduction='sum')
        predictions = outputs.topk(min(5, outputs.shape[-1]), dim=1).indices
        correct = predictions.eq(targets[:, None])
        totals[0] += loss.double()
        totals[1] += correct[:, 0].sum()
        totals[2] += correct.any(dim=1).sum()
        totals[3] += targets.numel()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
    loss_sum, top1, top5, count = totals.tolist()
    if count == 0:
        raise ValueError('Validation dataset is empty')
    acc1, acc5, loss = 100.0 * top1 / count, 100.0 * top5 / count, loss_sum / count
    logger.info(f'Validation: {int(count)} images | Acc@1 {acc1:.3f} | Acc@5 {acc5:.3f} | loss {loss:.4f}')
    return acc1, acc5, loss


def _gpu_model(model):
    model = model.module if hasattr(model, 'module') else model
    parameter = next(model.parameters())
    if parameter.device.type != 'cuda':
        raise ValueError('Performance metrics require a model on CUDA')
    if parameter.dtype != torch.float32:
        raise ValueError('Use FP32 model weights; select AMP through precision="amp"')
    return model.eval(), parameter.device


def _gpu_input(input_shape, device):
    if len(input_shape) != 4 or any(int(size) <= 0 for size in input_shape):
        raise ValueError('input_shape must contain positive (batch, channels, height, width)')
    return torch.randn(*input_shape, device=device)


def _runtime_model(model, precision, runtime, compile_mode, compile_fullgraph):
    if precision not in ('fp32', 'amp'):
        raise ValueError('precision must be "fp32" or "amp"')
    if runtime not in ('eager', 'compiled'):
        raise ValueError('runtime must be "eager" or "compiled"')
    if runtime == 'compiled':
        return torch.compile(model, mode=compile_mode, fullgraph=compile_fullgraph)
    return model


def _autocast(precision):
    return torch.amp.autocast('cuda') if precision == 'amp' else nullcontext()


@torch.inference_mode()
def compute_macs(model, logger, input_shape=(1, 3, 224, 224)):
    """Count per-image operations using fvcore, reporting unsupported operators."""
    from fvcore.nn import FlopCountAnalysis, parameter_count

    model, device = _gpu_model(model)
    inputs = _gpu_input(input_shape, device)
    analyzer = FlopCountAnalysis(model, inputs)
    analyzer.unsupported_ops_warnings(False)
    analyzer.uncalled_modules_warnings(False)
    macs = analyzer.total() / input_shape[0]
    unsupported = dict(analyzer.unsupported_ops())
    params = parameter_count(model)['']
    result = {
        'macs': macs,
        'gmacs': macs / 1e9,
        'flops': 2.0 * macs,
        'gflops': 2.0 * macs / 1e9,
        'params': params,
        'params_m': params / 1e6,
        'batch': input_shape[0],
        'unsupported_ops': unsupported,
    }
    logger.info(f'Counted GMACs/image {result["gmacs"]:.6f} | '
                f'derived GFLOPs/image {result["gflops"]:.6f} | params {params}')
    if unsupported:
        logger.warning(f'Incomplete operation count; unsupported operators: {unsupported}')
    return result


@torch.inference_mode()
def compute_gpu_latency(model, logger, input_shape, precision, runtime, settings):
    """Measure batch latency with CUDA events; exclude allocation and warmup."""
    model, device = _gpu_model(model)
    inputs = _gpu_input(input_shape, device)
    run_model = _runtime_model(model, precision, runtime,
                               settings.COMPILE_MODE, settings.COMPILE_FULLGRAPH)
    with torch.cuda.device(device):
        for _ in range(settings.LATENCY_WARMUP_ITERS):
            with _autocast(precision):
                run_model(inputs)
        torch.cuda.synchronize(device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times_ms = []
        for _ in range(settings.LATENCY_MEASURE_ITERS):
            torch.cuda.synchronize(device)
            start.record()
            with _autocast(precision):
                run_model(inputs)
            end.record()
            torch.cuda.synchronize(device)
            times_ms.append(start.elapsed_time(end))

    times_ms = np.asarray(times_ms, dtype=np.float64)
    result = {
        'batch': input_shape[0], 'runtime': runtime, 'precision': precision,
        'measure_iters': len(times_ms),
        'mean_ms': float(times_ms.mean()), 'std_ms': float(times_ms.std()),
        'min_ms': float(times_ms.min()), 'max_ms': float(times_ms.max()),
        'p50_ms': float(np.percentile(times_ms, 50)),
        'p90_ms': float(np.percentile(times_ms, 90)),
        'p95_ms': float(np.percentile(times_ms, 95)),
        'p99_ms': float(np.percentile(times_ms, 99)),
    }
    logger.info(f'GPU latency | {runtime} | {precision} | {result}')
    return result


@torch.inference_mode()
def throughput(model, logger, batch_size, input_shape, precision, runtime, settings):
    """Measure image throughput from synchronized per-batch wall-clock timings."""
    model, device = _gpu_model(model)
    inputs = _gpu_input((batch_size, *input_shape[1:]), device)
    run_model = _runtime_model(model, precision, runtime,
                               settings.COMPILE_MODE, settings.COMPILE_FULLGRAPH)
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < settings.THROUGHPUT_WARMUP_SECONDS:
            with _autocast(precision):
                run_model(inputs)
        torch.cuda.synchronize(device)

        timings, elapsed_total = [], 0.0
        while elapsed_total < settings.THROUGHPUT_MEASURE_SECONDS:
            torch.cuda.synchronize(device)
            start_time = time.perf_counter()
            with _autocast(precision):
                run_model(inputs)
            torch.cuda.synchronize(device)
            duration = time.perf_counter() - start_time
            timings.append(duration)
            elapsed_total += duration

    timings = np.asarray(timings, dtype=np.float64)
    per_iteration = batch_size / timings
    result = {
        'batch': batch_size, 'runtime': runtime, 'precision': precision,
        'mean_throughput_fps': float(per_iteration.mean()),
        'std_throughput_fps': float(per_iteration.std()),
        'global_throughput_fps': float(batch_size * len(timings) / timings.sum()),
        'num_iters': len(timings), 'total_time_s': float(timings.sum()),
    }
    logger.info(f'GPU throughput | {runtime} | {precision} | {result}')
    return result


def run_metrics(model, logger, config):
    """Measure one model on one GPU, applying its inference fusion when available."""
    settings = config.METRICS
    precision = settings.PRECISION if config.AMP_ENABLE else 'fp32'
    runtime = settings.RUNTIME
    if precision not in ('fp32', 'amp') or runtime not in ('eager', 'compiled'):
        raise ValueError('METRICS requires PRECISION fp32/amp and RUNTIME eager/compiled')
    for name, value, minimum in (
        ('DATA.BATCH_SIZE', config.DATA.BATCH_SIZE, 1),
        ('DATA.IMG_SIZE', config.DATA.IMG_SIZE, 1),
        ('MODEL.SPECTNET.IN_CHANS', config.MODEL.SPECTNET.IN_CHANS, 1),
        ('MACS_BATCH_SIZE', settings.MACS_BATCH_SIZE, 1),
        ('LATENCY_BATCH_SIZE', settings.LATENCY_BATCH_SIZE, 1),
        ('LATENCY_WARMUP_ITERS', settings.LATENCY_WARMUP_ITERS, 0),
        ('LATENCY_MEASURE_ITERS', settings.LATENCY_MEASURE_ITERS, 1),
    ):
        if type(value) is not int or value < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    for name in ('THROUGHPUT_WARMUP_SECONDS', 'THROUGHPUT_MEASURE_SECONDS'):
        value = getattr(settings, name)
        if (type(value) not in (int, float) or not math.isfinite(value)
                or value < 0 or (name == 'THROUGHPUT_MEASURE_SECONDS' and value == 0)):
            raise ValueError(f'METRICS.{name} must be finite and '
                             'nonnegative (strictly positive for measurement)')
    for name in ('COMPILE_FULLGRAPH', 'FUSE', 'CUDNN_BENCHMARK'):
        if type(getattr(settings, name)) is not bool:
            raise ValueError(f'METRICS.{name} must be a boolean')
    for name in ('TF32_MATMUL', 'TF32_CUDNN'):
        if getattr(settings, name) is not None and type(getattr(settings, name)) is not bool:
            raise ValueError(f'METRICS.{name} must be a boolean or None')
    if settings.COMPILE_MODE not in (
            'default', 'reduce-overhead', 'max-autotune', 'max-autotune-no-cudagraphs'):
        raise ValueError('Unsupported METRICS.COMPILE_MODE')
    batch_size = config.DATA.BATCH_SIZE
    dimensions = (config.MODEL.SPECTNET.IN_CHANS, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE)
    input_shape = (settings.LATENCY_BATCH_SIZE, *dimensions)
    model, device = _gpu_model(model)
    torch.backends.cudnn.benchmark = settings.CUDNN_BENCHMARK
    if settings.TF32_MATMUL is not None:
        torch.backends.cuda.matmul.allow_tf32 = settings.TF32_MATMUL
    if settings.TF32_CUDNN is not None:
        torch.backends.cudnn.allow_tf32 = settings.TF32_CUDNN
    fuse = getattr(model, 'fuse_for_inference', None)
    if settings.FUSE and callable(fuse):
        fused_model = fuse()
        if fused_model is not None:
            model = fused_model
    model = prepare_for_inference(model, config)
    model.requires_grad_(False)
    logger.info(f'Metrics | GPU {torch.cuda.get_device_name(device)} | torch {torch.__version__} | '
                f'input {input_shape} | throughput batch {batch_size} | '
                f'runtime {runtime} | precision {precision} | fusion {settings.FUSE and callable(fuse)} | '
                f'merged S1 {config.INFERENCE.MERGE_STAGE1} | '
                f'TF32 matmul {torch.backends.cuda.matmul.allow_tf32} | '
                f'TF32 cuDNN {torch.backends.cudnn.allow_tf32}')
    return {
        'operations': compute_macs(model, logger, (settings.MACS_BATCH_SIZE, *dimensions)),
        'latency': compute_gpu_latency(model, logger, input_shape, precision, runtime, settings),
        'throughput': throughput(model, logger, batch_size, input_shape, precision, runtime, settings),
    }
