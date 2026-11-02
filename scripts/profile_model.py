"""Forward latency, adapted from BCE+Dice v8 cell 14 (50 warm-ups, 300 timings)."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', required=True)
    parser.add_argument('--sam-checkpoint', required=True)
    parser.add_argument('--variant', choices=['tiny', 'small', 'base', 'large'], default='tiny')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--size', type=int, choices=[512], default=512)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iterations', type=int, default=300)
    parser.add_argument('--path', choices=['semantic', 'dual'], default='semantic')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.iterations < 1 or args.batch_size < 1:
        parser.error('Warm-up must be nonnegative; iterations and batch size must be positive')
    if args.device.startswith('cpu') and args.precision == 'fp16':
        parser.error('Use fp32 or bf16 on CPU')
    return args


def main():
    args = parse_args()
    import torch
    from src.checkpoints import load_model, checkpoint_metadata

    device = torch.device(args.device)
    model = load_model(args.weights, args.sam_checkpoint, args.variant, device)
    dummy = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    forward = model.forward_semantic if args.path == 'semantic' else model
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[args.precision]
    timings = []
    with torch.inference_mode(), torch.autocast(device.type, dtype=dtype, enabled=args.precision != 'fp32'):
        for _ in range(args.warmup):
            forward(dummy)
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
        for _ in range(args.iterations):
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    start.record()
                    forward(dummy)
                    end.record()
                    end.synchronize()
                    timings.append(start.elapsed_time(end))
            else:
                start_time = time.perf_counter()
                forward(dummy)
                timings.append((time.perf_counter() - start_time) * 1000)
    payload = {
        'configuration': vars(args), 'checkpoint': checkpoint_metadata(args.weights, model),
        'hardware': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
        'torch_version': torch.__version__,
        'forward_batch_latency_ms_mean': statistics.mean(timings),
        'forward_batch_latency_ms_std_population': statistics.pstdev(timings),
        'registered_parameters': sum(p.numel() for p in model.parameters()),
        'parameters_marked_trainable': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'auxiliary_projection_parameters': sum(p.numel() for layer in
            (model.osm_side1, model.osm_side2, model.osm_head) for p in layer.parameters()),
        'count_scope': 'Registration and requires_grad counts; not a census of parameters used in computation',
        'timing_scope': 'Forward only, preallocated device tensor; excludes IO, preprocessing, transfers and export',
        'timings_ms': timings,
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle, indent=2)
    print(f"Forward batch latency: {payload['forward_batch_latency_ms_mean']:.3f} ms")


if __name__ == '__main__':
    main()
