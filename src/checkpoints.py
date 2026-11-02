"""Strict loading of published two-, three- and four-channel auxiliary heads."""

import hashlib
from pathlib import Path


AUX_LAYOUTS = {
    2: ['road', 'building'],
    3: ['unused', 'road', 'building'],  # paper-models v6 and BCE+Dice v8.
    4: ['road', 'building', 'water', 'tree'],
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_state(path):
    import torch

    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict):
        for key in ('model_state_dict', 'state_dict'):
            if key in state:
                state = state[key]
                break
    if not isinstance(state, dict) or not state:
        raise ValueError('Expected a nonempty model state dictionary')
    if all(key.startswith('module.') for key in state):
        state = {key[7:]: value for key, value in state.items()}
    return state


def auxiliary_channels(state):
    counts = []
    for name in ('osm_side1', 'osm_side2', 'osm_head'):
        weight, bias = state.get(name + '.weight'), state.get(name + '.bias')
        if weight is None or bias is None:
            raise ValueError(f'Missing auxiliary projection: {name}')
        if len(weight.shape) != 4:
            raise ValueError(f'Invalid auxiliary projection shape: {name}')
        count = weight.shape[0]
        if tuple(weight.shape) != (count, 64, 1, 1) or tuple(bias.shape) != (count,):
            raise ValueError(f'Invalid auxiliary projection shape: {name}')
        counts.append(count)
    if len(set(counts)) != 1 or counts[0] not in AUX_LAYOUTS:
        raise ValueError(f'Inconsistent or unsupported auxiliary channel counts: {counts}')
    for name in ('side1', 'side2', 'head'):
        weight, bias = state.get(name + '.weight'), state.get(name + '.bias')
        if weight is None or bias is None or tuple(weight.shape) != (9, 64, 1, 1) or tuple(bias.shape) != (9,):
            raise ValueError(f'Expected nine semantic classes in {name}')
    return counts[0]


def load_model(weights, sam_checkpoint, variant='tiny', device='cpu'):
    from src.model import DualDecoder_SAM2UNet

    state = checkpoint_state(weights)
    channels = auxiliary_channels(state)
    model = DualDecoder_SAM2UNet(checkpoint_path=sam_checkpoint, variant=variant,
                                 num_classes=9, aux_channels=channels)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def checkpoint_metadata(weights, model):
    channels = model.osm_head.out_channels
    return {'path': str(Path(weights).resolve()), 'sha256': file_sha256(weights),
            'aux_channels': channels, 'aux_layout': AUX_LAYOUTS[channels]}
