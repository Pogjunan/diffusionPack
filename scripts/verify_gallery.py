"""Verify the published gallery without requiring an evaluation split or a GPU."""
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def check_hash(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f'Artifact hash mismatch: {path.name}')


def verify():
    folder = Path(__file__).resolve().parents[1] / 'docs/assets/light-cifar10-epoch20'
    manifest = json.loads((folder/'provenance.json').read_text())
    for item in manifest['artifacts']:
        check_hash(folder/item['path'], item['sha256'])
    samples = sorted(folder.glob('sample_*.png'))
    assert len(samples) == 30
    grid = np.asarray(Image.open(folder/'grid.png'))
    unique = set()
    for i, path in enumerate(samples):
        with Image.open(path) as image:
            assert image.size == (32, 32) and image.mode == 'RGB'
            pixels = np.asarray(image)
        row, col = divmod(i, 4)
        np.testing.assert_array_equal(pixels, grid[2+34*row:34+34*row, 2+34*col:34+34*col])
        unique.add(hashlib.sha256(pixels.tobytes()).hexdigest())
    assert len(unique) == 30
    try:
        check_hash(samples[0], '0'*64)
    except ValueError:
        negative_control = True
    else:
        raise AssertionError('Incorrect artifact hash was accepted')
    return {'status': 'PASS', 'artifacts_verified': len(manifest['artifacts']),
            'samples': len(samples), 'native_rgb_size': [32, 32],
            'unique_samples': len(unique), 'grid_matches_individuals': True,
            'wrong_hash_rejected': negative_control,
            'limitations': 'File integrity only. Does not measure image quality or prove training provenance.'}


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
