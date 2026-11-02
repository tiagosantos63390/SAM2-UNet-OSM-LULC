import contextlib
from importlib.util import find_spec
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def require_dependencies(*names):
    missing = [name for name in names if find_spec(name) is None]
    if missing:
        raise unittest.SkipTest('Missing optional dependencies: ' + ', '.join(missing))


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        require_dependencies('torch', 'sam2', 'torchvision', 'numpy', 'cv2',
                             'albumentations', 'PIL', 'hydra', 'omegaconf', 'iopath')
        import torch
        from src import model
        from src import checkpoints
        cls.torch = torch
        cls.model_module = model
        cls.checkpoints = checkpoints

    def test_checkpoint_layouts_and_wrappers(self):
        torch = self.torch
        for channels in (2, 3, 4):
            state = {}
            for name in ('side1', 'side2', 'head', 'osm_side1', 'osm_side2', 'osm_head'):
                count = channels if name.startswith('osm_') else 9
                state[name + '.weight'] = torch.zeros(count, 64, 1, 1)
                state[name + '.bias'] = torch.zeros(count)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'model.pth'
                for wrapper in (None, 'state_dict', 'model_state_dict'):
                    prefixed = {'module.' + key: value for key, value in state.items()}
                    torch.save(prefixed if wrapper is None else {wrapper: prefixed}, path)
                    loaded = self.checkpoints.checkpoint_state(path)
                    self.assertEqual(set(loaded), set(state))
                    self.assertEqual(self.checkpoints.auxiliary_channels(loaded), channels)
            state['osm_side2.weight'] = torch.zeros(channels + 1, 64, 1, 1)
            with self.assertRaises(ValueError):
                self.checkpoints.auxiliary_channels(state)

    def test_semantic_forward_skips_auxiliary_projections(self):
        torch = self.torch
        nn = torch.nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.stem = nn.Conv2d(3, 4, 1)
                self.image_encoder = nn.Module()
                self.image_encoder.trunk = nn.Module()
                self.image_encoder.trunk.blocks = nn.ModuleList()
                self.sam_mask_decoder = nn.Module()
                self.sam_mask_decoder.conv_s0 = nn.Conv2d(4, 4, 1)
                self.sam_mask_decoder.conv_s1 = nn.Conv2d(4, 4, 1)

            def forward_image(self, x):
                f = torch.nn.functional.avg_pool2d(self.stem(x), 4)
                f2 = torch.nn.functional.avg_pool2d(f, 2)
                f3 = torch.nn.functional.avg_pool2d(f2, 2)
                return {'backbone_fpn': [self.sam_mask_decoder.conv_s0(f),
                                         self.sam_mask_decoder.conv_s1(f2), f3]}

        for channels in (2, 3, 4):
            with patch.object(self.model_module, '_build_encoder_with_fallback', return_value=Encoder()), \
                 patch.object(self.model_module, 'IMAGE_SIZE', 32):
                model = self.model_module.DualDecoder_SAM2UNet(aux_channels=channels).eval()
            x = torch.randn(1, 3, 32, 32)
            with torch.no_grad():
                expected = model(x)[0]
                with contextlib.ExitStack() as stack:
                    for name in ('osm_side1', 'osm_side2', 'osm_head'):
                        stack.enter_context(patch.object(getattr(model, name), 'forward',
                                                         side_effect=AssertionError('Auxiliary projection called')))
                    projection = model.encoder.sam_mask_decoder.conv_s0
                    hook = stack.enter_context(patch.object(projection, 'forward', wraps=projection.forward))
                    actual = model.forward_semantic(x)
                    hook.assert_called_once()
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_raster_scale_edges_and_nodata(self):
        require_dependencies('rasterio', 'tqdm')
        import numpy as np
        import rasterio
        from rasterio.transform import from_origin
        from scripts import predict_large_tif as prediction
        torch = self.torch

        class Model:
            def forward_semantic(self, x):
                logits = torch.zeros(1, 9, 512, 512)
                logits[:, 0] = 1
                return logits

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'input.tif'
            output = str(Path(directory) / 'prediction')
            image = np.full((3, 600, 600), 80, dtype=np.uint8)
            image[:, :30, :30] = 255
            with rasterio.open(source, 'w', driver='GTiff', width=600, height=600,
                               count=3, dtype='uint8', crs='EPSG:32629',
                               transform=from_origin(400000, 4500000, .7, .9), nodata=255) as dst:
                dst.write(image)
            with patch.object(prediction, 'load_model', return_value=Model()):
                mask = prediction.predict_mask_on_large_tif(source, 'unused', 'unused', output, device='cpu')
            with rasterio.open(output + '.tif') as result:
                self.assertEqual(result.crs.to_epsg(), 32629)
                self.assertEqual(result.transform, from_origin(400000, 4500000, .5, .5))
                self.assertEqual((result.width, result.height), (840, 1080))
                self.assertEqual(result.nodata, 255)
                self.assertEqual(int(mask[0, 0]), 255)
                self.assertTrue(np.all(mask[100:-1, 100:-1] == 0))
            with self.assertRaises(ValueError):
                prediction.target_grid(SimpleNamespace(crs=rasterio.crs.CRS.from_epsg(4326)), .5)


if __name__ == '__main__':
    unittest.main()
