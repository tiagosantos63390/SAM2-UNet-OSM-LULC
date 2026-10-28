"""OpenEarthMap with uint8 0/1 auxiliary masks (255 and other values invalid)."""

from pathlib import Path

import numpy as np
import torch
import albumentations as A
import cv2
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

AUX_KEYS = ('osm', 'osm_bldg', 'osm_water', 'osm_tree')

def get_augmentation_pipeline(seed=None):
    pipeline = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.2, p=0.1),
    ], additional_targets={key: 'mask' for key in AUX_KEYS})
    if seed is not None and hasattr(pipeline, 'set_random_seed'):
        pipeline.set_random_seed(seed)
    return pipeline

def _read_split_file(split_files: dict, split: str):
    split_file = split_files['train'] if split == 'train' else split_files['val'] if split == 'val' else split_files['test']
    with open(split_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]

def _resolve_sample_paths(root_dir: Path, tile_name: str):
    tile_name = tile_name.strip()
    if not tile_name: return None, None
    filename = tile_name if Path(tile_name).suffix else f'{tile_name}.tif'
    stem = Path(filename).stem
    city = stem.rsplit('_', 1)[0] if '_' in stem else ''
    img_path = root_dir / city / 'images' / filename if city else root_dir / 'images' / filename
    lbl_path = root_dir / city / 'labels' / filename if city else root_dir / 'labels' / filename
    if img_path.is_file() and lbl_path.is_file(): return img_path, lbl_path
    return None, None

class ToTensor:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image = TF.to_tensor(image)
        label_np = np.array(label, dtype=np.uint8)
        if label_np.ndim == 3: label_np = label_np[:, :, 0]
        
        result = {'image': image, 'label': torch.from_numpy(label_np).long()}
        
        # Keep void pixels distinct from valid background and foreground.
        for key in AUX_KEYS:
            if key not in sample and key in ('osm_water', 'osm_tree'):
                continue
            if sample.get(key) is not None:
                result[key] = torch.from_numpy(np.array(sample[key], dtype=np.uint8)).long()
            else:
                result[key] = torch.full_like(result['label'], 255).long()
            result[key + '_valid'] = (result[key] == 0) | (result[key] == 1)
            result[key][~result[key + '_valid']] = 255
        return result

class Resize:
    def __init__(self, size): self.size = (size, size)
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        result = {
            'image': TF.resize(image, self.size, InterpolationMode.BILINEAR),
            'label': TF.resize(label, self.size, InterpolationMode.NEAREST)
        }
        for key in AUX_KEYS:
            if sample.get(key) is not None:
                result[key] = TF.resize(sample[key], self.size, InterpolationMode.NEAREST)
        return result

class Normalize:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        result = dict(sample)
        result['image'] = TF.normalize(image, self.mean, self.std)
        return result

class AlbumentationsTransform:
    def __init__(self, augmentation_pipeline): self.aug = augmentation_pipeline
    def __call__(self, sample):
        image_np = np.array(sample['image'], dtype=np.uint8)
        label_np = np.array(sample['label'], dtype=np.uint8)
        masks = {key: np.array(sample[key], dtype=np.uint8) for key in AUX_KEYS if sample.get(key) is not None}
        augmented = self.aug(image=image_np, mask=label_np, **masks)
        res = {
            'image': Image.fromarray(augmented['image']),
            'label': Image.fromarray(augmented['mask']),
        }
        for key in masks:
            res[key] = Image.fromarray(augmented[key])
        return res

def _get_osm_index(osm_dir, is_bldg=False, suffixes=None):
    index = {}
    if suffixes is None:
        suffixes = ('_buildings_mask', '_building_mask') if is_bldg else ('_roads_mask', '_highway_mask')
    if osm_dir and Path(osm_dir).exists():
        for p in sorted(Path(osm_dir).rglob('*')):
            # PNG exports are annotated previews, not supervision rasters.
            if not p.is_file() or p.suffix.lower() not in ('.tif', '.tiff'):
                continue
            tile = p.stem
            for suffix in suffixes:
                if tile.endswith(suffix):
                    tile = tile[:-len(suffix)]
                    break
            else:
                if not tile.rsplit('_', 1)[-1].isdigit():
                    continue
            if tile in index:
                raise ValueError(f'Ambiguous auxiliary mask for {tile}: {index[tile]} and {p}')
            index[tile] = p
    return index


def _read_aux_mask(path, size):
    if path is not None:
        try:
            with Image.open(path) as image:
                values = np.array(image)
            if values.ndim == 2 and values.shape == (size[1], size[0]):
                valid = (values == 0) | (values == 1)
                return Image.fromarray(np.where(valid, values, 255).astype(np.uint8))
        except (OSError, ValueError):
            pass
    return Image.new('L', size, 255)

class OpenEarthMapDataset(Dataset):
    def __init__(self, root_dir, osm_dir=None, bldg_dir=None, split='train', size=512, mode='train', seed=None, split_files=None, water_dir=None, tree_dir=None):
        self.root_dir = Path(root_dir)
        self.osm_dir = Path(osm_dir) if osm_dir else None
        self.bldg_dir = Path(bldg_dir) if bldg_dir else None
        self.water_dir = Path(water_dir) if water_dir else None
        self.tree_dir = Path(tree_dir) if tree_dir else None
        self.images, self.gts, self.osm_masks, self.bldg_masks = [], [], [], []
        self.water_masks, self.tree_masks = [], []
        
        osm_index = _get_osm_index(self.osm_dir, is_bldg=False)
        bldg_index = _get_osm_index(self.bldg_dir, is_bldg=True)
        water_index = _get_osm_index(self.water_dir, suffixes=('_water_mask',))
        tree_index = _get_osm_index(self.tree_dir, suffixes=('_trees_mask',))
        
        unresolved = []
        for tile_name in _read_split_file(split_files, split):
            image_path, label_path = _resolve_sample_paths(self.root_dir, tile_name)
            if image_path is not None:
                self.images.append(str(image_path))
                self.gts.append(str(label_path))
                
                stem = Path(tile_name if Path(tile_name).suffix else f'{tile_name}.tif').stem
                self.osm_masks.append(str(osm_index[stem]) if stem in osm_index else None)
                self.bldg_masks.append(str(bldg_index[stem]) if stem in bldg_index else None)
                self.water_masks.append(str(water_index[stem]) if stem in water_index else None)
                self.tree_masks.append(str(tree_index[stem]) if stem in tree_index else None)
            else:
                unresolved.append(tile_name)
        if unresolved:
            raise FileNotFoundError(
                f'{split} split has {len(unresolved)} unresolved image/label pairs under {self.root_dir}: '
                + ', '.join(unresolved)
            )
                
        pipeline = [Resize(size), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        if mode == 'train': pipeline.insert(1, AlbumentationsTransform(get_augmentation_pipeline(seed=seed)))
        self.transform = transforms.Compose(pipeline)
        
    def __getitem__(self, idx):
        sample = {'image': Image.open(self.images[idx]).convert('RGB'), 'label': Image.open(self.gts[idx]).convert('L')}
        sample['osm'] = _read_aux_mask(self.osm_masks[idx], sample['label'].size)
        sample['osm_bldg'] = _read_aux_mask(self.bldg_masks[idx], sample['label'].size)
        if self.water_dir is not None:
            sample['osm_water'] = _read_aux_mask(self.water_masks[idx], sample['label'].size)
        if self.tree_dir is not None:
            sample['osm_tree'] = _read_aux_mask(self.tree_masks[idx], sample['label'].size)
        return self.transform(sample)
        
    def __len__(self): return len(self.images)
