"""OpenEarthMap dataset with OSM road/building auxiliary masks, taken
line-for-line from the final training notebook
(notebooks/train_paper_models_seeds42-44.ipynb)."""

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

def get_augmentation_pipeline(seed=None):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.2, p=0.1),
    ], additional_targets={'osm': 'mask', 'osm_bldg': 'mask'})

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
    if img_path.exists() and lbl_path.exists(): return img_path, lbl_path
    return None, None

class ToTensor:
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        osm, osm_bldg = sample.get('osm', None), sample.get('osm_bldg', None)
        image = TF.to_tensor(image)
        label_np = np.array(label, dtype=np.uint8)
        if label_np.ndim == 3: label_np = label_np[:, :, 0]
        
        result = {'image': image, 'label': torch.from_numpy(label_np).long()}
        
        # Missing masks are safely filled with 255
        if osm is not None:
            osm_np = np.array(osm, dtype=np.uint8)
            result['osm'] = torch.from_numpy((osm_np > 127).astype(np.uint8)).long()
        else:
            result['osm'] = torch.full_like(result['label'], 255).long()
            
        if osm_bldg is not None:
            bldg_np = np.array(osm_bldg, dtype=np.uint8)
            result['osm_bldg'] = torch.from_numpy((bldg_np > 127).astype(np.uint8)).long()
        else:
            result['osm_bldg'] = torch.full_like(result['label'], 255).long()
            
        return result

class Resize:
    def __init__(self, size): self.size = (size, size)
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        osm, osm_bldg = sample.get('osm', None), sample.get('osm_bldg', None)
        result = {
            'image': TF.resize(image, self.size, InterpolationMode.BILINEAR),
            'label': TF.resize(label, self.size, InterpolationMode.NEAREST)
        }
        if osm is not None: result['osm'] = TF.resize(osm, self.size, InterpolationMode.NEAREST)
        if osm_bldg is not None: result['osm_bldg'] = TF.resize(osm_bldg, self.size, InterpolationMode.NEAREST)
        return result

class Normalize:
    def __init__(self, mean, std): self.mean, self.std = mean, std
    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        result = {'image': TF.normalize(image, self.mean, self.std), 'label': label}
        if 'osm' in sample: result['osm'] = sample['osm']
        if 'osm_bldg' in sample: result['osm_bldg'] = sample['osm_bldg']
        return result

class AlbumentationsTransform:
    def __init__(self, augmentation_pipeline): self.aug = augmentation_pipeline
    def __call__(self, sample):
        image_np = np.array(sample['image'], dtype=np.uint8)
        label_np = np.array(sample['label'], dtype=np.uint8)
        osm_np = np.array(sample.get('osm', Image.new('L', sample['label'].size)), dtype=np.uint8)
        bldg_np = np.array(sample.get('osm_bldg', Image.new('L', sample['label'].size)), dtype=np.uint8)
        
        augmented = self.aug(image=image_np, mask=label_np, osm=osm_np, osm_bldg=bldg_np)
        res = {
            'image': Image.fromarray(augmented['image']),
            'label': Image.fromarray(augmented['mask']),
        }
        if 'osm' in sample: res['osm'] = Image.fromarray(augmented['osm'])
        if 'osm_bldg' in sample: res['osm_bldg'] = Image.fromarray(augmented['osm_bldg'])
        return res

GLOBAL_OSM_INDEX = None
GLOBAL_BLDG_INDEX = None

def _get_osm_index(osm_dir, is_bldg=False):
    global GLOBAL_OSM_INDEX, GLOBAL_BLDG_INDEX
    if is_bldg and GLOBAL_BLDG_INDEX is not None: return GLOBAL_BLDG_INDEX
    if not is_bldg and GLOBAL_OSM_INDEX is not None: return GLOBAL_OSM_INDEX
    
    index = {}
    if osm_dir and osm_dir.exists():
        for p in osm_dir.rglob('*'):
            if p.is_file():
                index[p.stem] = p
                index[p.parent.name] = p
                if p.stem.endswith('_highway_mask'):
                    index[p.stem.replace('_highway_mask', '')] = p
                if p.stem.endswith('_building_mask'):
                    index[p.stem.replace('_building_mask', '')] = p
    
    if is_bldg: GLOBAL_BLDG_INDEX = index
    else: GLOBAL_OSM_INDEX = index
    return index

class OpenEarthMapDataset(Dataset):
    def __init__(self, root_dir, osm_dir=None, bldg_dir=None, split='train', size=512, mode='train', seed=None, split_files=None):
        self.root_dir = Path(root_dir)
        self.osm_dir = Path(osm_dir) if osm_dir else None
        self.bldg_dir = Path(bldg_dir) if bldg_dir else None
        self.images, self.gts, self.osm_masks, self.bldg_masks = [], [], [], []
        
        osm_index = _get_osm_index(self.osm_dir, is_bldg=False)
        bldg_index = _get_osm_index(self.bldg_dir, is_bldg=True)
        
        for tile_name in _read_split_file(split_files, split):
            image_path, label_path = _resolve_sample_paths(self.root_dir, tile_name)
            if image_path is not None:
                self.images.append(str(image_path))
                self.gts.append(str(label_path))
                
                stem = Path(tile_name if Path(tile_name).suffix else f'{tile_name}.tif').stem
                self.osm_masks.append(str(osm_index[stem]) if stem in osm_index else None)
                self.bldg_masks.append(str(bldg_index[stem]) if stem in bldg_index else None)
                
        pipeline = [Resize(size), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        if mode == 'train': pipeline.insert(1, AlbumentationsTransform(get_augmentation_pipeline(seed=seed)))
        self.transform = transforms.Compose(pipeline)
        
    def __getitem__(self, idx):
        sample = {'image': Image.open(self.images[idx]).convert('RGB'), 'label': Image.open(self.gts[idx]).convert('L')}
        if self.osm_masks[idx] is not None: sample['osm'] = Image.open(self.osm_masks[idx]).convert('L')
        if self.bldg_masks[idx] is not None: sample['osm_bldg'] = Image.open(self.bldg_masks[idx]).convert('L')
        return self.transform(sample)
        
    def __len__(self): return len(self.images)
