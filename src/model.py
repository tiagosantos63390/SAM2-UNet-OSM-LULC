"""Dual-decoder SAM2-UNet model, taken line-for-line from the final training
notebook (notebooks/train_paper_models_seeds42-44.ipynb)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from sam2.build_sam import build_sam2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'sam2' package is required. Install it from "
        "https://github.com/facebookresearch/sam2 (pip install -e .)"
    ) from exc

IMAGE_SIZE = 512

VARIANT_CFG = {
    'tiny': 'sam2_hiera_t.yaml',
    'small': 'sam2_hiera_s.yaml',
    'base': 'sam2_hiera_b+.yaml',
    'large': 'sam2_hiera_l.yaml',
}

def _build_encoder_with_fallback(cfg_name, checkpoint_path):
    candidates = [cfg_name, f'sam2_configs/{cfg_name}', f'configs/sam2/{cfg_name}', f'configs/{cfg_name}']
    for c in candidates:
        try: return build_sam2(c, checkpoint_path)
        except Exception: pass
    raise RuntimeError(f'Failed to build SAM2 with config {cfg_name}.')

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y, diff_x = x2.size()[2] - x1.size()[2], x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))

class Adapter(nn.Module):
    def __init__(self, blk):
        super().__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt = nn.Sequential(nn.Linear(dim, 32), nn.GELU(), nn.Linear(32, dim), nn.GELU())
    def forward(self, x): return self.block(x + self.prompt(x))

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFBModified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFBModified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))
        return self.relu(x_cat + self.conv_res(x))

class DualDecoder_SAM2UNet(nn.Module):
    def __init__(self, checkpoint_path=None, variant='tiny', num_classes=9):
        super().__init__()
        self.encoder = _build_encoder_with_fallback(VARIANT_CFG[variant], checkpoint_path)
        
        backbone_channels = self._infer_backbone_channels()
        for param in self.encoder.parameters(): param.requires_grad = False
        if hasattr(self.encoder, 'sam_mask_decoder'):
            for param in self.encoder.sam_mask_decoder.parameters(): param.requires_grad = True
            
        self.rfb1 = RFBModified(backbone_channels[0], 64)
        self.rfb2 = RFBModified(backbone_channels[1], 64)
        self.rfb3 = RFBModified(backbone_channels[2], 64)
        
        self.up1 = Up(128, 64)
        self.up2 = Up(128, 64)
        
        self.side1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.side2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # OSM auxiliary head: channel 0 = roads, channel 1 = buildings
        self.osm_side1 = nn.Conv2d(64, 2, kernel_size=1)
        self.osm_side2 = nn.Conv2d(64, 2, kernel_size=1)
        self.osm_head = nn.Conv2d(64, 2, kernel_size=1)
        
        self.upsample8 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        
        blocks = []
        for block in self.encoder.image_encoder.trunk.blocks: blocks.append(Adapter(block))
        self.encoder.image_encoder.trunk.blocks = nn.Sequential(*blocks)

    @torch.no_grad()
    def _infer_backbone_channels(self):
        was_training = self.encoder.training
        self.encoder.eval()
        first_param = next(self.encoder.parameters())
        probe = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE, device=first_param.device)
        feats = self.encoder.forward_image(probe)['backbone_fpn']
        if was_training: self.encoder.train()
        return [f.shape[1] for f in feats]

    def forward(self, x):
        feats = self.encoder.forward_image(x)['backbone_fpn']
        x1, x2, x3 = self.rfb1(feats[0]), self.rfb2(feats[1]), self.rfb3(feats[2])
        
        x = self.up1(x3, x2)
        out2_oem, out2_osm = self.side1(x), self.osm_side1(x)
        
        x = self.up2(x, x1)
        out3_oem, out3_osm = self.side2(x), self.osm_side2(x)
        
        out_oem = self.upsample4(self.head(x)) + self.upsample8(out2_oem) + self.upsample4(out3_oem)
        out_osm = self.upsample4(self.osm_head(x)) + self.upsample8(out2_osm) + self.upsample4(out3_osm)
        
        return out_oem, out_osm
