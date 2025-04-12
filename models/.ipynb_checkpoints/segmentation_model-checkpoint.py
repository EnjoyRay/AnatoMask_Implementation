import torch
import torch.nn as nn
import torch.nn.functional as F
from decoder3D import LightDecoder
from encoder3D import SparseEncoder
from AnatoMask import SparK
from STUNet_head import STUNet

class SegmentationModel(nn.Module):
    def __init__(self, input_size=(768, 768, 90), num_classes=16, checkpoint_path=None):
        super().__init__()
        
        # 构建 STUNet encoder
        cnn = STUNet(
            input_channels=1,
            num_classes=1,  # 注意这里仍为 1，不影响 encoder
            depth=[1, 1, 1, 1, 1, 1],
            dims=[32, 64, 128, 256, 512, 512],
            pool_op_kernel_sizes=[[2, 2, 2]] * 5,
            conv_kernel_sizes=[[3, 3, 3]] * 6,
            enable_deep_supervision=False
        )
        encoder = SparseEncoder(cnn=cnn, input_size=input_size, sbn=False)

        # 构建分割用 decoder
        decoder = LightDecoder(
            up_sample_ratio=encoder.downsample_ratio,
            width=encoder.enc_feat_map_chs[-1],
            sbn=False,
            out_channel=num_classes
        )

        # 封装为 SparK
        self.model = SparK(sparse_encoder=encoder, dense_decoder=decoder, mask_ratio=0.0)

        # 如果提供 checkpoint，则加载 encoder 权重
        if checkpoint_path:
            print(f"🧠 Loading encoder weights from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict

            # 只加载 encoder
            encoder_dict = {k: v for k, v in state_dict.items() if "sparse_encoder.sp_cnn" in k}
            missing, unexpected = self.model.load_state_dict(encoder_dict, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")

    def forward(self, x):
        B, _, H, W, D = x.shape
        p = self.model.downsample_ratio

        # padding 以保证能被 p 整除
        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        pad_d = (p - D % p) % p
        x = F.pad(x, (0, pad_d, 0, pad_w, 0, pad_h), mode='constant', value=0)

        H_, W_, D_ = x.shape[2:]
        full_mask = torch.ones((B, 1, H_ // p, W_ // p, D_ // p), dtype=torch.bool, device=x.device)

        out = self.model(x, active_b1ff=full_mask)

        # 移除 padding
        if pad_h > 0:
            out = out[:, :, :-pad_h, :, :]
        if pad_w > 0:
            out = out[:, :, :, :-pad_w, :]
        if pad_d > 0:
            out = out[:, :, :, :, :-pad_d]

        return out  # shape: [B, num_classes, H, W, D]
