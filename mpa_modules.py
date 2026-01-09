import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.nn.modules import Conv, C2f, autopad

# -----------------------------------------------------------------------
# 1. BHCA: Batch Normalization and HardSwish Coordinate Attention
# Source: Fig. 3 [cite: 213] and Section 4.1 [cite: 98, 99]
# -----------------------------------------------------------------------
class BHCA(nn.Module):
    def __init__(self, c1, reduction=32):
        super().__init__()
        self.bn = nn.BatchNorm2d(c1)
        self.act = nn.Hardswish() # [cite: 98]
        
        # Coordinate Attention components [cite: 99]
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, c1 // reduction)
        
        self.conv1 = nn.Conv2d(c1, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act1 = nn.ReLU() # [cite: 226]
        
        self.conv_h = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, c1, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        
        # BH Process: Batch Norm -> HardSwish [cite: 222]
        x = self.act(self.bn(x))
        
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.act1(self.bn1(self.conv1(y)))
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.sigmoid(self.conv_h(x_h))
        a_w = self.sigmoid(self.conv_w(x_w))

        # Re-apply attention to the original identity input?
        # Paper says: "refined by BN and activation layers, and fused with original features" [cite: 100, 239]
        # Looking at Fig 3 BHCA diagram, the attention weights multiply the feature map *after* BN/Hardswish.
        # But usually CA multiplies the original input. Fig 3 shows the output of BHCA entering the multiplier.
        out = identity * a_h * a_w
        return out

# -----------------------------------------------------------------------
# 2. LKDCG: Large Kernel Depthwise Convolution Group
# Source: Fig. 3 [cite: 204] and Section 4.1 [cite: 91, 95]
# -----------------------------------------------------------------------
class LKDCG(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # Branch 1: 3x3 DW -> 1x1 -> 3x3 DW [cite: 91, 92]
        self.b1_conv1 = Conv(c1, c1, 3, g=c1) # DW
        self.b1_conv2 = Conv(c1, c1, 1)       # PW
        self.b1_conv3 = Conv(c1, c1, 3, g=c1) # DW
        
        # Branch 2: 3x3 DW -> 1x1 -> 7x7 DW [cite: 92]
        self.b2_conv1 = Conv(c1, c1, 3, g=c1) # DW
        self.b2_conv2 = Conv(c1, c1, 1)       # PW
        self.b2_conv3 = Conv(c1, c1, 7, g=c1) # DW (Large Kernel)
        
        # Fusion [cite: 93, 94]
        self.silu = nn.SiLU()
        self.final_conv1 = Conv(c1, c1, 1)
        self.final_conv2 = Conv(c1, c2, 3, g=c1) # Adjust to c2

    def forward(self, x):
        # Parallel branches
        b1 = self.b1_conv3(self.b1_conv2(self.b1_conv1(x)))
        b2 = self.b2_conv3(self.b2_conv2(self.b2_conv1(x)))
        
        # Element-wise add -> SiLU -> 1x1 -> 3x3
        y = b1 + b2
        y = self.silu(y)
        y = self.final_conv2(self.final_conv1(y))
        return y

# -----------------------------------------------------------------------
# 3. MPCA Module (Combines LKDCG and BHCA)
# Source: Fig. 3 [cite: 233]
# -----------------------------------------------------------------------
class MPCA(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.lkdcg = LKDCG(c1, c2)
        self.bhca = BHCA(c1) # Input to BHCA is raw features

    def forward(self, x):
        # Fig 3 shows Input splitting into LKDCG and BHCA
        # Then element-wise multiplication at the end
        feat_lkdcg = self.lkdcg(x)
        feat_bhca = self.bhca(x)
        
        # Note: LKDCG might change channels to c2. BHCA keeps channels c1.
        # Assuming inside C2f bottleneck c1 == c2 usually.
        return feat_lkdcg * feat_bhca

# -----------------------------------------------------------------------
# 4. C2f_MPCA: Replaces Bottleneck with MPCA
# Source: Fig. 4 [cite: 238] and Section 4.1 [cite: 240]
# -----------------------------------------------------------------------
class C2f_MPCA(nn.Module):
    """
    Faster Implementation of CSP Bottleneck with 2 convolutions.
    Replaces standard Bottleneck with MPCA block.
    """
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        
        # The main change: Use MPCA instead of Bottleneck
        self.m = nn.ModuleList(MPCA(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

# -----------------------------------------------------------------------
# 5. PSA: Partial Self-Attention
# Source: Fig. 5 [cite: 265] and Section 4.2 
# -----------------------------------------------------------------------
class PSA(nn.Module):
    def __init__(self, c1, c2, e=0.5):
        super().__init__()
        c_mid = int(c1 * e)
        self.cv1 = Conv(c1, 2 * c_mid, 1) # Split into two parts [cite: 245]
        
        # Part B gets MHSA + FFN
        # Simplified MHSA for efficiency (as typically used in YOLO variants)
        self.mhsa_dim = c_mid
        self.num_heads = 4 # Default assumption
        self.qkv = nn.Linear(c_mid, c_mid * 3)
        self.proj = nn.Linear(c_mid, c_mid)
        
        # FFN: Conv 1x1 expand -> Conv 1x1 squeeze [cite: 253]
        self.ffn = nn.Sequential(
            Conv(c_mid, c_mid * 2, 1),
            Conv(c_mid * 2, c_mid, 1)
        )
        
        self.cv_final = Conv(c_mid * 2, c2, 1) # Concatenate and fuse [cite: 254]

    def forward(self, x):
        # 1x1 Conv and Split [cite: 246]
        x_split = self.cv1(x).chunk(2, 1) 
        x_part_a = x_split[0] # Preserves original features
        x_part_b = x_split[1] # Goes to MHSA
        
        # MHSA Implementation [cite: 247, 251]
        B, C, H, W = x_part_b.shape
        # Flatten for attention
        flat_x = x_part_b.flatten(2).transpose(1, 2) # B, N, C
        qkv = self.qkv(flat_x).reshape(B, H*W, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        scale = (C // self.num_heads) ** -0.5 # [cite: 249]
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1) # [cite: 250]
        
        x_attn = (attn @ v).transpose(1, 2).reshape(B, H*W, C)
        x_attn = self.proj(x_attn).transpose(1, 2).reshape(B, C, H, W)
        
        # FFN
        x_part_b_out = self.ffn(x_attn)
        
        # Concat and Final Conv [cite: 254]
        return self.cv_final(torch.cat((x_part_a, x_part_b_out), 1))