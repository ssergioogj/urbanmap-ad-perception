import torch
import timm
import torch.nn as nn
import torch.nn.functional as F

class Up(nn.Module):
    """
    Spatially aligns a deep feature map (with low resolution and high semantic information) with a shallow map (high resolution with more geometric detail) using bilinear interpolation and combines them using convolutions
    """
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # Scale x1 for match spatial size of x2
        x1 = F.interpolate(x1, size=x2.shape[-2:], mode="bilinear", align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class CamEncode(nn.Module):
    """
    Implements "Lift" of LSS
    """

    # Lift step is defined as following:
    # 1. Extract features from the image with the EfficientNet backbone (Deep feature map (c4): low resolution and high semantic info, Shallow map (c2): high resoltion and low semantic info)
    # 2. Mix the features in a intermediate resolution map
    # 3. Predicts a discrete probability distribution over the depth and a context features vector for each pixel
    # 4. Computes the outer product (DxC) to generate a 3D visual cone (frustum of features)

    def __init__(self, d_bins: int = 41, c_chanels: int = 64, backbone_name: str = "efficientnet_b0", pretrained: bool = True):
        super().__init__()
        self.d_bins = d_bins    # Discrete depth intervals (from 4 to 45 meters). Less than 4 is not captured by the camera and more than 45 is not relevant for taking an actual decision
        self.c_chanels = c_chanels # Dimension of the semantic vector for each 3D point 

        self.image_backbone = timm.create_model(
            backbone_name, 
            pretrained=pretrained,  # Trained with ImageNet
            features_only=True,     # Extracts intermediate maps without classification head
            out_indices=(2,4)       # Selects step C2 (stride 8) and C4 (stride 32)
        )

        feature_info = self.image_backbone.feature_info.channels()

        c2_dim = feature_info[0]
        c4_dim = feature_info[1]

        self.neck = Up(in_channels=c4_dim, skip_channels=c2_dim, out_channels=128)

        self.depth_context_conv = nn.Conv2d(in_channels=128, out_channels=self.d_bins + self.c_chanels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: Tensor de entrada con forma (B, N, 3, H, W)
               B = Batch size
               N = Number of cameras (6)
               3 = Canales RGB
               H = Heigth of the image
               W = Width of the image

        Returns:
            frustum_features: (B, N, D, C, H_feat, W_feat)
            depth_prob:       (B, N, D, H_feat, W_feat)
        """

        B, N, C_in, H, W = x.shape

        # Collapse batch size and number of cameras
        x = x.view(B * N, C_in, H, W)

        # Extract features
        features = self.image_backbone(x)

        c2, c4 = features[0], features[1]

        # Scales c4 to spatial size of c2 and combine channels
        fusion = self.neck(c4, c2)

        # 1x1 proyection D+C channels
        out = self.depth_context_conv(fusion)

        # 'out' is a tensor that mix depth and context
        # Depth logits indicates the prob distribution for 'self.d_bins'
        # Context is the semantic vector 
        depth_logits = out[:, :self.d_bins]
        context = out[:, self.d_bins:]

        # Normalize depth probs per pixel (apply softmax for obtaining a probability distribution)
        depth_prob = F.softmax(depth_logits, dim=1)

        # Lift: Exterior product with tensor broadcasting
        # depth_prob:  (B*N, D, 1, H_feat, W_feat) -> add singleton dimension on C
        # context:     (B*N, 1, C, H_feat, W_feat) -> add singleton dimension on D
        # Multiply each scalar prob of depth per the semantic vector
        frustum_features = depth_prob.unsqueeze(2) * context.unsqueeze(1)

        # Restore dimensions decoupling batch and cameras
        _, D, C_feat, H_f, W_f = frustum_features.shape
        frustum_features = frustum_features.view(B, N, D, C_feat, H_f, W_f)
        depth_prob = depth_prob.view(B, N, D, H_f, W_f)

        return frustum_features, depth_prob

