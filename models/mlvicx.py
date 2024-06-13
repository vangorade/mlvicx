import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.nn.init as init

__all__ = ['mlvicx']

class Projection_MLP(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim):
        super(Projection_MLP, self).__init__()
        
        self.l1 = nn.Sequential(nn.Linear(in_dim, hid_dim),
                                nn.BatchNorm1d(hid_dim),
                                nn.ReLU(inplace=True))
        
        self.l2 = nn.Sequential(nn.Linear(hid_dim, hid_dim),
                                nn.BatchNorm1d(hid_dim),
                                nn.ReLU(inplace=True))
        
        self.l3 = nn.Sequential(nn.Linear(hid_dim, out_dim),
                                nn.BatchNorm1d(out_dim))

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)

        return x


class Projection_CONV(nn.Module):
    def __init__(self, input_channels=512, final_size = 4096):
        super(Projection_CONV, self).__init__()
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)), 
            nn.Flatten(),  
            nn.Linear(input_channels * 4 * 4, final_size),
            nn.BatchNorm1d(final_size),
            nn.ReLU(inplace=True),
            nn.Linear(final_size, final_size),
            nn.BatchNorm1d(final_size),
            nn.ReLU(inplace=True),
            nn.Linear(final_size, final_size),
            nn.BatchNorm1d(final_size)
        )

    def forward(self, x):
        x = self.projection(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, input_dim, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(input_dim, input_dim//reduction_ratio, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(input_dim//reduction_ratio, input_dim, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)  

class Context_bottleneck(nn.Module):
    def __init__(self,input_dim,reduction_ratio=2 ):
        super().__init__()
        self.cam = ChannelAttention(input_dim=input_dim,reduction_ratio=reduction_ratio)
        self.sam = SpatialAttention()
        
    def forward(self,x):
        cx  = self.cam(x) * x
        sx  = self.sam(cx) * cx
        
        return sx

class ResizeAndProject(nn.Module):
    def __init__(self, config):
        super(ResizeAndProject, self).__init__()
        self.output_channels = config['model']['out_channels']
        self.output_size     = config['model']['out_size']
        
        # projection
        input_dim   = config['model']['projection']['input_dim']
        hidden_dim  = config['model']['projection']['hidden_dim']
        output_dim  = config['model']['projection']['output_dim']
#         self.mlp_head  = Projection_MLP(input_dim, hidden_dim, output_dim)
        self.conv_head = Projection_CONV(input_channels = input_dim, final_size = output_dim)
        
        self.convs = nn.ModuleList()

        
    def _apply_conv(self, tensor, index):
        conv = self.convs[index].to(tensor.device)
        return conv(tensor)    
        
    
    def forward(self, tensors):
        if not tensors:
            raise ValueError("The input list of tensors should not be empty.")
        bs = tensors[0].size(0)
        common_shape = (bs, self.output_channels, self.output_size, self.output_size)
        resized_tensors = []
        # Ensure the list of convs is same length as tensors
        while len(self.convs) < len(tensors):
            self.convs.append(nn.Conv2d(tensors[len(self.convs)].size(1), self.output_channels, kernel_size=1))
        
        for i, tensor in enumerate(tensors):
            resized_tensor = F.interpolate(tensor, size=self.output_size, mode='bilinear', align_corners=False)
            resized_tensor = self._apply_conv(resized_tensor, i)
            resized_tensors.append(resized_tensor)

        # Ensure all tensors are of the same shape
        for t in resized_tensors:
            assert t.shape == common_shape, f"Resized tensor shape {t.shape} does not match common shape {common_shape}"
        
        # Element-wise addition of all resized tensors
        x = torch.stack(resized_tensors).sum(dim=0)
        x = self.conv_head(x)        
        return x

class EncoderwithProjection(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.pretrained  = config['model']['backbone']['pretrained']
        net_name         = config['model']['backbone']['type']
        base_encoder     = models.__dict__[net_name](pretrained=self.pretrained)
        num_ftrs         = base_encoder.fc.in_features
        
        if not self.pretrained:
            self._initialize_weights(base_encoder)
                
        self.encoder = nn.Sequential(*list(base_encoder.children())[:-1])
        
        # projection
        input_dim   = config['model']['projection']['input_dim']
        hidden_dim  = config['model']['projection']['hidden_dim']
        output_dim  = config['model']['projection']['output_dim']
        self.projetion = Projection_MLP(input_dim, hidden_dim, output_dim)
        
        #attention blocks
        self.cbt1 = Context_bottleneck(self.encoder[0].out_channels)
        self.cbt2 = Context_bottleneck(self.encoder[4][1].conv2.out_channels)
        self.cbt3 = Context_bottleneck(self.encoder[5][1].conv2.out_channels)
        self.cbt4 = Context_bottleneck(self.encoder[6][1].conv2.out_channels)        
        self.cbt5 = Context_bottleneck(self.encoder[7][1].conv2.out_channels)
        
        self.resize_project = ResizeAndProject(config)
        self.avgpool        = nn.AdaptiveAvgPool2d((1, 1))
        
        
    def _initialize_weights(self, module):
        if isinstance(module, nn.Conv2d):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            init.constant_(module.weight, 1)
            init.constant_(module.bias, 0)
        elif isinstance(module, nn.Linear):
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0)   
                

    def forward(self, x):
        map1   = self.encoder[:3](x)
        a_map1 = self.cbt1(map1)
        map2   = self.encoder[3:5](a_map1)
        a_map2 = self.cbt2(map2)
        map3   = self.encoder[5:6](a_map2)
        a_map3 = self.cbt3(map3)
        map4   = self.encoder[6:7](a_map3)
        a_map4 = self.cbt4(map4)
        map5   = self.encoder[7:8](a_map4)
        a_map5 = self.cbt5(map5)
        x      = torch.flatten(self.avgpool(a_map5),1)
        x      = self.projetion(x)   
        maps   = [a_map1, a_map2, a_map3, a_map4, a_map5]
        dense_map = self.resize_project(maps)
                
        return x,  dense_map  

class mlvicx(nn.Module):
    def __init__(self, config):
        super().__init__()
           
        self.sim_coeff       = config['model']['loss']['sim_coeff']
        self.std_coeff       = config['model']['loss']['std_coeff']
        self.cov_coeff       = config['model']['loss']['cov_coeff']
        self.encoder   = EncoderwithProjection(config)
        
    def invariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes mse loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: invariance loss (mean squared error).
        """
        return F.mse_loss(z1, z2)

    def variance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes variance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: variance regularization loss.
        """
        eps = 1e-4
        std_z1 = torch.sqrt(z1.var(dim=0) + eps)
        std_z2 = torch.sqrt(z2.var(dim=0) + eps)
        std_loss = torch.mean(F.relu(1 - std_z1))/2 + torch.mean(F.relu(1 - std_z2))/2
        return std_loss

    def covariance_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """Computes covariance loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
        Returns:
            torch.Tensor: covariance regularization loss.
        """

        N, D = z1.size()
        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)
        cov_z1 = (z1.T @ z1) / (N - 1)
        cov_z2 = (z2.T @ z2) / (N - 1)
        diag     = torch.eye(D, device=z1.device)
        cov_loss = cov_z1[~diag.bool()].pow_(2).sum()/D + cov_z2[~diag.bool()].pow_(2).sum()/D
        return cov_loss

    def vicreg_loss_func(self,
                         z1: torch.Tensor,
                         z2: torch.Tensor,
                         sim_loss_weight: float = 25.0,
                         var_loss_weight: float = 25.0,
                         cov_loss_weight: float = 1.0,
                         ) -> torch.Tensor:
        
        # https://github.com/facebookresearch/vicreg/blob/main/main_vicreg.py

        """Computes VICReg's loss given batch of projected features z1 from view 1 and
        projected features z2 from view 2.
        Args:
            z1 (torch.Tensor): NxD Tensor containing projected features from view 1.
            z2 (torch.Tensor): NxD Tensor containing projected features from view 2.
            sim_loss_weight (float): invariance loss weight.
            var_loss_weight (float): variance loss weight.
            cov_loss_weight (float): covariance loss weight.
        Returns:
            torch.Tensor: VICReg loss.
        """
        z1 = z1.reshape(z1.size(0), -1)
        z2 = z2.reshape(z2.size(0), -1)
        
        sim_loss = self.invariance_loss(z1, z2)

        var_loss = self.variance_loss(z1, z2)
        cov_loss = self.covariance_loss(z1, z2)

        loss = (self.sim_coeff    * sim_loss 
                + self.std_coeff  * var_loss 
                + self.cov_coeff  * cov_loss)
        return loss

    def forward(self, img):
        img_v1, img_v2 = img
        z1, d1 = self.encoder(img_v1)
        z2, d2 = self.encoder(img_v2)
        loss  = self.vicreg_loss_func(z1,z2)
        loss += self.vicreg_loss_func(d1,d2)      
        return loss.mean()
