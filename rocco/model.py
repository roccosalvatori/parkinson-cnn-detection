import torch
import torch.nn as nn
import timm

class AttentionModule(nn.Module):
    def __init__(self, in_channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        
        
        self.query_conv = nn.Conv2d(in_channels, in_channels, 1)
        self.key_conv = nn.Conv2d(in_channels, in_channels, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.out_conv = nn.Conv2d(in_channels, in_channels, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        
        
        query = self.query_conv(x).view(B, self.num_heads, C // self.num_heads, -1)
        key = self.key_conv(x).view(B, self.num_heads, C // self.num_heads, -1)
        value = x.view(B, self.num_heads, C // self.num_heads, -1)
        
        
        attention = torch.matmul(query, key.transpose(-2, -1))
        attention = self.softmax(attention / (C // self.num_heads) ** 0.5)
        
        
        out = torch.matmul(attention, value)
        out = out.view(B, C, H, W)
        out = self.out_conv(out)
        
        return out + x  

class PSLAModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        
     
        self.backbone = timm.create_model('efficientnet_b2', 
                                        pretrained=True,
                                        drop_rate=0.3,  
                                        drop_path_rate=0.2,  
                                        num_classes=0)
        
        backbone_channels = 1408
        
        self.input_bn = nn.BatchNorm2d(1)
        
        
        self.attention = nn.Sequential(
            AttentionModule(backbone_channels),
            nn.Dropout2d(0.3) 
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),  
            nn.Linear(backbone_channels, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.input_bn(x)
        
   
        x = x.repeat(1, 3, 1, 1)
        
        features = self.backbone.forward_features(x)
        features = self.attention(features)
        pooled = self.global_pool(features).flatten(1)
        output = self.classifier(pooled)
        return output 