
import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.Tanh(),
            nn.Linear(in_features // 2, 1)
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        attn_weights = torch.softmax(self.attention(x), dim=1)
        attended_features = torch.sum(attn_weights * x, dim=1)
        return self.classifier(attended_features)

class CLIPFineTunerWithAttention(nn.Module):
    def __init__(self, clip_model, num_classes):
        super().__init__()
        self.model = clip_model
        self.attention_head = AttentionHead(clip_model.visual.output_dim, num_classes)
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            features = self.model.encode_image(x).float()
        return self.attention_head(features)
