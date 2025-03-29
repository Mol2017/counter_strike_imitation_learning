import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
import torchvision.transforms as transforms


class CSTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.n_frame = 32
        self.n_head = 8
        self.n_encoder_layer = 8
        
        
        self.resnet_dim = 512
        self.transformer_dim = 512
        self.feedforward_dim = 2 * self.transformer_dim
        self.action_dim = 12
        
        # CNN backbone
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.tokenizer = torch.nn.Sequential(*list(resnet.children())[:-1])
        
        # Linear projection from resnet output to transformer dim
        self.token_projection = nn.Linear(self.resnet_dim, self.transformer_dim)
        
        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(self.n_frame, self.transformer_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_dim, 
            nhead=self.n_head, 
            dim_feedforward=self.feedforward_dim, 
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.n_encoder_layer)
        
        # Action head
        self.action_head = nn.Sequential(
            nn.Linear(self.transformer_dim, self.transformer_dim),
            nn.ReLU(),
            nn.Linear(self.transformer_dim, self.action_dim),
        )
    
        
    def forward(self, inputs: dict[str, torch.tensor]) -> dict[str, torch.tensor]:
        images = inputs['image'] # [B, T, C, H, W]
        mask = inputs['mask'] # [B, T]
        
        n_batch, _, _, _, _ = images.shape
        images = images.view(-1, 3, 224, 224) # [B * T, C, H, W]
        tokens = self.tokenizer(images) # [B * T, 512, 1, 1]
        tokens = tokens.view(n_batch, self.n_frame, -1) # [B, T, 512]
        tokens += self.pos_embed.unsqueeze(0) # [B, T, 512]
        
        hidden_states = self.transformer_encoder(tokens, src_key_padding_mask=mask) # [B, T, 512]
        actions = self.action_head(hidden_states[:, -1, :]) # [B, 12]
        
        outputs = {
            'action': actions
        }
        return outputs

inputs = {'image': torch.randn(1, 32, 3, 224, 224), 'mask': torch.randn(1, 32)}
model = CSTransformer()
output = model(inputs)
print(output['action'].shape)