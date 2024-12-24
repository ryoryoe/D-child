import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=8, in_channels=2, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        p = self.patch_size

        # (B, C, T, H, W) -> (B, T, C, H, W)
        # 時系列方向 T を2番目の軸に、C を3番目に移動
        x = x.permute(0, 2, 1, 3, 4)  # (B, T, C, H, W)

        # パッチ分割
        # unfoldによりH,W方向にパッチ化する
        x = x.unfold(3, p, p).unfold(4, p, p)  # (B, T, C, H/p, W/p, p, p)
        B, T, C, Hp, Wp, _, _ = x.shape

        # パッチ毎にreshape: (B, T, C, Hp*Wp, p*p)
        x = x.contiguous().view(B, T, C, Hp*Wp, p*p)

        # (B, T, Hp*Wp, C*p*p) の形に並べ替え (時系列*Tパッチ数がsequence軸になる)
        x = x.permute(0, 1, 3, 2, 4).contiguous().view(B, T*Hp*Wp, C*p*p)

        # 線形射影
        x = self.proj(x)  # (B, T*n_patches, embed_dim)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class CFDTransformerModel(nn.Module):
    def __init__(self, patch_size=8, in_channels=2, d_model=256, nhead=8, num_layers=6, H=32, W=32):
        super().__init__()
        self.patch_size = patch_size
        self.H = H
        self.W = W
        self.in_channels = in_channels

        self.patch_embed = PatchEmbed(patch_size=patch_size, in_channels=in_channels, embed_dim=d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=1024, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_enc = PositionalEncoding(d_model)

        # 2つの出力層を追加
        self.out_proj_velocity = nn.Linear(d_model, patch_size * patch_size * in_channels)
        self.out_proj_pressure = nn.Linear(d_model, patch_size * patch_size * in_channels)

    def forward(self, x):
        # xは (B, C, T, H, W) の形で入力される想定
        x = self.patch_embed(x)  # (B, T*n_patches, d_model)
        x = self.pos_enc(x)
        x = self.transformer_encoder(x)  # (B, T*n_patches, d_model)

        # 最後の16トークンが次フレームのパッチ表現とする
        # T=4, H=32, W=32, patch=8 -> (H/p * W/p) = 4*4=16パッチ
        x = x[:, -16:, :]  # (B, 16, d_model)

        # 次フレームの速度予測 (B,16,patch_size*patch_size*in_channels)
        velocity = self.out_proj_velocity(x)
        # 次フレームの圧力予測 (B,16,patch_size*patch_size*in_channels)
        pressure = self.out_proj_pressure(x)

        p = self.patch_size
        n_patches_per_frame = (self.H // p) * (self.W // p)

        # 速度出力の形状を再構成 (B, C*p*p, 16) -> (B, C, 16, p, p) -> (B, C, H, W)
        velocity = velocity.view(velocity.size(0), n_patches_per_frame, self.in_channels, p, p)
        velocity = velocity.permute(0,2,1,3,4).contiguous()
        # (B, C, 16, 8, 8) -> (B, C, 4, 4, 8, 8) に再構成してからC,H,Wへ
        velocity = velocity.view(velocity.size(0), self.in_channels, self.H//p, self.W//p, p, p)
        # 最終的に (B, C, H, W)
        velocity = velocity.permute(0,1,2,4,3,5).contiguous().view(velocity.size(0), self.in_channels, self.H, self.W)

        # 圧力出力の形状を再構成 (B,16,C*p*p) -> (B,C,H,W)
        pressure = pressure.view(pressure.size(0), n_patches_per_frame, self.in_channels, p, p)
        pressure = pressure.permute(0,2,1,3,4).contiguous()
        pressure = pressure.view(pressure.size(0), self.in_channels, self.H//p, self.W//p, p, p)
        pressure = pressure.permute(0,1,2,4,3,5).contiguous().view(pressure.size(0), self.in_channels, self.H, self.W)

        return velocity, pressure

# テストコード
if __name__ == '__main__':
    B = 2  # バッチサイズ
    D = 2
    T = 10  # 時系列の長さ
    H = 32  # 高さ
    W = 32  # 幅
    inp = torch.randn(B,D, T, H, W)  # 入力データ

    model = CFDTransformerModel()  # モデルインスタンスの作成
    velocity_out, pressure_out = model(inp)  # モデルの実行

    print("Input shape:", inp.shape)  # (B, 4, 32, 32)
    print("Velocity output shape:", velocity_out.shape)  # (B, 32, 32)
    print("Pressure output shape:", pressure_out.shape)  # (B, 32, 32)

