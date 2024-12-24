import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=8, in_channels=1, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Linear(patch_size * patch_size * in_channels, embed_dim)
    
    def forward(self, x):
        B, T, H, W = x.shape
        p = self.patch_size
        x = x.unfold(2, p, p).unfold(3, p, p)  # (B, T, H/p, W/p, p, p)
        B, T, Hp, Wp, _, _ = x.shape
        x = x.contiguous().view(B, T, Hp*Wp, p*p)
        x = x.view(B, T*Hp*Wp, p*p)
        x = self.proj(x)
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
    def __init__(self, patch_size=8, in_channels=1, d_model=256, nhead=8, num_layers=6, H=32, W=32):
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
        x = self.patch_embed(x)  # (B, T*n_patches, d_model)
        x = self.pos_enc(x)

        x = self.transformer_encoder(x)  # (B, T*n_patches, d_model)

        # 最後の16トークンが次フレームのパッチ表現とする
        x = x[:, -16:, :]  # (B, 16, d_model)

        # 次フレームの速度予測
        velocity = self.out_proj_velocity(x)  # (B, 16, patch_size*patch_size)

        # 次フレームの圧力予測
        pressure = self.out_proj_pressure(x)  # (B, 16, patch_size*patch_size)

        p = self.patch_size
        n_patches_per_frame = (self.H // p) * (self.W // p)

        # 速度出力の形状を再構成
        velocity = velocity.view(velocity.size(0), n_patches_per_frame, p, p)  # (B, 16, 8, 8)
        velocity = velocity.view(velocity.size(0), self.H // p, self.W // p, p, p)  # (B, 4, 4, 8, 8)
        velocity = velocity.permute(0, 1, 3, 2, 4).contiguous().view(velocity.size(0), self.H, self.W)

        # 圧力出力の形状を再構成
        pressure = pressure.view(pressure.size(0), n_patches_per_frame, p, p)  # (B, 16, 8, 8)
        pressure = pressure.view(pressure.size(0), self.H // p, self.W // p, p, p)  # (B, 4, 4, 8, 8)
        pressure = pressure.permute(0, 1, 3, 2, 4).contiguous().view(pressure.size(0), self.H, self.W)

        return velocity, pressure

# メモリ消費テストコード
if __name__ == '__main__':
    import time
    from torch.cuda import memory_allocated, max_memory_allocated

    B = 2  # バッチサイズ
    T = 4  # 時系列の長さ
    H = 32  # 高さ
    W = 32  # 幅

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device} is available.")

    inp = torch.randn(B, T, H, W, device=device)  # 入力データ
    model = CFDTransformerModel().to(device)  # モデルインスタンスの作成
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    # ダミーターゲットデータ
    target_velocity = torch.randn(B, H, W, device=device)
    target_pressure = torch.randn(B, H, W, device=device)

    torch.cuda.reset_peak_memory_stats(device)  # メモリ計測の初期化

    start_time = time.time()

    # フォワードパス
    velocity_out, pressure_out = model(inp)

    # 損失計算
    loss_velocity = criterion(velocity_out, target_velocity)
    loss_pressure = criterion(pressure_out, target_pressure)
    loss = loss_velocity + loss_pressure

    # バックプロパゲーション
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    end_time = time.time()

    print("Input shape:", inp.shape)  # (B, 4, 32, 32)
    print("Velocity output shape:", velocity_out.shape)  # (B, 32, 32)
    print("Pressure output shape:", pressure_out.shape)  # (B, 32, 32)
    print("Loss:", loss.item())

    print("Execution time (s):", end_time - start_time)
    print("Memory allocated (MB):", memory_allocated(device) / 1024**2)
    print("Max memory allocated (MB):", max_memory_allocated(device) / 1024**2)

