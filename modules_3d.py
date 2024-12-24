import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        #self.mha = nn.MultiheadAttention(channels, 8, batch_first=True)
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        #size = x.shape[-1]
        batch_size, channels, depth, height, width = x.shape
        x = x.view(-1, self.channels, depth * height * width).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, depth, height, width)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        # z方向にはプーリングを適用せず、xとy方向にのみプーリングを適用する
        #self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),#メッシュサイズが16*16*16の時
            #nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1)),#メッシュサイズが32*32*4の時
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        #x = self.maxpool(x)
        emb = self.emb_layer(t)[:, :, None,None, None].repeat(1, 1,x.shape[-3] ,x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="trilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x) #メッシュサイズが16*16*16の時
        #x = F.interpolate(x, scale_factor=(2, 2, 1), mode='trilinear', align_corners=True)#メッシュサイズが32*32*4の時
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :,None, None, None].repeat(1, 1,x.shape[-3], x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=True):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv3d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)

class Input_2VModel(nn.Module):
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_2VModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2, 3*16*16*16)
        #self.linear1 = nn.Linear(2, 3*16*16*16)
        #self.linear2 = nn.Linear(8*16*16*16, 3**16*16*16)
        self.norm1 = nn.LayerNorm(3*16*16*16)
        #self.norm2 = nn.LayerNorm(2*16*16*16)
        self.T = 1000 #ノイズを加える回数
        self.beta_1 = 1e-6 #t=1のノイズの大きさ(最初1.0e-4)
        self.beta_T = 2.0e-4 #t=Tのノイズの大きさ(最初0.02)
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列
        
    def diffusion_process(self, x0,t=None):
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0, device=self.device) #ノイズを生成
        alpha_bar = self.alpha_bars[t].reshape(-1, 1,1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
        z = self.linear1(x)
        z = self.relu(z)
        z = self.norm1(z)
        #z = self.linear2(z)
        #z = self.relu(z)
        #z = self.norm2(z)
        #z = z.view(-1, 3,32,32,4)
        z = z.view(-1, 3,16,16,16)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Image_Model(nn.Module):
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_Image_Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2, 3*16*16*16)
        #self.linear1 = nn.Linear(2, 3*16*16*16)
        #self.linear2 = nn.Linear(8*16*16*16, 3**16*16*16)
        self.norm1 = nn.LayerNorm(3*16*16*16)
        #self.norm2 = nn.LayerNorm(2*16*16*16)
        self.T = 1000 #ノイズを加える回数
        self.beta_1 = 1e-6 #t=1のノイズの大きさ(最初1.0e-4)
        self.beta_T = 2.0e-4 #t=Tのノイズの大きさ(最初0.02)
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列
        
    def diffusion_process(self, x0,t=None):
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0, device=self.device) #ノイズを生成
        alpha_bar = self.alpha_bars[t].reshape(-1, 1,1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
        #z = self.linear1(x)
        #z = self.relu(z)
        #z = self.norm1(z)
        #z = self.linear2(z)
        #z = self.relu(z)
        #z = self.norm2(z)
        #z = z.view(-1, 3,32,32,4)
        #z = z.view(-1, 3,16,16,16)
        z,t,noise = self.diffusion_process(x)
        z = self.UNet(z,t) 
        return z

