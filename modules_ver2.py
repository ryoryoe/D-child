
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            #nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same", bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same", bias=False),
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
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Down_first(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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
        #x = center_crop(x, 64, 64)
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up_last(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
        #print(f"{x.shape=}")
        x = self.up(x)
        #print(f"{skip_x.shape=}")
        #print(f"{x.shape=}")
        
        #x = center_crop(x, 100, 100)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
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
        #print(f"{x.shape=}")
        x = self.up(x)
        #print(f"{skip_x.shape=}")
        #print(f"{x.shape=}")
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

def center_crop(x, new_height, new_width):
    """
    テンソルの中央部分をクロップする関数。
    
    :param x: 入力テンソル。サイズは (N, C, H, W)。
    :param new_height: クロップ後の高さ。
    :param new_width: クロップ後の幅。
    :return: クロップされたテンソル。
    """
    height, width = x.shape[2], x.shape[3]
    #print(f"{height=}")
    #print(f"{width=}")
    start_x = width // 2 - new_width // 2
    start_y =height // 2 - new_height // 2
    end_x = start_x + new_width
    end_y = start_y + new_height
    #print(f"{start_x=}")
    #print(f"{end_x=}")
    return x[:, :, start_y:end_y, start_x:end_x]


# クロップして30x30に変換

class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, time_dim=256, remove_deep_conv=True):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(1, 64)
        self.down1 = Down_first(64, 128)
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

        #self.up1 = Up(256, 128)
        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up_last(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, 1, kernel_size=1)

    def pos_encoding(self, t, channels):#sinとcosの値を計算して結合する。これによっt時系列関係を特徴づけることが出来る
        #t = t.to("cpu")
        #print(f"{t.device=}")
        #channels = channels.to("cpu")
        #print(f"{channels.device=}")
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        #inv_freq.to("cpu")
        #print(f"{inv_freq.device=}")
        pos_enc_a = torch.sin(t.repeat(1, channels // 2).to("cuda:0") * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2).to("cuda:0") * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        pos_enc.to("cuda:0")
        return pos_enc #時系列関係を特徴づけるためのベクトル

    def unet_forwad(self, x, t):#tはノイズを何回加えたかを表す
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
        t = t.unsqueeze(-1) # (B, T) -> (B, T, 1)
        t = self.pos_encoding(t, self.time_dim) # (B, T, 1) -> (B, T, 256)
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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.bn1 = nn.BatchNorm2d(num_features=16)     
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.bn2 = nn.BatchNorm2d(num_features=32)     
        self.fc_mu = nn.Linear(8*8*32, 32*32)
        self.fc_logvar = nn.Linear(8*8*32, 32*32)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        mu = mu.view(-1,1,32,32)
        logvar = self.fc_logvar(x)
        logvar = logvar.view(-1,1,32,32)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(1*32*32, 8*8*32)
        self.conv_trans1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #x = x.reshape(-1,32,32)
        x = self.fc(x)
        x = x.view(-1, 32, 8,8)
        #x = x.view(x.size(0), 32, 8, 8)
        x = F.relu(self.conv_trans1(x))
        x = torch.sigmoid(self.conv_trans2(x))
        return x

def reparameterize(mu, log_var):
    std = torch.exp(0.5*log_var)  # 標準偏差はlog_varの指数関数
    eps = torch.randn_like(std)  # 標準正規分布からサンプリング
    return mu + eps*std  # 潜在変数zを計算


class DDPM(nn.Module):
    def __init__(self, T):
        super(DDPM,self).__init__()
        self.T = T #ノイズを加える回数
        # β1 and βt はオリジナルの ddpm reportに記載されている値を採用します
        self.beta_1 = 1e-4 #t=1のノイズの大きさ
        self.beta_T = 0.02 #t=Tのノイズの大きさ
        # β = [β1, β2, β3, ... βT] (length = T)
        #self.betas = torch.linspace(self.beta_1, self.beta_T, T, device=device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        self.betas = torch.linspace(self.beta_1, self.beta_T, T)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        # α = [α1, α2, α3, ... αT] (length = T)
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列

    def forward(self, x0, t=None):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #print(device)
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],)) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
            #print(f"{t.device=}")
            #t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0).to("cpu") #ノイズを生成
        #print(f"{noise.device=}")
        alpha_bars = self.alpha_bars[t]
        #print(f"{alpha_bars.device=}")
        alpha_bars.to(device)
        #print(f"{alpha_bars.device=}")
        alpha_bar = alpha_bars.reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        x0 = x0.to("cpu")
        #print(f"{x0.device=}")
        #print(f"{torch.sqrt(1-alpha_bar)=}")
        #alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1).to(device) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        #print(f"{xt.device=}")
        xt = xt.to(device)
        #print(f"{xt.device=}")
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ

class Stable_diffusion(nn.Module):
    def __init__(self):
        super(Stable_diffusion,self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.diffusion_model = DDPM(1000) 
        self.unet = UNet()

    def forward(self, x):
        encode_avg, encode_std = self.encoder(x)
        x=reparameterize(encode_avg,encode_std)
        #x = torch.stack((encode_avg, encode_std), dim=1)
        #xt, t, noise = ddpm.diffusion_process(x)
        xt, t, noise = self.diffusion_model(x)
        x = self.unet(xt,t)
        out = self.decoder(x)
        return out, encode_avg, encode_std
