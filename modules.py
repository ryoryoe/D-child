import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.conv1= nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same", bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same", bias=False)
        self.gn1 = nn.GroupNorm(1, mid_channels)
        self.gn2 = nn.GELU()
        self.gn3 = nn.GroupNorm(1, out_channels)
        self.double_conv = nn.Sequential(
            #nn.Conv2d(2, 64, kernel_size=3, padding=1, bias=False),
            #nn.Conv2d(in_channels, mid_channels, kernel_size=2, padding="same", bias=False),
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding="same", bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding="same", bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            x_ = x
            #print(f"doubleconv_{x.shape=}")
            x = self.conv1(x)
            #print(f"conv1_{x.shape=}")
            x = self.gn1(x)
            #print(f"gn1_{x.shape=}")
            x = self.gn2(x)
            #print(f"gn2_{x.shape=}")
            x = self.conv2(x)
            #print(f"conv2_{x.shape=}")
            x = self.gn3(x)
            #print(f"gn3_{x.shape=}")
            #return self.double_conv(x)
            return F.gelu(x_ + x)
        else:
            #print(f"doubleconv_{x.shape=}")
            x = self.conv1(x)
            #print(f"conv1_{x.shape=}")
            x = self.gn1(x)
            #print(f"gn1_{x.shape=}")
            x = self.gn2(x)
            #print(f"gn2_{x.shape=}")
            x = self.conv2(x)
            #print(f"conv2_{x.shape=}")
            x = self.gn3(x)
            #print(f"gn3_{x.shape=}")
            #return self.double_conv(x)
            return x

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
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

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
        #x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up_sum_and_cat(nn.Module): #残差接続のみ実装
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
        #print(f"up2_first_{x.shape=}")
        x = self.up(x)
        #print(f"up2_second_{x.shape=}")
        #print(f"{skip_x.shape=}")
        #print(f"{x.shape=}")
        x = x + skip_x
        x = torch.cat([skip_x, x], dim=1)
        #print(f"up2_cat_{x.shape=}")
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
class Up_sum(nn.Module): #残差接続のみ実装
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
        #print(f"up2_first_{x.shape=}")
        x = self.up(x)
        #print(f"up2_second_{x.shape=}")
        #print(f"{skip_x.shape=}")
        #print(f"{x.shape=}")
        x = x + skip_x
        #x = torch.cat([skip_x, x], dim=1)
        #print(f"up2_cat_{x.shape=}")
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb

class Up_new_0406(nn.Module):
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
        #print(f"up2_first_{x.shape=}")
        x = self.up(x)
        #print(f"up2_second_{x.shape=}")
        #print(f"{skip_x.shape=}")
        #print(f"{x.shape=}")
        x = torch.cat([skip_x, x], dim=1)
        #print(f"up2_cat_{x.shape=}")
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
        #print(f"up2_first_{x.shape=}")
        x = self.up(x)
        #print(f"up2_second_{x.shape=}")
        #print(f"{skip_x.shape=}")
        #print(f"{x.shape=}")
        x = torch.cat([skip_x, x], dim=1)
        #print(f"up2_cat_{x.shape=}")
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
class conditional_diffusion_0407_sum_and_cat(nn.Module):
    def __init__(self, c_in=2, c_out=2, time_dim=256, remove_deep_conv=True):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        
        #encoder
        self.down1 = DoubleConv(2, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        #self.down4 = Down(256, 512)
        self.down4 = Down(256, 256)
        
        #decoder(deccoderは残差接続を行っている)
        self.up1 = Up_sum_and_cat(512, 256) #使ってない
        self.up2 = Up_sum_and_cat(512, 128) 
        self.up3 = Up_sum_and_cat(256, 64)
        self.up4 = Up_sum_and_cat(128, 64)
        self.up5 = nn.Conv2d(64, 2, kernel_size=1)
        
        #bottom
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        #linear
        self.linear2 = nn.Linear(2,2*32*32)
        self.linear64 = nn.Linear(2,64*32*32)
        self.linear128 = nn.Linear(2,128*16*16)
        self.linear256 = nn.Linear(2,256*8*8)
        self.linear512 = nn.Linear(2,512*4*4)
        self.linear6 = nn.Linear(2,128*8*8)
        self.linear7 = nn.Linear(2,64*16*16)
        
        #attention
        self.sa64 = SelfAttention(64)
        self.sa128 = SelfAttention(128)
        self.sa256 = SelfAttention(256)
        self.sa512 = SelfAttention(512)

        """if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)"""


    def pos_encoding(self, t, channels):#sinとcosの値を計算して結合する。これによっt時系列関係を特徴づけることが出来る
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc #時系列関係を特徴づけるためのベクトル

    def unet_forwad(self, x, t):#tはノイズを何回加えたかを表す
        x1 = self.down1(x)
        x2 = self.down2(x1, t)
        x2 = self.sa128(x2)
        x3 = self.down3(x2, t)
        x3 = self.sa256(x3)
        x4 = self.down4(x3, t)
        x4 = self.sa256(x4)
        x = self.up2(x4, x3, t)
        x = self.sa128(x)
        x = self.up3(x, x2, t)
        x = self.sa64(x)
        x = self.up4(x, x1, t)
        x = self.sa64(x)
        output = self.up5(x)
        return output

    def forward(self, x, t):
        t = t.unsqueeze(-1) # (B, T) -> (B, T, 1)
        t = self.pos_encoding(t, self.time_dim) # (B, T, 1) -> (B, T, 256)
        return self.unet_forwad(x, t)
class conditional_diffusion_0407_sum(nn.Module):
    def __init__(self, c_in=2, c_out=2, time_dim=256, remove_deep_conv=True):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        
        #encoder
        self.down1 = DoubleConv(2, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        #self.down4 = Down(256, 512)
        self.down4 = Down(256, 256)
        
        #decoder(deccoderは残差接続を行っている)
        self.up1 = Up_sum(512, 256) #使ってない
        self.up2 = Up_sum(256, 128) 
        self.up3 = Up_sum(128, 64)
        self.up4 = Up_sum(64, 64)
        self.up5 = nn.Conv2d(64, 2, kernel_size=1)
        
        #bottom
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        #linear
        self.linear2 = nn.Linear(2,2*32*32)
        self.linear64 = nn.Linear(2,64*32*32)
        self.linear128 = nn.Linear(2,128*16*16)
        self.linear256 = nn.Linear(2,256*8*8)
        self.linear512 = nn.Linear(2,512*4*4)
        self.linear6 = nn.Linear(2,128*8*8)
        self.linear7 = nn.Linear(2,64*16*16)
        
        #attention
        self.sa64 = SelfAttention(64)
        self.sa128 = SelfAttention(128)
        self.sa256 = SelfAttention(256)
        self.sa512 = SelfAttention(512)

        """if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)"""


    def pos_encoding(self, t, channels):#sinとcosの値を計算して結合する。これによっt時系列関係を特徴づけることが出来る
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc #時系列関係を特徴づけるためのベクトル

    def unet_forwad(self, x, t):#tはノイズを何回加えたかを表す
        x1 = self.down1(x)
        x2 = self.down2(x1, t)
        x2 = self.sa128(x2)
        x3 = self.down3(x2, t)
        x3 = self.sa256(x3)
        x4 = self.down4(x3, t)
        x4 = self.sa256(x4)
        x = self.up2(x4, x3, t)
        x = self.sa128(x)
        x = self.up3(x, x2, t)
        x = self.sa64(x)
        x = self.up4(x, x1, t)
        x = self.sa64(x)
        output = self.up5(x)
        return output

    def forward(self, x, t):
        t = t.unsqueeze(-1) # (B, T) -> (B, T, 1)
        t = self.pos_encoding(t, self.time_dim) # (B, T, 1) -> (B, T, 256)
        return self.unet_forwad(x, t)


class conditional_diffusion_0406(nn.Module):
    def __init__(self, c_in=2, c_out=2, time_dim=256, remove_deep_conv=True):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        
        #encoder
        self.down1 = DoubleConv(2, 64)
        self.down2 = Down(64, 128)#center_cropを使うので最初のdownは特殊にしてた名残
        self.down3 = Down(128, 256)
        #self.down4 = Down(256, 512)
        self.down4 = Down(256, 256)
        
        #decoder(deccoderはtorch.catを用いて配列を列方向に結合してから使うので特徴量は2倍になる)
        self.up1 = Up(512, 256) #使ってない
        self.up2 = Up(512, 128) #firstはskip_connectionがないので倍になることはない
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)
        self.up5 = nn.Conv2d(64, 2, kernel_size=1)
        
        #bottom
        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)
        #linear
        self.linear2 = nn.Linear(2,2*32*32)
        self.linear64 = nn.Linear(2,64*32*32)
        self.linear128 = nn.Linear(2,128*16*16)
        self.linear256 = nn.Linear(2,256*8*8)
        self.linear512 = nn.Linear(2,512*4*4)
        self.linear6 = nn.Linear(2,128*8*8)
        self.linear7 = nn.Linear(2,64*16*16)
        
        #attention
        self.sa64 = SelfAttention(64)
        self.sa128 = SelfAttention(128)
        self.sa256 = SelfAttention(256)
        self.sa512 = SelfAttention(512)

        """if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)"""


    def pos_encoding(self, t, channels):#sinとcosの値を計算して結合する。これによっt時系列関係を特徴づけることが出来る
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc #時系列関係を特徴づけるためのベクトル

    def unet_forwad(self, x, t):#tはノイズを何回加えたかを表す
        #print(f"{v.shape=}")
        #v1 = self.linear2(v).view(-1,2,32,32)
        #x += v1 
        #x0 = x
        #print(f"{x.shape=}")
        x1 = self.down1(x)
        #x1 = self.sa64(x1)
        #v2 = self.linear64(v).view(-1,64,32,32)
        #x1 += v2
        #print(f"{x1.shape=}")
        x2 = self.down2(x1, t)
        x2 = self.sa128(x2)
        #v3 = self.linear128(v).view(-1,128,16,16)
        #x2 += v3
        #print(f"{x2.shape=}")
        x3 = self.down3(x2, t)
        x3 = self.sa256(x3)
        x4 = self.down4(x3, t)
        x4 = self.sa256(x4)
        #x4 = self.bot1(x4)
        #x4 = self.bot2(x4)
        #x4 = self.bot3(x4)
        #v4 = self.linear256(v).view(-1,256,8,8)
        #x3 += v4
        #print(f"{x3.shape=}")
        #x4 = self.down4(x3, t)
        #x4 = self.sa512(x4)
        #v5 = self.linear512(v).view(-1,512,4,4)
        #x4 += v5
        #print(f"{x4.shape=}")
        #x4 = self.bot1(x4)
        #if not self.remove_deep_conv:
        #    x4 = self.bot2(x4)
        #x4 = self.bot3(x4)
        x = self.up2(x4, x3, t)
        x = self.sa128(x)
        #v6 = self.linear256(v).view(-1,256,8,8)
        #x += v6
        #print(f"dec_256_{x.shape=}")
        x = self.up3(x, x2, t)
        x = self.sa64(x)
        #v6 = self.linear128(v).view(-1,128,16,16)
        #x += v6
        x = self.up4(x, x1, t)
        x = self.sa64(x)
        #v7 = self.linear64(v).view(-1,64,32,32)
        #x += v7
        output = self.up5(x)
        #v8 = self.linear2(v).view(-1,2,32,32)
        #output = output + x0 + v8
        return output

    def forward(self, x, t):
        t = t.unsqueeze(-1) # (B, T) -> (B, T, 1)
        t = self.pos_encoding(t, self.time_dim) # (B, T, 1) -> (B, T, 256)
        return self.unet_forwad(x, t)

class UNet(nn.Module):
    def __init__(self, c_in=2, c_out=2, time_dim=256, remove_deep_conv=True):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(2, 64)
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
        self.outc = nn.Conv2d(64, 2, kernel_size=1)

    def pos_encoding(self, t, channels):#sinとcosの値を計算して結合する。これによっt時系列関係を特徴づけることが出来る
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
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
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.fc_mu = nn.Linear(8*8*32, 32*32)
        self.fc_logvar = nn.Linear(8*8*32, 32*32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        mu = mu.view(-1,32,32)
        logvar = self.fc_logvar(x)
        logvar = logvar.view(-1,32,32)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(32*32, 8*8*32)
        self.conv_trans1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = x.reshape(-1,32*32)
        x = self.fc(x)
        x = x.view(x.size(0), 32, 8, 8)
        x = F.relu(self.conv_trans1(x))
        #print(f"conv_trans1_{x.shape=}")
        x = torch.sigmoid(self.conv_trans2(x))
        #print(f"end_decoder_{x.shape=}")
        return x

class Vae_Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Vae_Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        
    def reparameterization(self, mean, var,device):
        epsilon = torch.randn_like(var).to(device)        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x,device):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var),device) # takes exponential function (log var -> var)
        x_hat = self.Decoder(z)
        
        return x_hat, mean, log_var

def vae_loss_function(x, x_hat, mean, log_var):
    mse_loss = nn.MSELoss(reduction='sum')
    #reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reproduction_loss = mse_loss(x_hat, x)
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

