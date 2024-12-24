import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))


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

class Up_last_to_high_resolution(nn.Module): #最後の層のスケールファクターを4にして画像の高解像度化に対応
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True)
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
        self.fc = nn.Linear(2*32*32, 8*8*32)
        self.conv_trans1 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv_trans2 = nn.ConvTranspose2d(16, 2, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, x):
        x = x.reshape(-1,2*32*32)
        x = self.fc(x)
        x = x.view(x.size(0), 32, 8, 8)
        x = F.relu(self.conv_trans1(x))
        x = torch.sigmoid(self.conv_trans2(x))
        return x

class Input_VModel(nn.Module):
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_VModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2, 2*32*32)
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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
        z = self.linear1(x)
        z = self.relu(z)
        z = z.view(-1, 2,32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z


class Input_VModel2_0504(nn.Module):
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_VModel2_0504, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(2, 8*32*32)
        self.linear2 = nn.Linear(8*32*32, 2*32*32)
        self.norm1 = nn.LayerNorm(8*32*32)
        self.norm2 = nn.LayerNorm(2*32*32)
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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
        z = self.linear1(x)
        z = self.relu(z)
        z = self.norm1(z)
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        z = z.view(-1, 2,32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z


class Input_2VModel(nn.Module):
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_2VModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(4, 8*32*32)
        self.linear2 = nn.Linear(8*32*32, 2*32*32)
        self.norm1 = nn.LayerNorm(8*32*32)
        self.norm2 = nn.LayerNorm(2*32*32)
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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
        z = self.linear1(x)
        z = self.relu(z)
        z = self.norm1(z)
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        z = z.view(-1, 2,32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_For(nn.Module):
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_Mix_For, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(6, 8*32*32)
        self.linear2 = nn.Linear(8*32*32, 2*32*32)
        self.norm1 = nn.LayerNorm(8*32*32)
        self.norm2 = nn.LayerNorm(2*32*32)
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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
        z = self.linear1(x)
        z = self.relu(z)
        z = self.norm1(z)
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        z = z.view(-1, 2,32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide(nn.Module): #速度、入り口、出口の位置の情報を分割して入力する
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_Mix_divide, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 8*32*32)
        self.linear1_b = nn.Linear(2, 8*32*32)
        self.linear1_c = nn.Linear(2, 8*32*32)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(8*32*32)
        self.norm1_b = nn.LayerNorm(8*32*32)
        self.norm1_c = nn.LayerNorm(8*32*32)
        
        # Define other layers
        self.linear2 = nn.Linear(8*32*32, 2*32*32)
        self.norm2 = nn.LayerNorm(2*32*32)

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z = z_a + z_b + z_c
        
        # Further processing
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)

        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_no_sum(nn.Module): #速度、入り口、出口の位置の情報を分割して入力した後、足さずにtorch.catで結合する
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_Mix_divide_no_sum, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 4*32*32)
        self.linear1_b = nn.Linear(2, 4*32*32)
        self.linear1_c = nn.Linear(2, 4*32*32)
        self.linear2_a = nn.Linear(4*32*32,8*32*32)
        self.linear2_b = nn.Linear(4*32*32,8*32*32)
        self.linear2_c = nn.Linear(4*32*32,8*32*32)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(4*32*32)
        self.norm1_b = nn.LayerNorm(4*32*32)
        self.norm1_c = nn.LayerNorm(4*32*32)
        self.norm2_a = nn.LayerNorm(8*32*32)
        self.norm2_b = nn.LayerNorm(8*32*32)
        self.norm2_c = nn.LayerNorm(8*32*32)
        
        # Define other layers
        self.linear2 = nn.Linear(8*32*32*3, 4*32*32)
        self.linear3 = nn.Linear(4*32*32, 2*32*32)
        self.norm2 = nn.LayerNorm(4*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        z_a = self.linear2_a(z_a)
        z_b = self.linear2_b(z_b)
        z_c = self.linear2_c(z_c)
        
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        
        z_a = self.norm2_a(z_a)
        z_b = self.norm2_b(z_b)
        z_c = self.norm2_c(z_c)
        # Combine the processed features
        z = torch.cat([z_a,z_b,z_c],dim=1)
        
        # Further processing
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)

        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_CNN_cat(nn.Module): #速度、入り口、出口の位置の情報を分割して入力した後,それぞれをCNNで畳み込んで特徴量を捉える
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_Mix_divide_CNN_cat, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 4*32*32)
        self.linear1_b = nn.Linear(2, 4*32*32)
        self.linear1_c = nn.Linear(2, 4*32*32)
        self.linear2_a = nn.Linear(4*32*32,2*32*32)
        self.linear2_b = nn.Linear(4*32*32,2*32*32)
        self.linear2_c = nn.Linear(4*32*32,2*32*32)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(4*32*32)
        self.norm1_b = nn.LayerNorm(4*32*32)
        self.norm1_c = nn.LayerNorm(4*32*32)
        self.norm2_a = nn.LayerNorm(2*32*32)
        self.norm2_b = nn.LayerNorm(2*32*32)
        self.norm2_c = nn.LayerNorm(2*32*32)

        self.conv_cat = nn.Conv2d(in_channels=6, out_channels=2, kernel_size=3, padding=1)
        
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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        z_a = self.linear2_a(z_a)
        z_b = self.linear2_b(z_b)
        z_c = self.linear2_c(z_c)
        
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        
        z_a = self.norm2_a(z_a)
        z_b = self.norm2_b(z_b)
        z_c = self.norm2_c(z_c)
        
        # Reshape z_a, z_b, z_c to (batch_size, channels, height, width) before stacking
        z_a = z_a.view(-1, 2, 32, 32)
        z_b = z_b.view(-1, 2, 32, 32)
        z_c = z_c.view(-1, 2, 32, 32)

        # Stack z_a, z_b, z_c along the channel dimension (assume each has shape [batch, channels, height, width])
        z = torch.cat([z_a, z_b, z_c], dim=1)
        z = self.conv_cat(z)

        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_all_linear(nn.Module): #速度、入り口、出口の位置の情報を分割して入力して結合前の線型結合層を深くする
    def __init__(self, UNet):
        super(Input_Mix_divide_all_linear, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 8*32*32)
        self.linear1_b = nn.Linear(2, 4*32*32)
        self.linear1_c = nn.Linear(2, 4*32*32)
        self.linear2_a = nn.Linear(8*32*32,16*32*32)
        self.linear2_b = nn.Linear(4*32*32,8*32*32)
        self.linear2_c = nn.Linear(4*32*32,8*32*32)
        self.linear3_a = nn.Linear(16*32*32,32*32*32)
        self.linear3_b = nn.Linear(8*32*32,16*32*32)
        self.linear3_c = nn.Linear(8*32*32,16*32*32)
        self.linear4_a = nn.Linear(32*32*32,16*32*32)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(8*32*32)
        self.norm1_b = nn.LayerNorm(4*32*32)
        self.norm1_c = nn.LayerNorm(4*32*32)
        self.norm2_a = nn.LayerNorm(16*32*32)
        self.norm2_b = nn.LayerNorm(8*32*32)
        self.norm2_c = nn.LayerNorm(8*32*32)
        self.norm3_a = nn.LayerNorm(32*32*32)
        self.norm3_b = nn.LayerNorm(16*32*32)
        self.norm3_c = nn.LayerNorm(16*32*32)
        self.norm4_a = nn.LayerNorm(16*32*32)
        
        # Define other layers
        self.linear2 = nn.Linear(16*32*32, 16*32*32)
        self.linear3 = nn.Linear(16*32*32, 8*32*32)
        self.linear4 = nn.Linear(8*32*32, 2*32*32)
        self.norm2 = nn.LayerNorm(16*32*32)
        self.norm3 = nn.LayerNorm(8*32*32)
        self.norm4 = nn.LayerNorm(2*32*32)

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        z_a = self.linear2_a(z_a)
        z_b = self.linear2_b(z_b)
        z_c = self.linear2_c(z_c)
        z_a = self.relu(z_a)        
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        z_a = self.norm2_a(z_a)
        z_b = self.norm2_b(z_b)
        z_c = self.norm2_c(z_c)
        z_a = self.linear3_a(z_a)
        z_b = self.linear3_b(z_b)
        z_c = self.linear3_c(z_c)
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        z_a = self.norm3_a(z_a)
        z_b = self.norm3_b(z_b)
        z_c = self.norm3_c(z_c)
        z_a = self.linear4_a(z_a)
        z_a = self.relu(z_a)
        z_a = self.norm4_a(z_a)

        # Combine the processed features
        z = z_a + z_b + z_c
        
        # Further processing
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        z = self.linear4(z)
        z = self.relu(z)
        z = self.norm4(z)

        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_add_CNN(nn.Module): #速度、入り口、出口の位置の情報を分割して入力して結合前の層にCNN層を追加する
    def __init__(self, UNet):
        super(Input_Mix_divide_add_CNN, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 8*32*32)
        self.linear1_b = nn.Linear(2, 8*32*32)
        self.linear1_c = nn.Linear(2, 8*32*32)
        self.linear2_a = nn.Linear(8*32*32,2*32*32)
        self.linear2_b = nn.Linear(8*32*32,2*32*32)
        self.linear2_c = nn.Linear(8*32*32,2*32*32)

        #結合前の変更部分
        self.conv1_a = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.conv1_b = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.conv1_c = nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1)
        self.conv2_a = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2_b = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv2_c = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        self.conv3_a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3_b = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv3_c = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.conv4_a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4_b = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.conv4_c = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn1_a = nn.BatchNorm2d(8)
        self.bn1_b = nn.BatchNorm2d(8)
        self.bn1_c = nn.BatchNorm2d(8)
        self.bn2_a = nn.BatchNorm2d(16)
        self.bn2_b = nn.BatchNorm2d(16)
        self.bn2_c = nn.BatchNorm2d(16)
        self.bn3_a = nn.BatchNorm2d(32)
        self.bn3_b = nn.BatchNorm2d(32)
        self.bn3_c = nn.BatchNorm2d(32)
        self.bn4_a = nn.BatchNorm2d(64)
        self.bn4_b = nn.BatchNorm2d(64)
        self.bn4_c = nn.BatchNorm2d(64)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(8*32*32)
        self.norm1_b = nn.LayerNorm(8*32*32)
        self.norm1_c = nn.LayerNorm(8*32*32)
        self.norm2_a = nn.LayerNorm(2*32*32)
        self.norm2_b = nn.LayerNorm(2*32*32)
        self.norm2_c = nn.LayerNorm(2*32*32)
        
        # Define other layers
        self.linear2 = nn.Linear(64*8*8, 32*32*32)
        self.linear3 = nn.Linear(32*32*32, 16*32*32)
        self.linear4 = nn.Linear(16*32*32, 8*32*32)
        self.linear5 = nn.Linear(8*32*32, 2*32*32)
        self.norm2 = nn.LayerNorm(32*32*32)
        self.norm3 = nn.LayerNorm(16*32*32)
        self.norm4 = nn.LayerNorm(8*32*32)
        self.norm5 = nn.LayerNorm(2*32*32)

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        z_a = self.linear2_a(z_a)
        z_b = self.linear2_b(z_b)
        z_c = self.linear2_c(z_c)
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        z_a = self.norm2_a(z_a)
        z_b = self.norm2_b(z_b)
        z_c = self.norm2_c(z_c)

        z_a = z_a.view(-1, 2, 32, 32)
        z_b = z_b.view(-1, 2, 32, 32)
        z_c = z_c.view(-1, 2, 32, 32)

        z_a = self.conv1_a(z_a)
        z_b = self.conv1_b(z_b)
        z_c = self.conv1_c(z_c)
        z_a = self.bn1_a(z_a)
        z_b = self.bn1_b(z_b)
        z_c = self.bn1_c(z_c)
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        z_a = self.conv2_a(z_a)
        z_b = self.conv2_b(z_b)
        z_c = self.conv2_c(z_c)
        z_a = self.bn2_a(z_a)
        z_b = self.bn2_b(z_b)
        z_c = self.bn2_c(z_c)
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        z_a = self.pooling(z_a)
        z_b = self.pooling(z_b)
        z_c = self.pooling(z_c)
        z_a = self.conv3_a(z_a)
        z_b = self.conv3_b(z_b)
        z_c = self.conv3_c(z_c)
        z_a = self.bn3_a(z_a)
        z_b = self.bn3_b(z_b)
        z_c = self.bn3_c(z_c)
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        z_a = self.conv4_a(z_a)
        z_b = self.conv4_b(z_b)
        z_c = self.conv4_c(z_c)
        z_a = self.bn4_a(z_a)
        z_b = self.bn4_b(z_b)
        z_c = self.bn4_c(z_c)
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)
        z_a = self.pooling(z_a)
        z_b = self.pooling(z_b)
        z_c = self.pooling(z_c)

        # Combine the processed features
        z = z_a + z_b + z_c
        z = z.view(-1,64*8*8)
        
        # Further processing
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        z = self.linear4(z)
        z = self.relu(z)
        z = self.norm4(z)
        z = self.linear5(z)
        z = self.relu(z)
        z = self.norm5(z)

        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_rasiudal(nn.Module): #速度、入り口、出口の位置の情報を分割して入力する,残差接続のように後でもう一度足し合わせる
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_Mix_divide_rasiudal, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 8*32*32)
        self.linear1_b = nn.Linear(2, 8*32*32)
        self.linear1_c = nn.Linear(2, 8*32*32)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(8*32*32)
        self.norm1_b = nn.LayerNorm(8*32*32)
        self.norm1_c = nn.LayerNorm(8*32*32)
        
        # Define other layers
        self.linear2 = nn.Linear(8*32*32, 8*32*32)
        self.norm2 = nn.LayerNorm(8*32*32)
        self.linear3 = nn.Linear(8*32*32, 2*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z = z_a + z_b + z_c
        
        # Further processing
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)

        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z_ = z_a + z_b + z_c
        # Further processing
        z_ = self.linear2(z_)
        z_ = self.relu(z_)
        z_ = self.norm2(z_)

        z = z + z_
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z_ = z_a + z_b + z_c
        
        # Further processing
        z_ = self.linear2(z_)
        z_ = self.relu(z_)
        z_ = self.norm2(z_)
        z = z + z_
        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_rasiudal_6(nn.Module): #速度、入り口、出口の位置の情報を分割して入力する,残差接続のように後でもう一度足し合わせるのをできるだけ沢山行う
    #def __init__(self):
    def __init__(self, UNet):
        super(Input_Mix_divide_rasiudal_6, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 8*32*32)
        self.linear1_b = nn.Linear(2, 8*32*32)
        self.linear1_c = nn.Linear(2, 8*32*32)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(8*32*32)
        self.norm1_b = nn.LayerNorm(8*32*32)
        self.norm1_c = nn.LayerNorm(8*32*32)
        
        # Define other layers
        self.linear2 = nn.Linear(8*32*32, 8*32*32)
        self.norm2 = nn.LayerNorm(8*32*32)
        self.linear3 = nn.Linear(8*32*32, 2*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

        self.T = 1000 #ノイズを加える回数
        self.beta_1 = 1e-6 #t=1のノイズの大きさ(最初1.0e-4)
        self.beta_T = 2.0e-4 #t=Tのノイズの大きさ(最初0.02)
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列
        self.repeat = 6 #残差接続を行う回数
    def diffusion_process(self, x0,t=None):
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0, device=self.device) #ノイズを生成
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def skip_connect(self,z,x):
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features

        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z_ = z_a + z_b + z_c
        # Further processing
        z_ = self.linear2(z_)
        z_ = self.relu(z_)
        z_ = self.norm2(z_)

        z = z + z_

        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        return z

    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z = z_a + z_b + z_c
        
        # Further processing
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)

        for i in range(self.repeat):
            z = self.skip_connect(z,x)

        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z


class Input_Mix_divide_rasiudal_InOut_Fusion(nn.Module): #速度、入り口、出口の位置の情報を分割して入力する,入り口と出口を先に結合
    def __init__(self, UNet):
        super(Input_Mix_divide_rasiudal_InOut_Fusion, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.Linear(2, 8*32*32)
        self.linear1_b = nn.Linear(2, 8*32*32)
        self.linear1_c = nn.Linear(2, 8*32*32)
        
        # Activation and normalization layers
        self.relu = nn.ReLU()
        self.norm1_a = nn.LayerNorm(8*32*32)
        self.norm1_b = nn.LayerNorm(8*32*32)
        self.norm1_c = nn.LayerNorm(8*32*32)
        
        # Define other layers
        self.linear2 = nn.Linear(8*32*32, 8*32*32)
        self.norm2 = nn.LayerNorm(8*32*32)
        self.linear3 = nn.Linear(8*32*32, 2*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

        self.T = 1000 #ノイズを加える回数
        self.beta_1 = 1e-6 #t=1のノイズの大きさ(最初1.0e-4)
        self.beta_T = 2.0e-4 #t=Tのノイズの大きさ(最初0.02)
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列
        self.repeat = 6 #残差接続を行う回数
    def diffusion_process(self, x0,t=None):
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0, device=self.device) #ノイズを生成
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def skip_connect(self,z,x):
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features

        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z_ = z_b + z_c
        # Further processing
        z_ = self.linear2(z_)
        z_ = self.relu(z_)
        z_ = self.norm2(z_)

        z_ = z_a + z_
        z_ = self.linear2(z_)
        z_ = self.relu(z_)
        z_ = self.norm2(z_)
        
        z = z + z_
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        return z

    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a(x_a)
        z_b = self.linear1_b(x_b)
        z_c = self.linear1_c(x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a(z_a)
        z_b = self.norm1_b(z_b)
        z_c = self.norm1_c(z_c)

        # Combine the processed features
        z = z_b + z_c
        
        # Further processing
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)
        z += z_a 
        z = self.linear2(z)
        z = self.relu(z)
        z = self.norm2(z)

        for i in range(self.repeat):
            z = self.skip_connect(z,x)

        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_rasiudal_WeightedAdder(nn.Module): #速度、入り口、出口の位置の情報を分割して入力する,足す時に重みをつける
    def __init__(self, UNet,repeat=6):
        super(Input_Mix_divide_rasiudal_WeightedAdder, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.repeat = repeat #残差接続を行う回数
        self.UNet = UNet
        self.relu = nn.ReLU()
        self.linear1_a = nn.ModuleList()
        self.linear1_b = nn.ModuleList()
        self.linear1_c = nn.ModuleList()
        self.norm1_a = nn.ModuleList()
        self.norm1_b = nn.ModuleList()
        self.norm1_c = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        self.weight1 = nn.ModuleList()
        self.weight2 = nn.ModuleList()
        self.weight3 = nn.ModuleList()
        #for i in range(repeat+1):
        self.linear1_a.append(nn.Linear(2, 8*32*32))
        self.linear1_b.append(nn.Linear(2, 8*32*32))
        self.linear1_c.append(nn.Linear(2, 8*32*32))
        self.norm1_a.append(nn.LayerNorm(8*32*32))
        self.norm1_b.append(nn.LayerNorm(8*32*32))
        self.norm1_c.append(nn.LayerNorm(8*32*32))
        self.linear2.append(nn.Linear(8*32*32, 8*32*32))
        self.norm2.append(nn.LayerNorm(8*32*32))
        self.weight1.append(nn.Linear(8*32*32, 8*32*32))
        self.weight2.append(nn.Linear(8*32*32, 8*32*32))
        self.weight3.append(nn.Linear(8*32*32, 8*32*32))
            #if i != 0:
            #    self.linear2.append(nn.Linear(8*32*32, 8*32*32))
            #    self.norm2.append(nn.LayerNorm(8*32*32))
        
        # Define other layers
        self.linear3 = nn.Linear(8*32*32, 2*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def skip_connect(self,z,x,i,light=True):
        if light:
            i=0
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features

        # Process each pair through its linear layer
        z_a = self.linear1_a[i](x_a)
        z_b = self.linear1_b[i](x_b)
        z_c = self.linear1_c[i](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[i](z_a)
        z_b = self.norm1_b[i](z_b)
        z_c = self.norm1_c[i](z_c)

        # Combine the processed features
        z_ = self.weight1[i](z_a) + self.weight2[i](z_b) + self.weight3[i](z_c)
        # Further processing
        z_ = self.linear2[i](z_)
        z_ = self.relu(z_)
        z_ = self.norm2[i](z_)

        z = z + z_

        z = self.linear2[i](z)
        #z = self.linear2[i+1](z)
        z = self.relu(z)
        z = self.norm2[i](z)
        #z = self.norm2[i+1](z)
        return z

    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a[0](x_a)
        z_b = self.linear1_b[0](x_b)
        z_c = self.linear1_c[0](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[0](z_a)
        z_b = self.norm1_b[0](z_b)
        z_c = self.norm1_c[0](z_c)

        # Combine the processed features
        z = self.weight1[0](z_a) + self.weight2[0](z_b) + self.weight3[0](z_c)
        
        # Further processing
        z = self.linear2[0](z)
        z = self.relu(z)
        z = self.norm2[0](z)

        for i in range(self.repeat):
            z = self.skip_connect(z,x,i+1)

        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z



class Input_Mix_divide_skip_detailed(nn.Module): #スキップ接続の重みを共有せずに一つづつ定義して6層用意する
    def __init__(self, UNet,repeat=6):
        super(Input_Mix_divide_skip_detailed, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        
        # Activation and normalization layers
        self.repeat = repeat #残差接続を行う回数
        self.relu = nn.ReLU()
        
        self.linear3 = nn.Linear(8*32*32, 2*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

        self.linear1_a = nn.ModuleList()
        self.linear1_b = nn.ModuleList()
        self.linear1_c = nn.ModuleList()
        self.norm1_a = nn.ModuleList()
        self.norm1_b = nn.ModuleList()
        self.norm1_c = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(repeat+1):
            self.linear1_a.append(nn.Linear(2, 8*32*32))
            self.linear1_b.append(nn.Linear(2, 8*32*32))
            self.linear1_c.append(nn.Linear(2, 8*32*32))
            self.norm1_a.append(nn.LayerNorm(8*32*32))
            self.norm1_b.append(nn.LayerNorm(8*32*32))
            self.norm1_c.append(nn.LayerNorm(8*32*32))
            self.linear2.append(nn.Linear(8*32*32, 8*32*32))
            self.norm2.append(nn.LayerNorm(8*32*32))

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def skip_connect(self,z,x,i):
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features

        # Process each pair through its linear layer
        z_a = self.linear1_a[i](x_a)
        z_b = self.linear1_b[i](x_b)
        z_c = self.linear1_c[i](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[i](z_a)
        z_b = self.norm1_b[i](z_b)
        z_c = self.norm1_c[i](z_c)

        # Combine the processed features
        z_ = z_a + z_b + z_c
        # Further processing
        z_ = self.linear2[i](z_)
        z_ = self.relu(z_)
        z_ = self.norm2[i](z_)

        z = z + z_

        z = self.linear2[i](z)
        z = self.relu(z)
        z = self.norm2[i](z)
        return z

    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a[0](x_a)
        z_b = self.linear1_b[0](x_b)
        z_c = self.linear1_c[0](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[0](z_a)
        z_b = self.norm1_b[0](z_b)
        z_c = self.norm1_c[0](z_c)

        # Combine the processed features
        z = z_a + z_b + z_c
        
        # Further processing
        z = self.linear2[0](z)
        z = self.relu(z)
        z = self.norm2[0](z)

        for i in range(1,self.repeat):
            z = self.skip_connect(z,x,i)

        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_skip_detailed_12(nn.Module): #スキップ接続の重みを共有せずに一つづつ定義して6層用意する
    def __init__(self, UNet,repeat=12):
        super(Input_Mix_divide_skip_detailed_12, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        
        # Activation and normalization layers
        self.repeat = repeat #残差接続を行う回数
        self.relu = nn.ReLU()
        
        self.linear3 = nn.Linear(8*32*32, 2*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

        self.linear1_a = nn.ModuleList()
        self.linear1_b = nn.ModuleList()
        self.linear1_c = nn.ModuleList()
        self.norm1_a = nn.ModuleList()
        self.norm1_b = nn.ModuleList()
        self.norm1_c = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(repeat+1):
            self.linear1_a.append(nn.Linear(2, 8*32*32))
            self.linear1_b.append(nn.Linear(2, 8*32*32))
            self.linear1_c.append(nn.Linear(2, 8*32*32))
            self.norm1_a.append(nn.LayerNorm(8*32*32))
            self.norm1_b.append(nn.LayerNorm(8*32*32))
            self.norm1_c.append(nn.LayerNorm(8*32*32))
            self.linear2.append(nn.Linear(8*32*32, 8*32*32))
            self.norm2.append(nn.LayerNorm(8*32*32))

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def skip_connect(self,z,x,i):
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features

        # Process each pair through its linear layer
        z_a = self.linear1_a[i](x_a)
        z_b = self.linear1_b[i](x_b)
        z_c = self.linear1_c[i](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[i](z_a)
        z_b = self.norm1_b[i](z_b)
        z_c = self.norm1_c[i](z_c)

        # Combine the processed features
        z_ = z_a + z_b + z_c
        # Further processing
        z_ = self.linear2[i](z_)
        z_ = self.relu(z_)
        z_ = self.norm2[i](z_)

        z = z + z_

        z = self.linear2[i](z)
        z = self.relu(z)
        z = self.norm2[i](z)
        return z

    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a[0](x_a)
        z_b = self.linear1_b[0](x_b)
        z_c = self.linear1_c[0](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[0](z_a)
        z_b = self.norm1_b[0](z_b)
        z_c = self.norm1_c[0](z_c)

        # Combine the processed features
        z = z_a + z_b + z_c
        
        # Further processing
        z = self.linear2[0](z)
        z = self.relu(z)
        z = self.norm2[0](z)

        for i in range(1,self.repeat):
            z = self.skip_connect(z,x,i)

        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class Input_Mix_divide_skip_detailed_18(nn.Module): #スキップ接続の重みを共有せずに一つづつ定義して6層用意する
    def __init__(self, UNet,repeat=18):
        super(Input_Mix_divide_skip_detailed_18, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()
        
        # Activation and normalization layers
        self.repeat = repeat #残差接続を行う回数
        self.relu = nn.ReLU()
        
        self.linear3 = nn.Linear(8*32*32, 2*32*32)
        self.norm3 = nn.LayerNorm(2*32*32)

        self.linear1_a = nn.ModuleList()
        self.linear1_b = nn.ModuleList()
        self.linear1_c = nn.ModuleList()
        self.norm1_a = nn.ModuleList()
        self.norm1_b = nn.ModuleList()
        self.norm1_c = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(repeat+1):
            self.linear1_a.append(nn.Linear(2, 8*32*32))
            self.linear1_b.append(nn.Linear(2, 8*32*32))
            self.linear1_c.append(nn.Linear(2, 8*32*32))
            self.norm1_a.append(nn.LayerNorm(8*32*32))
            self.norm1_b.append(nn.LayerNorm(8*32*32))
            self.norm1_c.append(nn.LayerNorm(8*32*32))
            self.linear2.append(nn.Linear(8*32*32, 8*32*32))
            self.norm2.append(nn.LayerNorm(8*32*32))

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def skip_connect(self,z,x,i):
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features

        # Process each pair through its linear layer
        z_a = self.linear1_a[i](x_a)
        z_b = self.linear1_b[i](x_b)
        z_c = self.linear1_c[i](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[i](z_a)
        z_b = self.norm1_b[i](z_b)
        z_c = self.norm1_c[i](z_c)

        # Combine the processed features
        z_ = z_a + z_b + z_c
        # Further processing
        z_ = self.linear2[i](z_)
        z_ = self.relu(z_)
        z_ = self.norm2[i](z_)

        z = z + z_

        z = self.linear2[i](z)
        z = self.relu(z)
        z = self.norm2[i](z)
        return z

    def forward(self, x):
         # Split input tensor into three pairs of features
        x_a = x[:, :2]   # First two features
        x_b = x[:, 2:4]  # Next two features
        x_c = x[:, 4:]   # Last two features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a[0](x_a)
        z_b = self.linear1_b[0](x_b)
        z_c = self.linear1_c[0](x_c)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)
        z_b = self.relu(z_b)
        z_c = self.relu(z_c)

        z_a = self.norm1_a[0](z_a)
        z_b = self.norm1_b[0](z_b)
        z_c = self.norm1_c[0](z_c)

        # Combine the processed features
        z = z_a + z_b + z_c
        
        # Further processing
        z = self.linear2[0](z)
        z = self.relu(z)
        z = self.norm2[0](z)

        for i in range(1,self.repeat):
            z = self.skip_connect(z,x,i)

        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 32, 32)
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z

class To_high_resolution(nn.Module): #16*16から32*32に高解像度化するモデル
    def __init__(self, UNet):
        super(To_high_resolution, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def forward(self, x):
        z,t,noise = self.diffusion_process(x)
        z = self.UNet(z,t) 
        return z

class To_high_resolution_with_inlet_value(nn.Module): #16*16から32*32に高解像度化するモデル
    def __init__(self, UNet):
        super(To_high_resolution_with_inlet_value, self).__init__()
        self.repeat = 12
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.UNet = UNet
        self.relu = nn.ReLU()

        self.T = 1000 #ノイズを加える回数
        self.beta_1 = 1e-6 #t=1のノイズの大きさ(最初1.0e-4)
        self.beta_T = 2.0e-4 #t=Tのノイズの大きさ(最初0.02)
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列
        self.linear3 = nn.Linear(8*16*16, 2*16*16)
        self.norm3 = nn.LayerNorm(2*16*16)

        self.linear1_a = nn.ModuleList()
        self.norm1_a = nn.ModuleList()
        self.linear2 = nn.ModuleList()
        self.norm2 = nn.ModuleList()
        for i in range(self.repeat+1):
            self.linear1_a.append(nn.Linear(2, 8*16*16))
            self.norm1_a.append(nn.LayerNorm(8*16*16))
            self.linear2.append(nn.Linear(8*16*16, 8*16*16))
            self.norm2.append(nn.LayerNorm(8*16*16))

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
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def skip_connect(self,z,x,i):
        x_a = x[:, :2]   # First two features

        # Process each pair through its linear layer
        z_a = self.linear1_a[i](x_a)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)

        z_a = self.norm1_a[i](z_a)

        # Combine the processed features
        # Further processing
        z_ = self.linear2[i](z_a)

        z = z + z_

        z = self.linear2[i](z)
        z = self.relu(z)
        z = self.norm2[i](z)
        return z

    def diffusion_process(self, x0,t=None):
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0, device=self.device) #ノイズを生成
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ

    def forward(self, value_by_mesh,x_a):
         # Split input tensor into three pairs of features
        
        # Process each pair through its linear layer
        z_a = self.linear1_a[0](x_a)
        
        # Apply activation and normalization
        z_a = self.relu(z_a)

        z_a = self.norm1_a[0](z_a)
        # Further processing
        z = self.linear2[0](z_a)
        z = self.relu(z)
        z = self.norm2[0](z)

        for i in range(1,self.repeat):
            z = self.skip_connect(z,x_a,i)

        z = self.linear3(z)
        z = self.relu(z)
        z = self.norm3(z)
        # Reshape for diffusion process
        z = z.view(-1, 2, 16, 16)
        z = z + value_by_mesh
        z,t,noise = self.diffusion_process(z)
        z = self.UNet(z,t) 
        return z
