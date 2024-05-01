import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from tqdm import tqdm

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # 16x16 -> 8x8
        self.fc_mu = nn.Linear(8*8*32, 2*32*32)
        self.fc_logvar = nn.Linear(8*8*32, 2*32*32)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        mu = mu.view(-1,2,32,32)
        logvar = self.fc_logvar(x)
        logvar = logvar.view(-1,2,32,32)
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
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
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
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=2, c_out=2, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(2, 64)
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
        self.outc = nn.Conv2d(64, 2, kernel_size=1)

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
        output = self.unet_forwad(x, t)
        return output


class UNet_conditional(UNet):
    def __init__(self, c_in=2, c_out=2, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)


class Vae_Diffusion_Model(nn.Module):
    #def __init__(self):
    def __init__(self, Encoder, Decoder,UNet):
        super(Vae_Diffusion_Model, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.UNet = UNet
        self.T = 1000 #ノイズを加える回数
        self.beta_1 = 1e-6 #t=1のノイズの大きさ(最初1.0e-4)
        self.beta_T = 2.0e-4 #t=Tのノイズの大きさ(最初0.02)
        self.betas = torch.linspace(self.beta_1, self.beta_T, self.T, device=self.device)#t=1からt=Tまでのノイズの大きさを線形に変化させる
        self.alphas = 1.0 - self.betas #最初の位置から今の位置までに加えるノイズの合計
        # α bar [α_bar_1, α_bar_2, ... , α_bar_T] (length = T)
        self.alpha_bars = torch.cumprod(self.alphas, dim=0) #αの配列
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(self.device)        
        z = mean + var*epsilon                          # reparameterization trick
        return z

    def diffusion_process(self, x0,t=None):
        if t is None:
            t = torch.randint(low=1, high=self.T, size=(x0.shape[0],), device=self.device) #最初に受け取る値はnoneで、その場合はランダムにtを選ぶ
        noise = torch.randn_like(x0, device=self.device) #ノイズを生成
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1) #tの値に応じてα_barを選ぶ
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise #ノイズを加える
        return xt, t, noise #ノイズを加えきった画像、tの値、ノイズ
                
    def denoising_process(self, model, img, ts):#与えられた画像からノイズを取り除く関数
        batch_size = img.shape[0] #バッチサイズを取得
        model.eval() #モデルを評価モードにする
        with torch.no_grad():
            time_step_bar = reversed(range(1, ts)) #ノイズを復元するために逆順にループ
            for t in time_step_bar:  # ts, ts-1, .... 3, 2, 1
                # 整数値のtをテンソルに変換。テンソルのサイズは(バッチサイズ, )
                time_tensor = (torch.ones(batch_size, device=self.device) * t).long()#tの値をバッチサイズ分用意
                # 現在の画像からノイズを予測
                #prediction_noise = model(img, time_tensor,v)#ノイズを予測
                prediction_noise = model(img, time_tensor)#ノイズを予測
                # 現在の画像からノイズを少し取り除く
                img = self._calc_denoising_one_step(img, time_tensor, prediction_noise)
        model.train()
        return img
    
    def _calc_denoising_one_step(self, img, time_tensor, prediction_noise): #ノイズを計算する関数
            beta = self.betas[time_tensor].reshape(-1, 1, 1, 1) #tの値に応じてノイズの大きさを選ぶ
            sqrt_alpha = torch.sqrt(self.alphas[time_tensor].reshape(-1, 1, 1, 1)) #tの値に応じてαの値を選ぶ
            alpha_bar = self.alpha_bars[time_tensor].reshape(-1, 1, 1, 1) #tの値に応じてα_barの値を選ぶ
            sigma_t = torch.sqrt(beta)
            noise = torch.randn_like(img, device=self.device) if time_tensor[0].item() > 1 else torch.zeros_like(img, device=self.device) #t=1の時はノイズを0にする.   
            img = 1 / sqrt_alpha * (img - (beta / (torch.sqrt(1 - alpha_bar))) * prediction_noise) + sigma_t * noise #ノイズを取り除く
            return img

    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        z,t,noise = self.diffusion_process(z)
        #noise_estimate = self.conditional_diffusion_0406(z_,t) 
        z_ = self.UNet(z,t) 
        #z,_,_ = self.diffusion_process(z,t=self.T-1)
        #z = self.denoising_process(self.conditional_diffusion_0406,z,self.T-1)
        x_hat = self.Decoder(z_)
        return x_hat, mean, log_var

def vae_diffusion_loss_function(x, x_hat, mean, log_var):
    mse_loss = nn.MSELoss(reduction='sum')
    #reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reproduction_loss = mse_loss(x_hat, x)
    #noise_loss = mse_loss(noise,noise_estimate)
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD
def vae_loss_function(x, x_hat, mean, log_var):
    mse_loss = nn.MSELoss(reduction='sum')
    #reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    reproduction_loss = mse_loss(x_hat, x)
    KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())

    return reproduction_loss + KLD

