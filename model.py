import torch
import torch.nn as nn
import torch.nn.functional as F

class conv_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_layer, self).__init__()

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class SASA(nn.Module):
    '''
        Structure Affinity Self attention Module
    '''

    def __init__(self, in_dim):
        super(SASA, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        # self.mag_conv = nn.Conv2d(in_channels=5, out_channels=in_dim//32, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, PAF_mag):
        """
            inputs :
                x : input feature maps( B x C x H x W)
                Y : ( B x C x H x W), 1 denotes connectivity, 0 denotes non-connectivity
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, height, width = X.size()

        # PAF_mag = PAF_mag.contiguous()

        # Y = self.structure_encoder(PAF_mag, height, width)

        # connectivity_mask_vec = self.mag_conv(Y).view(m_batchsize, -1, width * height)  # B * C * (W*H)
        # affinity = torch.bmm(connectivity_mask_vec.permute(0, 2, 1),connectivity_mask_vec)  # B * (W*H) * (W*H)
        # affinity_centered = affinity - torch.mean(affinity) # centering
        # affinity_sigmoid = self.sigmoid( affinity_centered)

        proj_query = self.query_conv(X).view(m_batchsize, -1, width * height).permute(0, 2, 1)  #  B * (W*H) * C
        proj_key = self.key_conv(X).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        selfatten_map = torch.bmm(proj_query, proj_key)  # B * (W*H) * (W*H)
        selfatten_centered = selfatten_map - torch.mean(selfatten_map)  # centering
        selfatten_sigmoid = self.sigmoid(selfatten_centered)

        SASA_map = selfatten_sigmoid
        # SASA_map = selfatten_sigmoid * affinity_sigmoid

        proj_value = self.value_conv(X).view(m_batchsize, -1, width * height)  # B * C * (W*H)

        out = torch.bmm(proj_value, SASA_map.permute(0, 2, 1)) # B* C *(W*H)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + X
        return out, Y


class FlowGenerator(nn.Module):
    def __init__(self, n_channels):
        super(FlowGenerator, self).__init__()

        self.Encoder = nn.Sequential(
            conv_layer(n_channels, 64),
            conv_layer(64, 64),
            nn.MaxPool2d(2),
            conv_layer(64, 128),
            conv_layer(128, 128),
            nn.MaxPool2d(2),
            conv_layer(128, 256),
            conv_layer(256, 256),
            nn.MaxPool2d(2),
            conv_layer(256, 512),
            conv_layer(512, 512),
            nn.MaxPool2d(2),
            conv_layer(512, 1024),
            conv_layer(1024, 1024),
            conv_layer(1024, 1024),
            conv_layer(1024, 1024),
            conv_layer(1024, 1024),
        )

        self.SASA = SASA(in_dim=1024)

        self.Decoder = nn.Sequential(
            conv_layer(1024, 1024),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_layer(1024, 512),
            conv_layer(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            conv_layer(512, 256),
            conv_layer(256, 256),
            conv_layer(256, 128),
            conv_layer(128, 64),
            conv_layer(64, 32),
            nn.Conv2d(32, 2, kernel_size=1, padding=0),
            nn.Tanh(),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
        )

        # dilation_ksize = 17
        # self.dilation= torch.nn.MaxPool2d(kernel_size=dilation_ksize, stride=1, padding=int((dilation_ksize - 1) / 2))

    def warp(self, x, flow, mode='bilinear', padding_mode='zeros', coff=0.2):
        n, c, h, w = x.size()
        yv, xv = torch.meshgrid([torch.arange(h), torch.arange(w)])
        xv = xv.float() / (w - 1) * 2.0 - 1
        yv = yv.float() / (h - 1) * 2.0 - 1

        '''
        grid[0,:,:,0] =
        -1, .....1
        -1, .....1
        -1, .....1

        grid[0,:,:,1] =
        -1,  -1, -1
         ;        ;
         1,   1,  1


        image  -1 ~1       -128~128 pixel
        flow   -0.4~0.4     -51.2~51.2 pixel
        '''

        if torch.cuda.is_available():
            grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0).cuda()
        else:
            grid = torch.cat((xv.unsqueeze(-1), yv.unsqueeze(-1)), -1).unsqueeze(0)
        grid_x = grid + 2 * flow * coff
        warp_x = F.grid_sample(x, grid_x, mode=mode, padding_mode=padding_mode)
        return warp_x


    def forward(self, img1, img2, coff):
        '''
        img  -1 ~ 1, NCHW
        '''
        img_concat = torch.cat((img1, img2), dim=1)
        X = self.Encoder(img_concat)

        _, _, height, width = X.size()
        flow = self.Decoder(X)
        flow = flow.permute(0, 2, 3, 1)  # [n, 2, h, w] ==> [n, h, w, 2]
        warp_x = self.warp(img1, flow, coff=coff)
        warp_x = torch.clamp(warp_x, min=-1.0, max=1.0)
        return warp_x, flow

class ResnetBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, conv_shortcut=True):
        super(ResnetBlock2D, self).__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-05, affine=True)
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.time_emb_proj = nn.Linear(1280, out_channels, bias=True)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-05, affine=True)
        self.dropout = nn.Dropout(p=0.0, inplace=False)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.nonlinearity = nn.SiLU()
        self.conv_shortcut = None
        if conv_shortcut:
            self.conv_shortcut = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1
            )

    def forward(self, input_tensor):
        hidden_states = input_tensor
        hidden_states = self.norm1(hidden_states)
        hidden_states = self.nonlinearity(hidden_states)

        hidden_states = self.conv1(hidden_states)

        hidden_states = self.norm2(hidden_states)

        hidden_states = self.nonlinearity(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        output_tensor = input_tensor + hidden_states

        return output_tensor


class Downsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Downsample2D, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=2, padding=1
        )

    def forward(self, x):
        return self.conv(x)


class Upsample2D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample2D, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)


class DownBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, has_downsample=True):
        super(DownBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_channels, out_channels, conv_shortcut=(in_channels != out_channels)),
                ResnetBlock2D(out_channels, out_channels, conv_shortcut=False),
            ]
        )
        self.has_downsample = has_downsample
        if self.has_downsample:
            self.downsamplers = nn.ModuleList([Downsample2D(out_channels, out_channels)])

    def forward(self, hidden_states):
        output_states = []
        for module in self.resnets:
            hidden_states = module(hidden_states)
            output_states.append(hidden_states)

        if self.has_downsample:
            hidden_states = self.downsamplers[0](hidden_states)
            output_states.append(hidden_states)

        return hidden_states, output_states

class UpBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, prev_output_channel, has_upsampler=True):
        super(UpBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(out_channels + prev_output_channel, out_channels),
                ResnetBlock2D(out_channels * 2, out_channels),
                ResnetBlock2D(out_channels + in_channels, out_channels),
            ]
        )
        if has_upsampler:
            self.upsamplers = nn.ModuleList([Upsample2D(out_channels, out_channels)])
        else:
            self.upsamplers = None

    def forward(self, hidden_states, res_hidden_states_tuple):
        for resnet in self.resnets:
            res_hidden_states = res_hidden_states_tuple[-1]
            res_hidden_states_tuple = res_hidden_states_tuple[:-1]
            hidden_states = torch.cat([hidden_states, res_hidden_states], dim=1)
            hidden_states = resnet(hidden_states)
        if self.upsamplers is not None:
            for upsampler in self.upsamplers:
                hidden_states = upsampler(hidden_states)
        return hidden_states


class UNetMidBlock2D(nn.Module):
    def __init__(self, in_features):
        super(UNetMidBlock2D, self).__init__()
        self.resnets = nn.ModuleList(
            [
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
                ResnetBlock2D(in_features, in_features, conv_shortcut=False),
            ]
        )

    def forward(self, hidden_states):
        hidden_states = self.resnets[0](hidden_states)
        hidden_states = self.resnets[1](hidden_states)
        return hidden_states

class FlowUNet2DModel(nn.Module):
    def __init__(self):
        super(FlowUNet2DModel, self).__init__()

        # channels = [320, 640, 1280, 320]
        # channels = [128, 256, 512, 128]
        channels = [32, 64, 128, 32]

        self.conv_in = nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                DownBlock2D(in_channels=channels[0], out_channels=channels[0]),
                DownBlock2D(in_channels=channels[0], out_channels=channels[1]),
                DownBlock2D(in_channels=channels[1], out_channels=channels[2], has_downsample=False),
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                UpBlock2D(in_channels=channels[1], out_channels=channels[2], prev_output_channel=channels[2]),
                UpBlock2D(in_channels=channels[0], out_channels=channels[1], prev_output_channel=channels[2]),
                UpBlock2D(in_channels=channels[0], out_channels=channels[0], prev_output_channel=channels[1], has_upsampler=False),
            ]
        )
        self.mid_block = UNetMidBlock2D(channels[2])
        self.conv_norm_out = nn.GroupNorm(32, channels[3], eps=1e-05, affine=True)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(channels[3], 2, kernel_size=1, stride=1, padding=0)

    def forward(self, sample):

        sample = self.conv_in(sample)

        # 3. down
        s0 = sample
        sample, [s1, s2, s3] = self.down_blocks[0](sample)
        sample, [s4, s5, s6] = self.down_blocks[1](sample)
        sample, [s7, s8] = self.down_blocks[2](sample)

        # 4. mid
        sample = self.mid_block(sample)

        # 5. up
        sample = self.up_blocks[0](hidden_states=sample, res_hidden_states_tuple=[s6, s7, s8])
        sample = self.up_blocks[1](hidden_states=sample, res_hidden_states_tuple=[s3, s4, s5])
        sample = self.up_blocks[2](hidden_states=sample, res_hidden_states_tuple=[s0, s1, s2])

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        return sample
