import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# marks = skips
class DirectTemporalNeRF(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, zero_canonical=True, is_straightforward = False, is_ViT = False):
        super(DirectTemporalNeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.use_viewdirs = use_viewdirs
        self.memory = memory
        self.embed_fn = embed_fn
        self.zero_canonical = zero_canonical
        self.is_straightforward = is_straightforward
        self.is_ViT = is_ViT
        if self.is_straightforward:
            self._occ = NeRFOriginal(D=D, W=W, input_ch=(input_ch + input_ch_time), input_ch_views=input_ch_views,
                                 input_ch_time=input_ch_time, output_ch=output_ch, skips=skips,
                                 use_viewdirs=use_viewdirs, memory=memory, embed_fn=embed_fn, output_color_ch=3)
        elif self.is_ViT:
            self._occ = ViTNeRF(embed_dim=(input_ch + input_ch_time + input_ch_views))
        else:
            self._occ = NeRFOriginal(D=D, W=W, input_ch=(input_ch), input_ch_views=input_ch_views,
                                input_ch_time=input_ch_time, output_ch=output_ch, skips=skips,
                                use_viewdirs=use_viewdirs, memory=memory, embed_fn=embed_fn, output_color_ch=3)
            self._time, self._time_out = self.create_time_net()

    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):
            if i in self.memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    def query_time(self, new_pts, t, net, net_final):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(net):
            h = net[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)

        return net_final(h)

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        t = ts[0]
        assert len(torch.unique(t[:, :1])) == 1, "Only accepts all points from same time"
        cur_time = t[0, 0]
        if self.is_straightforward:
            input_pts = self.embed_fn(torch.cat([input_pts, t], dim=-1))
            dx = torch.zeros_like(input_pts[:, :3])
            out, _ = self._occ(torch.cat([input_pts, input_views], dim=-1), t)
        elif self.is_ViT:
            dx = torch.zeros_like(input_pts[:, :3])
            out, _ = self._occ(torch.cat([input_pts, t, input_views], dim=-1), t)
        else:
            if cur_time == 0. and self.zero_canonical:
                dx = torch.zeros_like(input_pts[:, :3])
            else:
                dx = self.query_time(input_pts, t, self._time, self._time_out)
                input_pts_orig = input_pts[:, :3]

                input_pts = self.embed_fn(input_pts_orig + dx)
            out, _ = self._occ(torch.cat([input_pts, input_views], dim=-1), t)
        return out, dx

class NeRFOriginal(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, output_color_ch=3, zero_canonical=True):
        # Depth : 网络的深度(论文中为8)
        # Width : 网络的维度(论文中为256)
        # input_channel : 3(Position，即normalized3维笛卡尔坐标)
        # input_channel_views : 3(Direction，即normalized3维笛卡尔坐标)
        # output_ch : 输出的维度
        # marks : 需要重新加入坐标输入的维度(依照论文为4)
        # use_viewdirs : 是否如论文中的网络一样使用feature : sigma
        super(NeRFOriginal, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skip = skips
        self.use_viewdirs = use_viewdirs

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in self.skip:
                in_channels += input_ch

            layers += [layer(in_channels, W)]
        self.pts_linears = nn.ModuleList(layers)
        self.views_linears = nn.ModuleList(
            [nn.Linear(input_ch_views + W, W // 2)]
        )
        
        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W // 2, output_color_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)
    
    def forward(self, x, ts=1):
        input_positions, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        
        # y : 传递变量
        y = input_positions
        for i, l in enumerate(self.pts_linears):
            y = self.pts_linears[i](y)
            y = F.relu(y)
            if i in self.skip:
                y = torch.cat([input_positions, y], dim=-1)
        
        if self.use_viewdirs:
            alpha = self.alpha_linear(y)
            feature = self.feature_linear(y)
            y = torch.cat([feature, input_views], dim=-1)
            
            for i, l in enumerate(self.views_linears):
                y = self.views_linears[i](y)
                # 论文中下面使用sigmoid激活函数
                y = F.relu(y)
            
            rgb = self.rgb_linear(y)
            outputs = torch.cat([rgb, alpha], dim=-1)
        else:
            outputs = self.output_linear(y)
        
        return outputs, torch.zeros_like(input_positions[:, :3])
    
    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))    
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))
        
        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))

class ViTEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(ViTEncoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, mask):
        # x: 输入张量，形状为 [batch_size, seq_length, embed_dim]
        # mask: 注意力掩码，形状为 [1, 1, seq_length, seq_length]

        # 自我注意力
        x = self.self_attn(x, x, x, attn_mask=mask)[0]
        x = self.norm1(x)

        # 前馈网络
        x = self.feed_forward(x)
        x = self.norm2(x)

        return x

class ViTEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout):
        super(ViTEncoder, self).__init__()
        self.layers = nn.ModuleList([
            ViTEncoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, mask):
        # x: 输入张量，形状为 [batch_size, seq_length, embed_dim]
        # mask: 注意力掩码，形状为 [1, 1, seq_length, seq_length]
        for layer in self.layers:
            x = layer(x, mask)
        return x

class ViTDecoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout):
        super(ViTDecoderLayer, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x: 输入张量，形状为 [batch_size, seq_length, embed_dim]
        # enc_output: 编码器输出，形状为 [batch_size, seq_length, embed_dim]
        # src_mask: 编码器注意力掩码，形状为 [1, 1, seq_length, seq_length]
        # tgt_mask: 解码器注意力掩码，形状为 [1, 1, seq_length, seq_length]

        # 自我注意力
        x = self.self_attn(x, x, x, attn_mask=tgt_mask)[0]
        x = self.norm1(x)

        # 交叉注意力
        x = self.cross_attn(x, enc_output, enc_output, attn_mask=src_mask)[0]
        x = self.norm2(x)

        # 前馈网络
        x = self.feed_forward(x)

        return x

class ViTDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout):
        super(ViTDecoder, self).__init__()
        self.layers = nn.ModuleList([
            ViTDecoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        # x: 输入张量，形状为 [batch_size, seq_length, embed_dim]
        # enc_output: 编码器输出，形状为 [batch_size, seq_length, embed_dim]
        # src_mask: 编码器注意力掩码，形状为 [1, 1, seq_length, seq_length]
        # tgt_mask: 解码器注意力掩码，形状为 [1, 1, seq_length, seq_length]
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x

class ViTNeRF(nn.Module):
    def __init__(self, seq_len=6, embed_dim=256, num_heads=8, num_layers=6, dropout=0.1, out_ch=4):
        super(ViTNeRF, self).__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.embed_fn = nn.Embedding(seq_len, embed_dim)
        self.encoder = ViTEncoder(embed_dim, num_heads, num_layers, dropout)
        self.decoder = ViTDecoder(embed_dim, num_heads, num_layers, dropout)
        self.linear = nn.Linear(embed_dim, out_ch)
        self.src_mask = torch.ones(1, seq_len, seq_len)
        self.tgt_mask = torch.ones(1, seq_len, seq_len)

    def forward(self, x, ts=1):
        x = self.embed_fn(x)
        enc_output = self.encoder(x, self.src_mask)
        dec_input = x[:, -1, :]
        output = self.decoder(dec_input, enc_output, self.rc_mask, self.tgt_mask)
        output = F.relu(self.linear(output))
        return output, torch.zeros_like(x[:, :3])

class NeRF:
    @staticmethod
    def get_by_name(type,  *args, **kwargs):
        print ("NeRF type selected: %s" % type)

        if type == "original":
            model = NeRFOriginal(*args, **kwargs)
        elif type == "direct_temporal":
            model = DirectTemporalNeRF(*args, **kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model