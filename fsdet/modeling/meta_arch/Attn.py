import ipdb
import torch
import torch.nn as nn
import torchvision
import ipdb

class ALL_Attention(nn.Module):
    def __init__(self, channel):
        super(ALL_Attention, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()

        # ipdb.set_trace()
        # [N, C/2, H * W]
        # print(x.shape)
        x_phi = self.conv_phi(x).view(b, c, -1).cuda()   #经过1*1卷积降低通道数   Q   torch.Size([1, 16, 1458])
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous().cuda()   #K
        # x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()       #V
        x_g = self.conv_g(x).view(b, c, -1).cuda().cuda()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_phi, x_theta).cuda()   #Q*K
        self.mk = nn.Linear(c, x_phi.shape[2], bias=False).cuda()
        #####此处插入
        mul_theta_phi = self.mk(mul_theta_phi).cuda()
        mul_theta_phi = self.softmax(mul_theta_phi).cuda()

        # print(mul_theta_phi[0,:,0])
        # [N, H * W, C/2]
        #此处插入
        # ipdb.set_trace()
        self.mv = nn.Linear(x_g.shape[2], c, bias=False).cuda()
        x_g = self.mv(x_g)
        # ipdb.set_trace()
        mul_theta_phi_g = torch.matmul(mul_theta_phi.permute(0, 2, 1).contiguous(), x_g).cuda()

        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b, c, h, w).cuda()
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g).cuda()
        out = mask + x
        return out

class All_att(nn.Module):
    def __init__(self, channel):
        super(All_att, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1,padding=0, bias=False)
        self.conv_theta = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_g = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=1)
        self.linear_0 = nn.Conv1d(channel, channel, 1, bias=False)
    def forward(self, x):
        # [N, C, H , W]
        b, c, h, w = x.size()
        # [N, C/2, H * W]
        x_phi = self.conv_phi(x).view(b, c, -1).cuda()   #经过1*1卷积降低通道数   Q
        # [N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous().cuda()   #K
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous().cuda()       #V
        self.mv = nn.Linear(x_g.shape[2], x_g.shape[2], bias=False).cuda()
        x_g = self.mv(x_g).cuda()
        # [N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi).cuda()   #Q*K
        self.mk = nn.Linear(mul_theta_phi.shape[2], mul_theta_phi.shape[2], bias=False).cuda()
        mul_theta_phi = self.mk(mul_theta_phi).cuda()
        mul_theta_phi = self.softmax(mul_theta_phi).cuda()
        # print(mul_theta_phi[0,:,0])
        # [N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g).cuda()
        # ipdb.set_trace()
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0,2,1).contiguous().view(b,c, h, w).cuda()
        # [N, C, H , W]
        mask = self.conv_mask(mul_theta_phi_g).cuda()
        out = mask + x
        return out

if __name__=='__main__':
    model = ALL_Attention(channel=256)
    print(model)

    input = torch.randn(16, 256, 12, 21)
    out = model(input)
    print(out.shape)