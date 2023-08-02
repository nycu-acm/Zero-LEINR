import torch
import torch.nn as nn
import torch.nn.functional as F
import math
#import pytorch_colors as colors
import numpy as np

from torchinfo import summary

class enhance_net_nopool(nn.Module):

	def __init__(self):
		super(enhance_net_nopool, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
		self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x, mode=None):

		x1 = self.relu(self.e_conv1(x))
		# p1 = self.maxpool(x1)
		assert not torch.isnan(x1).any(), 'x1 NAN'
		x2 = self.relu(self.e_conv2(x1))
		# p2 = self.maxpool(x2)
		assert not torch.isnan(x2).any(), 'x2 NAN'
		x3 = self.relu(self.e_conv3(x2))
		# p3 = self.maxpool(x3)
		assert not torch.isnan(x3).any(), 'x3 NAN'
		x4 = self.relu(self.e_conv4(x3))
		assert not torch.isnan(x4).any(), 'x4 NAN'

		x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
		assert not torch.isnan(x5).any(), 'x5 NAN'
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))
		assert not torch.isnan(x6).any(), 'x6 NAN'

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		assert not torch.isnan(x_r).any(), 'x7 NAN'
		r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)


		x = x + r1*(torch.pow(x,2)-x)
		x = x + r2*(torch.pow(x,2)-x)
		x = x + r3*(torch.pow(x,2)-x)
		enhance_image_1 = x + r4*(torch.pow(x,2)-x)		
		x = enhance_image_1 + r5*(torch.pow(enhance_image_1,2)-enhance_image_1)		
		x = x + r6*(torch.pow(x,2)-x)	
		x = x + r7*(torch.pow(x,2)-x)
		enhance_image = x + r8*(torch.pow(x,2)-x)
		r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
		
		return enhance_image_1,enhance_image,r

class enhance_net_nopool_plus(nn.Module):

	def __init__(self):
		super(enhance_net_nopool_plus, self).__init__()

		self.relu = nn.ReLU(inplace=True)

		number_f = 32
		self.e_conv1 = nn.Conv2d(3,3,3,1,1,groups=3,bias=True) 
		self.e_conv1_1x1 = nn.Conv2d(3,number_f,1,bias=True) 
		self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1, groups = number_f,bias=True)
		self.e_conv2_1x1 = nn.Conv2d(number_f, number_f,1, bias = True) 
		self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1, groups = number_f,bias=True)
		self.e_conv3_1x1 = nn.Conv2d(number_f, number_f,1, bias = True) 
		self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1, groups = number_f,bias=True)
		self.e_conv4_1x1 = nn.Conv2d(number_f, number_f,1, bias = True) 
		self.e_conv5 = nn.Conv2d(number_f*2,number_f*2,3,1,1, groups = number_f*2,bias=True)
		self.e_conv5_1x1 = nn.Conv2d(number_f*2, number_f,1, bias = True) 
		self.e_conv6 = nn.Conv2d(number_f*2,number_f*2,3,1,1, groups = number_f*2,bias=True)
		self.e_conv6_1x1 = nn.Conv2d(number_f*2, number_f,1, bias = True) 
		self.e_conv7 = nn.Conv2d(number_f*2,3,3,1,1,bias=True) 

		self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
		self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)


		
	def forward(self, x , mode, stop_mechanism = False):

		x1 = self.relu(self.e_conv1_1x1(self.e_conv1(x)))
		# p1 = self.maxpool(x1)
		x2 = self.relu(self.e_conv2_1x1(self.e_conv2(x1)))
		# p2 = self.maxpool(x2)
		x3 = self.relu(self.e_conv3_1x1(self.e_conv3(x2)))
		# p3 = self.maxpool(x3)
		x4 = self.relu(self.e_conv4_1x1(self.e_conv4(x3)))

		x5 = self.relu(self.e_conv5_1x1(self.e_conv5(torch.cat([x3,x4],1))))
		# x5 = self.upsample(x5)
		x6 = self.relu(self.e_conv6_1x1(self.e_conv6(torch.cat([x2,x5],1))))

		x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
		

# 		r = x_r
		if mode =="train":
			x = x + x_r*(torch.pow(x,2)-x)
			x = x + x_r*(torch.pow(x,2)-x)
			x = x + x_r*(torch.pow(x,2)-x)
			enhance_image_1 = x + x_r*(torch.pow(x,2)-x)		
			x = enhance_image_1 + x_r*(torch.pow(enhance_image_1,2)-enhance_image_1)		
			x = x + x_r*(torch.pow(x,2)-x)	
			x = x + x_r*(torch.pow(x,2)-x)
			x= x + x_r*(torch.pow(x,2)-x)
			#r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
			
		elif mode =="val":
			for idx in range(8):
				x = x + x_r*(torch.pow(x,2)-x)
				if stop_mechanism:
					expossure = torch.mean(x)
					if expossure > 0.6:
						return x,x,x_r
		
		return x,x,x_r

class fusion_block(nn.Module):
    def __init__(self):
        super(fusion_block , self).__init__()
        self.conv = nn.Conv2d(6 , 1 , 1,bias=True)
        self.act  = nn.Sigmoid()
    def forward(self, x1, x2):
        max_ = torch.max(x1 , x2)
        avg_ = (x1 + x2) / 2
        out_ = torch.cat( [ max_ , avg_] , dim = 1)
        out_ = self.act(self.conv(out_))
        return out_
    
class fusion_block_2(nn.Module):
    def __init__(self):
        super(fusion_block_2, self).__init__()
        self.in_channels = 6
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels * 2,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels * 2,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
            )
        self.conv_layer_2_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=3,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
            )
        self.conv_layer_2_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=3,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
            )
        
    def forward(self, x1, x2):
        input_ = torch.cat([x1, x2], dim=1)
        weight = self.conv_layer(input_)
        return self.conv_layer_2_1(weight), self.conv_layer_2_2(weight)

class fusion_block_3(nn.Module):
    def __init__(self):
        super(fusion_block_3, self).__init__()
        self.in_channels = 6
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels * 2,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=self.in_channels, out_channels=3,
                      kernel_size=3, stride=1, padding=1, padding_mode='reflect'),
            nn.LeakyReLU(0.2)
            )
    def forward(self, x1, x2):
        input_ = torch.cat([x1, x2], dim=1)
        out = self.conv_layer(input_)
        return out

if __name__ == "__main__":
    model = enhance_net_nopool().cuda()
    summary(model, (1, 3, 256, 256))
