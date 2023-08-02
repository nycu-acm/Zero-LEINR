import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision.models.vgg import vgg16
import numpy as np
from skimage.segmentation import slic
import cv2



class L_color(nn.Module):

    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape

        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)


        return k

			
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E
class L_exp(nn.Module):

    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        # print(1)
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):

        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)

        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val] ).cuda(),2))
        return d
        
class L_TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(L_TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
        # print(1)
    def forward(self, x ):
        # self.grad = np.ones(x.shape,dtype=np.float32)
        b,c,h,w = x.shape
        # x_de = x.cpu().detach().numpy()
        r,g,b = torch.split(x , 1, dim=1)
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k =torch.pow( torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        # print(k)
        

        k = torch.mean(k)
        return k

class perception_loss(nn.Module):
    def __init__(self):
        super(perception_loss, self).__init__()
        features = vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential() 
        self.to_relu_2_2 = nn.Sequential() 
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        # out = (h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3)
        return h_relu_4_3


class L_color_consistency(nn.Module):
    def __init__(self):
        super(L_color_consistency, self).__init__()
        self.mse = torch.nn.MSELoss(reduction='mean')
    """    # ITU-R BT.601 conversion
    def forward(self, x1rgb, x2rgb):
        def rgb2yCbCr(rgb): # (b, c=rgb, h, w)
            output = torch.zeros(x1rgb.shape)
            output[:, 0, :, :] = rgb[:, 0, :, :] * 65.481 + rgb[:, 1, :, :] * 128.553 + rgb[:, 2, :, :] * 24.966 + 16
            output[:, 1, :, :] = rgb[:, 0, :, :] * (-37.797) + rgb[:, 1, :, :] * (-74.203) + rgb[:, 2, :, :] * 112.0 + 128
            output[:, 2, :, :] = rgb[:, 0, :, :] * 112.0 + rgb[:, 1, :, :] * (-93.786) + rgb[:, 2, :, :] * (-18.214) + 128
            return output # (b, c=yCbCr, h, w)
    """
    # ITU-R BT.709 conversion
    def forward(self, x1rgb, x2rgb):
        def rgb2yCbCr(rgb): # (b, c=rgb, h, w)
            output = torch.zeros(x1rgb.shape)
            output[:, 0, :, :] = rgb[:, 0, :, :] * 0.2126 + rgb[:, 1, :, :] * 0.7152 + rgb[:, 2, :, :] * 0.0722
            output[:, 1, :, :] = rgb[:, 0, :, :] * (-0.1146) + rgb[:, 1, :, :] * (-0.3854) + rgb[:, 2, :, :] * 0.5
            output[:, 2, :, :] = rgb[:, 0, :, :] * 0.5 + rgb[:, 1, :, :] * (-0.4542) + rgb[:, 2, :, :] * (-0.0458)
            return output # (b, c=yCbCr, h, w)
        
        x1yCbCr = rgb2yCbCr(x1rgb)
        x2yCbCr = rgb2yCbCr(x2rgb)

        Cb_mse = self.mse(x1yCbCr[:, 1, :, :], x2yCbCr[:, 1, :, :]) # (b, h, w)
        Cr_mse = self.mse(x1yCbCr[:, 2, :, :], x2yCbCr[:, 2, :, :]) # (b, h, w)
        return Cb_mse + Cr_mse


class L_color_consistency_ratio(nn.Module):
    def __init__(self):
        super(L_color_consistency_ratio, self).__init__()
        self.abs = nn.L1Loss(reduction='mean')
    """    # ITU-R BT.601 conversion
    def forward(self, x1rgb, x2rgb):
        def rgb2yCbCr(rgb): # (b, c=rgb, h, w)
            output = torch.zeros(x1rgb.shape)
            output[:, 0, :, :] = rgb[:, 0, :, :] * 65.481 + rgb[:, 1, :, :] * 128.553 + rgb[:, 2, :, :] * 24.966 + 16
            output[:, 1, :, :] = rgb[:, 0, :, :] * (-37.797) + rgb[:, 1, :, :] * (-74.203) + rgb[:, 2, :, :] * 112.0 + 128
            output[:, 2, :, :] = rgb[:, 0, :, :] * 112.0 + rgb[:, 1, :, :] * (-93.786) + rgb[:, 2, :, :] * (-18.214) + 128
            return output # (b, c=yCbCr, h, w)
    """
    # ITU-R BT.709 conversion
    def forward(self, x1rgb, x2rgb):
        def rgb2yCbCr(rgb): # (b, c=rgb, h, w)
            output = torch.zeros(x1rgb.shape)
            output[:, 0, :, :] = rgb[:, 0, :, :] * 0.2126 + rgb[:, 1, :, :] * 0.7152 + rgb[:, 2, :, :] * 0.0722
            output[:, 1, :, :] = rgb[:, 0, :, :] * (-0.1146) + rgb[:, 1, :, :] * (-0.3854) + rgb[:, 2, :, :] * 0.5
            output[:, 2, :, :] = rgb[:, 0, :, :] * 0.5 + rgb[:, 1, :, :] * (-0.4542) + rgb[:, 2, :, :] * (-0.0458)
            return output # (b, c=yCbCr, h, w)
        
        x1yCbCr = rgb2yCbCr(x1rgb)
        x2yCbCr = rgb2yCbCr(x2rgb)
        # print(x1yCbCr[:, 1, :, :].size())
        # print(torch.sum(x1yCbCr[:, 1, :, :], [1, 2]).size())
        CbCr_ratio_1 = torch.sum(x1yCbCr[:, 1, :, :], [1, 2]) / torch.sum(x1yCbCr[:, 2, :, :], [1, 2])
        CbCr_ratio_2 = torch.sum(x2yCbCr[:, 1, :, :], [1, 2]) / torch.sum(x2yCbCr[:, 2, :, :], [1, 2])
        return self.abs(CbCr_ratio_1, CbCr_ratio_2)

class L_color_weighted(nn.Module):
    # "A Method to Improve Robustness of the Gray World Algorithm"
    # https://download.atlantis-press.com/article/25839570.pdf
    # This method aims to downweight the contribution of pixels 
    # that is in a large segmentation region, avoiding the 
    # violation of gray world assumption.

    def __init__(self):
        super(L_color_weighted, self).__init__()

    def forward(self, x ):

        b,c,h,w = x.shape # (b, c, h, w)
        scikit_x = x.permute(0, 2, 3, 1) # (b, h, w, c)
        scikit_x = scikit_x.to(torch.double).cpu().detach().numpy()
        
        segments = np.zeros((b, h, w))
        
        for bb in range(b):

            segment = slic(scikit_x[bb], n_segments=50) # (b, h, w)

            val, cnt = np.unique(segment, return_counts=True) 
            cnt = 1. / cnt
            segment = cnt[segment]
            segments[bb] = segment
        
        segments = torch.from_numpy(segments).unsqueeze(3) # (b, h, w, 1)
        segments = segments.repeat(1, 1, 1, 3) # (b, h, w, c)
        segments = segments.permute(0, 3, 1, 2) # (b, c, h, w) 

        x = torch.mul(segments.cuda(), x)
        
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k


class L_color_weighted_seg_cv2(nn.Module):
    # "A Method to Improve Robustness of the Gray World Algorithm"
    # https://download.atlantis-press.com/article/25839570.pdf
    # This method aims to downweight the contribution of pixels 
    # that is in a large segmentation region, avoiding the 
    # violation of gray world assumption.

    def __init__(self):
        super(L_color_weighted_seg_cv2, self).__init__()
        
        self.K = 5
        self.attempts = 1
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 
                         10, 1.0)

    def forward(self, x):

        b, c, h, w = x.shape # (b, c, h, w)
        x_cv = x.permute(0, 2, 3, 1) # (b, h, w, c)
        x_cv = x_cv.to(torch.double).cpu().detach().numpy()
        
        segments = np.zeros((b, h, w))
        
        for bb in range(b):
            
            rgb_image = x_cv[bb] # (h, w, c)
            vectorized = np.float32(rgb_image.reshape((-1, 3))) # (h*w, c)
            _, label, _ = cv2.kmeans(vectorized, 
                                     self.K, 
                                     None, 
                                     self.criteria, 
                                     self.attempts, 
                                     cv2.KMEANS_PP_CENTERS)
            segment = label.reshape((h, w))
            val, cnt = np.unique(segment, return_counts=True) 
            cnt = cnt / (h * w)
            cnt = 1. - cnt
            segment = cnt[segment]
            segments[bb] = segment / np.sum(segment)
        
        segments = torch.from_numpy(segments).unsqueeze(3) # (b, h, w, 1)
        segments = segments.repeat(1, 1, 1, 3) # (b, h, w, c)
        segments = segments.permute(0, 3, 1, 2) # (b, c, h, w) 

        x = torch.mul(segments.cuda(), x)
        
        mean_rgb = torch.sum(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        # print(mr, mg, mb)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)

        return k

class L_color_angle(nn.Module):
    """
    A loss to make sure that the color would not change after enhanced
    """
    def __init__(self):
        super(L_color_angle, self).__init__()
        self.clip_value = 0.999999
    
    def forward(self, org, enhance):
        norm_org = torch.nn.functional.normalize(org)
        norm_enhance = torch.nn.functional.normalize(enhance)
        dot = torch.sum(norm_org * norm_enhance, dim=1)
        dot = torch.clamp(dot, min=-self.clip_value, max=self.clip_value)
        angle = torch.acos(dot) * (180 / math.pi)
        
        return torch.mean(angle)
        