import gc
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


#############Baseline PointNet#####################

class Tnet(nn.Module):
    """
        TNet module to predict a transformation matrix.
        It has Conv1d in “downsample part ” and fully connected layer in “upsample part”.
    """
    def __init__(self, k=3):
        super().__init__()
        self.k=k
        self.conv1 = nn.Conv1d(k,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.fc1 = nn.Linear(1024,512)
        self.fc2 = nn.Linear(512,256)
        self.fc3 = nn.Linear(256,k*k)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
       

    def forward(self, input):
        # input.shape == (bs,n,3)
        bs = input.size(0)
        #print('input : ',input.shape)
        x = F.relu(self.bn1(self.conv1(input)))
        #print('after conv1 : ',x.shape)
        x = F.relu(self.bn2(self.conv2(x)))
        #print('after conv2 : ',x.shape)
        x = F.relu(self.bn3(self.conv3(x)))
        #print('after conv3 : ',x.shape)
        pool = nn.MaxPool1d(x.size(-1))(x)
        #print('after pool : ',pool.shape)
        flat = nn.Flatten(1)(pool)
        #print('after flat : ',flat.shape)
        x = F.relu(self.bn4(self.fc1(flat)))
        #print('after fc1 : ',x.shape)
        x = F.relu(self.bn5(self.fc2(x)))
        #print('after fc2 : ',x.shape)

        
        #Here, I initialize an identity matrix by default because we want to start training with no transformations at all. 
        #So, we just add an identity matrix to the output
        init = torch.eye(self.k, requires_grad=True).repeat(bs,1,1)#initialize as identity
        if x.is_cuda:
            init=init.cuda()   
        matrix = self.fc3(x).view(-1,self.k,self.k) + init
        return matrix


class Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = Tnet(k=64)
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, input):
        matrix3x3 = self.input_transform(input) # here, input is of shape (bs,3,n)
        input = torch.transpose(input,1,2)# input is now (bs,n,3)
        # apply input transform (batch matrix multiplication) to ensure invariance to transformations
        # Here, I apply the 3x3 transformation matrix,matrix3x3 predicted by T-Net to coordinates of input points,input 
        x = torch.bmm(input, matrix3x3).transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # similarly for the feature space
        # we apply feature transform
        matrix64x64 = self.feature_transform(x)
        x = torch.transpose(x,1,2)
        x = torch.bmm(x, matrix64x64).transpose(1,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        output = nn.Flatten(1)(x)
        return output, matrix3x3, matrix64x64

class PointNet(nn.Module):
    '''PointNet architecture,wrapped the transforms module with the last MLP and LogSoftmax at the output'''
    def __init__(self, classes = 40):
        super().__init__()
        self.transform = Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)    
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x, matrix3x3, matrix64x64 = self.transform(input)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.logsoftmax(self.fc3(x))
        return output, matrix3x3, matrix64x64
    
#############U-Net#####################
class conbr_block(nn.Module):
    '''Conv1d block'''
    def __init__(self, in_layer, out_layer):
        super(conbr_block, self).__init__()

        self.conv1 = nn.Conv1d(in_layer, out_layer, 1)
        self.bn = nn.BatchNorm1d(out_layer)
        self.relu = nn.ReLU()
    
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        out = self.relu(x)
        
        return out       

class UNET_1D(nn.Module):
    '''1D Unet architecture'''
    def __init__(self ,input_dim):
        super(UNET_1D, self).__init__()
        self.input_dim = input_dim
        
        self.conv1 = conbr_block(input_dim,64) 
        self.conv2 = conbr_block(64,128) 
        self.conv3 = conbr_block(128,256) 
        self.conv4 = conbr_block(256,512) 
        self.conv5 = conbr_block(512,1024) 

        self.up6 = nn.ConvTranspose1d(1024,512,kernel_size=1)#nn.Upsample() 
        self.conv6 = conbr_block(1024,512) 
        self.up7 = nn.ConvTranspose1d(512,256,kernel_size=1)#nn.Upsample() 
        self.conv7 = conbr_block(512,256) 
        self.up8 = nn.ConvTranspose1d(256,128,kernel_size=1)#nn.Upsample() 
        self.conv8 = conbr_block(256,128) 
        self.up9 = nn.ConvTranspose1d(128,64,kernel_size=1)#nn.Upsample() 
        self.conv9 = conbr_block(128,64) 
        
        #input_dim = k 
        self.outcov = nn.Conv1d(64, input_dim*input_dim , kernel_size=1)

            
    def forward(self, x):               
        #############Encoder#####################
        
        conv1 = self.conv1(x)
        # print(conv1.shape)
        pool1 = F.max_pool1d(conv1,kernel_size=1) #nn.AvgPool1d(conv1)
        # print(pool1.shape)
        
        conv2 = self.conv2(pool1)
        # print(conv2.shape)
        pool2 = F.max_pool1d(conv2,kernel_size=1) #nn.AvgPool1d(conv2) 
        # print(pool2.shape)
        
        conv3 = self.conv3(pool2)
        # print(conv3.shape)
        pool3 = F.max_pool1d(conv3,kernel_size=1) #nn.AvgPool1d(conv3)  
        # print(conv3.shape)
        
        conv4 = self.conv4(pool3)
        # print(conv4.shape)
        pool4 = F.max_pool1d(conv4,kernel_size=1) #nn.AvgPool1d(conv4)  
        # print(conv4.shape)
        
        conv5 = self.conv5(pool4)
        # print(conv5.shape)
        
        #############Decoder####################
        
        up6 = self.up6(conv5)
        up6 = torch.cat([up6, conv4], 1)
        conv6 = self.conv6(up6)
        # print(conv6.shape)
        
        del up6
        del conv4
        del conv5
        
        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv3], 1)
        conv7 = self.conv7(up7)
        # print(conv7.shape)
        
        del up7
        del conv6
        del conv3
        
        up8 = self.up8(conv7)
        up8 = torch.cat([up8, conv2], 1)
        conv8 = self.conv8(up8)
        # print(conv8.shape)
        
        del up8
        del conv7
        del conv2
        
        up9 = self.up9(conv8)
        up9 = torch.cat([up9, conv1], 1)
        conv9 = self.conv9(up9)   
        # print(conv9.shape)
        
        del up9
        del conv8
        del conv1
        
        out = self.outcov(conv9)
        
        del conv9
        gc.collect()
        torch.cuda.empty_cache()
        
        pool = nn.MaxPool1d(out.size(-1))(out)
        flat = nn.Flatten(1)(pool)
        #initialize as identity
        init = torch.eye(self.input_dim, requires_grad=True).repeat(bs,1,1)
        if x.is_cuda:
            init=init.cuda()
        matrix = flat.view(-1,self.input_dim,self.input_dim) + init
        return matrix    

#Change both input transform and feature transform T-Net 
#only keep last network before classification as FC layers mlp
#U-Net has skip connection (By introducing skip connections in the encoder-decoded architecture, fine-grained details can be recovered in the prediction)
class UNet_Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = UNET_1D(3) 
        self.feature_transform = UNET_1D(64) 
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, input):
        matrix3x3 = self.input_transform(input) # here, input is of shape (bs,3,n)
        input = torch.transpose(input,1,2)# input is now (bs,n,3)
        # apply input transform (batch matrix multiplication) 
        x = torch.bmm(input, matrix3x3).transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # similarly for the feature space
        # we apply input transform
        matrix64x64 = self.feature_transform(x)
        x = torch.transpose(x,1,2)
        x = torch.bmm(x, matrix64x64).transpose(1,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        output = nn.Flatten(1)(x)
        return output, matrix3x3, matrix64x64

class UNet_PointNet(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.transform = UNet_Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)    
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x, matrix3x3, matrix64x64 = self.transform(input)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.logsoftmax(self.fc3(x))
        return output, matrix3x3, matrix64x64

#Change only input transform T-Net to UNet
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class UNet_input_Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = UNET_1D(3) 
        self.feature_transform = Tnet(k=64) 
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, input):
        matrix3x3 = self.input_transform(input) # here, input is of shape (bs,3,n)
        input = torch.transpose(input,1,2)# input is now (bs,n,3)
        # apply input transform (batch matrix multiplication) 
        x = torch.bmm(input, matrix3x3).transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # similarly for the feature space
        # we apply input transform
        matrix64x64 = self.feature_transform(x)
        x = torch.transpose(x,1,2)
        x = torch.bmm(x, matrix64x64).transpose(1,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        output = nn.Flatten(1)(x)
        return output, matrix3x3, matrix64x64

class UNet_input_PointNet(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.transform = UNet_input_Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)    
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x, matrix3x3, matrix64x64 = self.transform(input)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.logsoftmax(self.fc3(x))
        return output, matrix3x3, matrix64x64

#Change only input transform T-Net to UNet
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class UNet_feature_Transform(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Tnet(k=3)
        self.feature_transform = UNET_1D(64) 
        self.conv1 = nn.Conv1d(3,64,1)
        self.conv2 = nn.Conv1d(64,128,1)
        self.conv3 = nn.Conv1d(128,1024,1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
       
    def forward(self, input):
        matrix3x3 = self.input_transform(input) # here, input is of shape (bs,3,n)
        input = torch.transpose(input,1,2)# input is now (bs,n,3)
        # apply input transform (batch matrix multiplication) 
        x = torch.bmm(input, matrix3x3).transpose(1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        
        # similarly for the feature space
        # we apply input transform
        matrix64x64 = self.feature_transform(x)
        x = torch.transpose(x,1,2)
        x = torch.bmm(x, matrix64x64).transpose(1,2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = nn.MaxPool1d(x.size(-1))(x)
        output = nn.Flatten(1)(x)
        return output, matrix3x3, matrix64x64

class UNet_feature_PointNet(nn.Module):
    def __init__(self, classes = 40):
        super().__init__()
        self.transform = UNet_feature_Transform()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, classes)    
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        x, matrix3x3, matrix64x64 = self.transform(input)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        output = self.logsoftmax(self.fc3(x))
        return output, matrix3x3, matrix64x64