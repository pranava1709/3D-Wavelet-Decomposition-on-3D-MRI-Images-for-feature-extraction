import torch 
import torch.nn as nn
import torch.nn.functional as nn1
import torch 
import torch.nn as nn
import torch.nn.functional as nn1
import ast 
import numpy as np
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt
import os
os.environ['CUDA_LAUNCH_BLOCKING'] ='1'

#os.environ['KMP_DUPLICATE_LIB_OK']='True'
from timeit import default_timer
from utilities3 import *
# from pytorch_wavelets import DWT, IDWT # (or import DWT, IDWT)
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
import nibabel
import tensorflow as tf
from google.colab.patches import cv2_imshow

lst = []
torch.manual_seed(0)
np.random.seed(0)

images = "/content/drive/MyDrive/New folder/UCSF-PDGM-0090_nifti/UCSF-PDGM-0090_ADC.nii.gz"
image = nibabel.load(images)
print(image.shape)
image_af = image.affine
print(image_af)
print(image_af.shape)
#
#image_T =torch.unsqueeze(image_T,dim=0)
#image_T = torch.unsqueeze(image_T,dim=1)

#print(image_T.shape)
image_T  = image.get_fdata()
#image_T = cv2.cvtColor(image_T,cv2.COLOR_BGR2GRAY)
image_T = torch.Tensor(image_T)
#print(image_T.shape)
plt.imsave('imn.png',image_T[:,:,0])
#image_af = image.affine
#print(image_af)
#print(image_af.shape)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
def WNO3d(in_channels, out_channels, image):


    with torch.no_grad():

        #image_T = torch.Tensor(image)
        #print(image_T.shape)
        #cv2.imwrite('im3.png',image_T[0,:,:])
    
        dwt3d = DWT_3D(wavename = 'db4')
        transform= dwt3d(image.to(device))

        lst.append(transform)
        lst_ar = np.array(lst)
        print(lst_ar[0][0].shape)

     
     

        #print(trr.shape)
        a1,a2,a3,a4,a5,a6,a7,a8 =lst_ar[0]
        #tr = np.concatenate((a1,a2,a3,a4,a5,a6,a7,a8),axis = 2)
        #print(tr.shape)
        print(a1.shape)
        xfeat = [a2,a4,a6,a8]  

        xfeat = np.array(xfeat)
        print(a1.shape)
        print(a2.shape)
        print(a3.shape)
        xfeat = np.concatenate((a2,a4,a6,a8),axis =0 )
        #xfeat = np.concatenate(xfeat)
        print(xfeat.shape)
        xcoeff = [a1,a3,a5,a7]
        xcoeff = np.concatenate((a1,a3,a5,a7),axis =0)
        #xcoeff = np.concatenate(xcoeff)
        print(xcoeff.shape)
        '''
        modes1 = tr.shape[-1]
        print(modes1)
        modes2 = tr.shape[-2]
        print(modes2)
        modes3 = tr.shape[-3]
        #modes4 = mat1.shape[3]

        scale = (1/(in_channels*out_channels))
        weights1 = nn.Parameter(
                scale *torch.randn(in_channels, out_channels, modes1, modes2, modes3))
        weights2 = nn.Parameter(
            scale *torch.randn(in_channels, out_channels, modes1, modes2, modes3))
        weights3 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))
        weights4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))

        weights5 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))
        weights6 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))
      
        weights1 = np.array(weights1)

        print(weights1.shape)
        '''
        
        

        modes1 = 480
        print(modes1)
        modes2 = 120
        print(modes2)
        modes3 = 78
        modes4 = 77
       
        scale = (1/(in_channels*out_channels))
        weights1 = nn.Parameter(
                scale *torch.randn(in_channels, out_channels, modes1, modes2, modes3))
        weights2 = nn.Parameter(
            scale *torch.randn(in_channels, out_channels, modes1, modes2, modes4))
        weights3 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))
        weights4 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))

        weights5 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))
        weights6 = nn.Parameter(
                scale * torch.randn(in_channels, out_channels, modes1, modes2, modes3))
      
        #weights1 = np.array(weights1)

        print(weights1.shape)
        

   
        xfeat = np.expand_dims(xfeat,axis =0)
        xfeat = np.expand_dims(xfeat,axis = 1)

        print(xfeat.shape)
        #xfeat = np.resize
        #xfeat = np.resize(xfeat,[xfeat.shape[0],xfeat.shape[0],xfeat.shape[0],xfeat.shape[2],xfeat.shape[3]])
        #weights = np.resize(weights1,[1,1,480,120,78])

        matmu = np.einsum("bixyz,ioxyz->boxyz", xfeat, weights1,casting = 'safe')
        print(matmu.shape)
      
        jja  = xcoeff
        print(jja.shape)

   
        jja = np.expand_dims(jja,axis = 0)
        jja = np.expand_dims(jja,axis = 1)
       
        print(jja.shape)
        print(weights2.shape)
       
        #jjaa = jja[:,:,:,:,:]
        #jja  =np.resize(jja,[jja.shape[0],jja.shape[1],weights2.shape[2],jja.shape[3],jja.shape[2]])
        #print(jja.shape)
    
        


        jja =  np.einsum("bixyz,ioxyz->boxyz",jja, weights2,casting = 'safe')
        for hh in jja:
            print(hh[0][0].shape)


        #jja =  np.einsum("bixyz,ioxyz->boxyz",jja, weights3)
        #jja =  np.einsum("bixyz,ioxyz->boxyz",jja, weights4)
        #jja =  np.einsum("bixyz,ioxyz->boxyz",jja, weights5)
        #jja =  np.einsum("bixyz,ioxyz->boxyz",jja, weights6)

        invdwt = IDWT_3D(wavename = 'db4')
        print(invdwt)
        print(matmu.shape)
        jjaaaa1 = np.array_split(jja[0][0],4,axis = 0)
        aaa,bbb,ccc,ddd = jjaaaa1
        #aaa= np.resize(aaa,[120,120,78])
        #bbb= np.resize(bbb,[120,120,78])
        #ccc= np.resize(ccc,[120,120,78])
        #ddd= np.resize(ddd,[120,120,78])
        
        
        mamu1 =np.array_split(matmu[0][0], 4,axis = 0)
        aa,bb,cc,dd= mamu1
        aa = torch.from_numpy(aa)
        bb = torch.from_numpy(bb)
        cc = torch.from_numpy(cc)
        dd = torch.from_numpy(dd)
        aaa = torch.from_numpy(aaa)
        bbb = torch.from_numpy(bbb)
        ccc = torch.from_numpy(ccc)
        ddd = torch.from_numpy(ddd)
        a1 = torch.unsqueeze(aa,axis = 0)
        a1 = torch.unsqueeze(a1,axis = 0)
        b1 = torch.unsqueeze(bb,axis = 0)
        b1 = torch.unsqueeze(b1,axis = 0)
        c1 = torch.unsqueeze(cc,axis = 0)
        c1 = torch.unsqueeze(c1,axis = 0)
        d1 = torch.unsqueeze(dd,axis = 0)
        d1 = torch.unsqueeze(d1,axis = 0)
        
        aaa = torch.unsqueeze(aaa,axis = 0)
        aaa = torch.unsqueeze(aaa,axis = 0)
        bbb= torch.unsqueeze(bbb,axis = 0)
        bbb = torch.unsqueeze(bbb,axis = 0)
        ccc= torch.unsqueeze(ccc,axis = 0)
        ccc = torch.unsqueeze(ccc,axis = 0)
        ddd= torch.unsqueeze(ddd,axis = 0)
        ddd = torch.unsqueeze(ddd,axis = 0)
        




      
        print(a1.shape)
        print(b1.shape)
        print(c1.shape)
        print(d1.shape)
        print(aaa.shape)
        print(bbb.shape)
        print(ccc.shape)
        print(ddd.shape)
        
        
        #a1 = torch.reshape(a1,[1,1,120,120,155])
        #a1 = a1.permute(0,1,3,4,2)
        #b1 = b1.permute(0,1,3,4,2)
        #c1 = c1.permute(0,1,3,4,2)
        #d1 = d1.permute(0,1,3,4,2)
        #aaa = aaa.permute(0,1,3,4,2)
        #bbb = bbb.permute(0,1,3,4,2)
        #ccc = ccc.permute(0,1,3,4,2)
        #ddd = ddd.permute(0,1,3,4,2)
        
        




        
        idwtt = invdwt(aaa,a1,bbb,b1,ccc,c1,ddd,d1)
        print(idwtt.shape)
        return idwtt
ss =WNO3d(1,1,image_T)

ss= ss.detach().numpy()
print(ss.shape)
ss1 = ss[0,0,:,:,0]
print(ss1.shape)
#plt.plot(ss1)
plt.savefig('sdbhes.png')
cv2.imwrite('imm.png',ss1)

ss = ss[0,0,:,:,:]

#with open('new.nii.gz','a') as ll:
#  ll.write(str(ss))
#  ll.write('\n')
#ss = ss.dtype('int16')
#ss1=  ss.detach().numpy()

aa1 = nibabel.Nifti1Image(ss,affine = image.affine)

nibabel.save(aa1,'new1.nii.gz')




