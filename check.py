from PyQt5 import QtGui,QtCore, QtWidgets
from gui.ui_model import *
import numpy as np
import sys
from options.test_options import TestOptions
from gui.ui_model import ui_model
from PyQt5 import QtWidgets
import os
import json
import time

class Options:
    def __init__(self, 
                 name='fake/real_classifier', 
                 model='tdanet', 
                 mask_type=[1, 2, 3], 
                 checkpoints_dir='./checkpoints', 
                 which_iter='latest',
                 gpu_ids=[],
                 text_config = 'config.bird.yml',
                 output_scale=4,
                 img_file='/data/dataset/train', 
                 mask_file='none',
                 loadSize=[266,266],
                 fineSize=[256, 256],
                 resize_or_crop='resize_and_crop',
                 no_flip=False,
                 no_rotation=False,
                 no_augment=False,                 
                 batchSize=10, 
                 nThreads=8,
                 no_shuffle=False,
                 display_winsize=256,
                 display_id=1,
                 display_port=8097,
                 display_single_pane_ncols = 0,
                 prior_alpha=0.8, 
                 prior_beta=8,
                 no_maxpooling=False,
                 update_language=False,
                 detach_embedding=False,
                 train_paths='two', 
                 dynamic_sigma=False, 
                 lambda_rec_l1=20.0, 
                 lambda_gen_l1=20.0, 
                 lambda_kl=20.0, 
                 lambda_gan=1.0,
                 lambda_match=0.1,
                 iter_count=1, 
                 niter=100,
                 niter_decay=0,
                 continue_train=False,
                 valid_file='/data/dataset/valid',
                 lr_policy='lambda', 
                 lr=1e-4, 
                 gan_mode='lsgan',
                 display_freq=100,
                 print_freq=100,
                 save_latest_freq=1000,
                 save_iters_freq=10000,
                 no_html=False,
                 
                 results_dir='./results/',
                 phase='test',
                 nsampling=50,
                 ncaptions=10,
                 save_number=10,
                 no_variance=False,
                ):
        
        self.name=name
        self.model=model
        self.mask_type=mask_type
        self.img_file=img_file # has paths of all the images
        self.mask_file=mask_file
        self.checkpoints_dir = checkpoints_dir
        self.which_iter = which_iter
        self.gpu_ids = gpu_ids
        self.text_config = text_config
        self.output_scale=output_scale
        self.batchSize=batchSize
        self.loadSize = loadSize
        self.resize_or_crop=resize_or_crop
        self.no_flip=no_flip
        self.no_rotation = no_rotation
        self.no_augment = no_augment
        self.nThreads = nThreads,
        self.no_shuffle=no_shuffle
        self.display_winsize = display_winsize
        self.display_id = display_id
        self.display_port = display_port
        self.fineSize=fineSize
        self.display_single_pane_ncols = display_single_pane_ncols
        self.no_maxpooling = no_maxpooling
        self.update_language = update_language
        self.detach_embedding = detach_embedding
        self.prior_alpha=prior_alpha
        self.prior_beta=prior_beta
        self.gan_mode=gan_mode
        self.no_variance=no_variance
        self.nsampling=nsampling
        self.train_paths=train_paths
        self.dynamic_sigma=dynamic_sigma
        self.lambda_rec_l1=lambda_rec_l1
        self.lambda_gen_l1=lambda_gen_l1
        self.lambda_kl=lambda_kl
        self.lambda_gan=lambda_gan
        self.lambda_match=lambda_match
        self.iter_count = iter_count
        self.niter = niter
        self.niter_decay = niter_decay
        self.continue_train = continue_train
        self.valid_file = valid_file
        self.lr_policy = lr_policy
        self.lr = lr
        self.display_freq = display_freq
        self.print_freq = print_freq
        self.save_latest_freq = save_latest_freq
        self.save_iters_freq = save_iters_freq
        self.no_html = no_html
        
        self.results_dir=results_dir
        self.phase=phase
        self.nsampling=nsampling
        self.ncaptions=ncaptions
        self.save_number=save_number
        self.no_variance=no_variance
        
        self.isTrain = True #MAKE IT FALSE WHEN TESTING!!!

''' do not remove the below comment '''

# file_path="check.json"

# with open(file_path,'r') as file:
#     data=json.load(file)

# image_dir="/home/hemanthgaddey/Documents/tdanet_/tdanet/datasets/CUB_200_2011/images"
# paths=[] #complete path of images
# names=[] #captions for respective image
# with open('img_paths.txt','r') as f:
#     classes=os.listdir(image_dir)
#     x=[np.random.randint(0,200) for _ in range(10)]
#     classes=[classes[i] for i in x]
#     for i in classes:
#         path=image_dir+'/'+i
#         files=os.listdir(path)[:10] # the file names
#         for j in files:
#             paths.append(path+'/'+j)
#             names.append(data[j])
# print(len(paths))
# print(len(names))
# with open('img_paths.txt','w') as img_path_file:
#     for i in paths:
#         img_path_file.write(i+'\n')
        
# with open('captions.txt','w') as caption_file:
#     for i in names:
#         caption_file.write(i[0]+'\n')


with open('img_paths.txt','r') as f:
    paths=f.readlines()
with open('captions.txt','r') as f:
    captions=f.readlines()
opt=Options(name='tda_bird',model="tdanet",mask_type=[0,1,2,3],img_file='./datasets/CUB_200_2011/train.flist',mask_file='./datasets/CUB_200_2011/train_mask.flist',text_config='config.bird.yml') #TRAINING OPTIONS, CHANGE FOR TESTING!!!
app = QtWidgets.QApplication(sys.argv)
opt = TestOptions().parse()
model = ui_model(opt)
for i in range(len(paths)):
    model.show()
    model.showImage(fname=paths[i][:-1])
    model.load_model()
    print("filling the mask")
    model.fill_mask(text=captions[i][:-1],name=paths[0][:-1])
    # time.sleep(100)
app.exec_()
app.quit()
    #sys.exit(app.exec_())