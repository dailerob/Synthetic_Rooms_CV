# -*- coding: utf-8 -*-
"""
Created on Tue Feb  8 12:46:51 2022

@author: dailerob
"""

import numpy as np
from PIL import Image
import pandas as pd
import os

path = r'C:\Users\rwtx\Desktop\MLDS_comp_2021\animations/metadata/'

path_list = os.listdir(path)



def get_vals(meta_image):
    
    meta_image = np.array(meta_image)
    
    save_array_r = np.zeros(256)
    save_array_g = np.zeros(256)
    save_array_b = np.zeros(256)
    
    num, count = np.unique(np.array(meta_image)[:,:,0], return_counts = True)
    save_array_r[num] = count
    
    num, count = np.unique(np.array(meta_image)[:,:,1], return_counts = True)
    save_array_g[num] = count
    
    num, count = np.unique(np.array(meta_image)[:,:,2], return_counts = True)
    save_array_b[num] = count
    
    save_array =np.stack([save_array_r, save_array_g, save_array_b], axis = 1)
    
    return save_array


def get_class(meta_image, threshold = 15):
    meta_image = np.array(meta_image)
    
    save_array_r = np.zeros(256)
    save_array_g = np.zeros(256)
    save_array_b = np.zeros(256)
    
    num, count = np.unique(np.array(meta_image)[:,:,0], return_counts = True)
    save_array_r[num] = count
    
    num, count = np.unique(np.array(meta_image)[:,:,1], return_counts = True)
    save_array_g[num] = count
    
    num, count = np.unique(np.array(meta_image)[:,:,2], return_counts = True)
    save_array_b[num] = count
    
    save_array =np.stack([save_array_r, save_array_g, save_array_b], axis = 1)
    
    class_array = np.arange(38)
    
    class_array[0] = np.sum(save_array[9:11,0]) > threshold
    class_array[1] = np.sum(save_array[25:28,0]) > threshold
    class_array[2] = np.sum(save_array[43:46,0]) > threshold
    class_array[3] = np.sum(save_array[61:64,0]) > threshold
    class_array[4] = np.sum(save_array[78:81,0]) > threshold
    class_array[5] = np.sum(save_array[94:96,0]) > threshold
    class_array[6] = np.sum(save_array[108:111,0]) > threshold
    class_array[7] = np.sum(save_array[121:124,0]) > threshold
    class_array[8] = np.sum(save_array[133:136,0]) > threshold
    class_array[9] = np.sum(save_array[143:146,0]) > threshold
    class_array[10] = np.sum(save_array[153:156,0]) > threshold
    class_array[11] = np.sum(save_array[161:164,0]) > threshold
    class_array[12] = np.sum(save_array[169:172,0]) > threshold
    class_array[13] = np.sum(save_array[176:179,0]) > threshold
    class_array[14] = np.sum(save_array[182:185,0]) > threshold
    class_array[15] = np.sum(save_array[188:190,0]) > threshold
    class_array[16] = np.sum(save_array[193:195,0]) > threshold
    class_array[17] = np.sum(save_array[197:200,0]) > threshold
    class_array[18] = np.sum(save_array[202:204,0]) > threshold
    class_array[19] = np.sum(save_array[9:11,1]) > threshold
    class_array[20] = np.sum(save_array[25:28,1]) > threshold
    class_array[21] = np.sum(save_array[43:46,1]) > threshold
    class_array[22] = np.sum(save_array[61:64,1]) > threshold
    class_array[23] = np.sum(save_array[78:81,1]) > threshold
    class_array[24] = np.sum(save_array[94:96,1]) > threshold
    class_array[25] = np.sum(save_array[108:111,1]) > threshold
    class_array[26] = np.sum(save_array[121:124,1]) > threshold
    class_array[27] = np.sum(save_array[133:136,1]) > threshold
    class_array[28] = np.sum(save_array[143:146,1]) > threshold
    class_array[29] = np.sum(save_array[153:156,1]) > threshold
    class_array[30] = np.sum(save_array[161:164,1]) > threshold
    class_array[31] = np.sum(save_array[169:172,1]) > threshold
    
    class_array[32] = np.sum(save_array[41:43,2]) > threshold
    class_array[33] = np.sum(save_array[89:92,2]) > threshold
    class_array[34] = np.sum(save_array[128:131,2]) > threshold
    class_array[35] = np.sum(save_array[157:159,2]) > threshold
    class_array[36] = np.sum(save_array[178:180,2]) > threshold
    class_array[37] = np.sum(save_array[193:196,2]) > threshold
    
    return class_array

    
    


count_vals = [get_vals(Image.open(path + i))for i in path_list[0:100]]
counts = np.stack(count_vals, axis = 2).sum(axis = 2)


columns = ["ceiling_fan","cieling_light","lamp","wall_light","kitchen fauncy light","window","art","bed","sofa","ottoman","rug","lounge chair","mirror","toilet","tub","sink","cabinet","overhead cabinet","kitchen_table","oven","refrigerator","desk","shelves","table","dresser","desk_light","book","plant","laptop","tv","chair","door",'hardwood','dark_hardwood','tile','hex tile','hex_marble_tile','marble_tile', 'bathroom', 'office','bedroom','kitchen','living room']

class_vals = [get_class(Image.open(path+i)) for i in path_list]

room_list = pd.read_csv('room_list.csv')
room_list['room_list'] = room_list['room_list'].astype(str)
room_list = pd.get_dummies(room_list)

class_vals2 = [np.array(list(class_vals[i]) + list(room_list.iloc[i,:])) for i in range(len(class_vals))]

tall_class = np.concatenate(class_vals2)



ids = np.array([str(int((np.floor(i/43))+1)).zfill(5) + '_' + str(i%43+1) for i in range(860000)])
tall_frame = pd.DataFrame(tall_class, ids, columns = ['value'])
tall_testing = tall_frame[:430000]
tall_training = tall_frame[430000:]

tall_training.to_csv('training.csv')
tall_testing.to_csv('testing.csv')

class_vals= pd.DataFrame(np.stack(class_vals2), columns = columns)



testing = class_vals[0:10000]
testing.to_csv('training_wide.csv')

#training = class_vals[10000:]

