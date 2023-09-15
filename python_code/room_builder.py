# -*- coding: utf-8 -*-
"""
Spyder Editor

this file generates room layouts to be used by blender when generating images.
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



#adj_p = the probability of a unit if a single adjacent unit is positive 
#corn_p = the probability of a unit if two or more adjacent units are positive
def gen_layout(num_v = 20, adj_p = .04, corn_p = .95):
    test = np.zeros((num_v,num_v))
    
    center = int(num_v/2)
    
    test[center-1:center+2,center-1:center+2] = 1.0
    
    num_rounds = 10
    
    for i in range(num_rounds):
        
        test_sum = test.copy()*4
        test_sum[:,1:] += test[:,:-1] #left sum
        test_sum[:,:-1] += test[:,1:] #right sum
        
        test_sum[1:,:] += test[:-1,:] #top sum
        test_sum[:-1,:] += test[1:,:] #bottom_sum
        
        
        pmat = test_sum.copy()
        
        pmat[pmat<=0] = 0
        pmat[pmat == 1] = adj_p
        pmat[pmat == 2] = corn_p
        pmat[pmat > 2] = 1
        
        rand_mat = np.random.binomial(1,pmat, (num_v,num_v)).astype(float)
        
        test = rand_mat.copy()
    plt.imshow(test)
    
    test[0,:] = 0
    test[-1,:] = 0
    test[:,0] = 0
    test[:,-1] = 0
    
    return test



def rotate_mat(mat, n = 1):
    if n == 1:
        return mat[::-1,:].T
    else:
        for i in range(n):
            mat = mat[::-1,:].T
        return mat

#tests if A meets the corner block analysis
def test_A(layout, loc):
    
    x = loc[0]
    y = loc[1]
    
    test_mat_1 = np.array([[0,0,0],[0,1,1],[0,1,1]])
    test_mat_2 = rotate_mat(test_mat_1, 1)
    test_mat_3 = rotate_mat(test_mat_1, 2)
    test_mat_4 = rotate_mat(test_mat_1, 3)
    
    test_mat_list = [test_mat_1, test_mat_2,test_mat_3,test_mat_4]
    
    test_region = layout[x-1:x+2,y-1:y+2]
    
    comp_array = [np.sum(np.abs(i-test_region)) for i in test_mat_list] #blocks that match will sum to zero in the comparison
    
    rotation = -1
    if any(comp_array == 0):
        rotation = np.argwhere(comp_array == 0)
    
    return rotation
    

#A
#for 2*2 block
mat_A = np.array([[1,1,1],
                  [1,1,1],
                  [1,1,1]])

mask_A = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])

#B
#for corner
mat_B = np.array([[0,0,0],
                  [0,1,1],
                  [0,1,1]])

mask_B = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])

#C
#for border block 
mat_C = np.array([[0,0,0],
                  [0,1,1],
                  [0,0,0]])

mask_C = np.array([[0,0,0],
                   [1,1,1],
                   [0,0,0]])

#E
#for floating 1*2
mat_E = np.array([[0,0,0],
                  [0,1,1],
                  [0,0,0]])

mask_E = np.array([[0,0,0],
                   [0,1,1],
                   [0,0,0]])


#H
#for 2*2 that is bordered
mat_H = np.array([[0,0,0],
                  [1,1,1],
                  [1,1,1]])

mask_H = np.array([[1,1,1],
                   [1,1,1],
                   [1,1,1]])


mat_dict = {}
mat_dict['mat_A'] = mat_A
mat_dict['mask_A'] = mask_A
mat_dict['mat_B'] = mat_B
mat_dict['mask_B'] = mask_B
mat_dict['mat_C'] = mat_C
mat_dict['mask_C'] = mask_C
#mat_dict['mat_D'] = mat_D
#mat_dict['mask_D'] = mask_D
mat_dict['mat_E'] = mat_E
mat_dict['mask_E'] = mask_E
#mat_dict['mat_F'] = mat_F
#mat_dict['mask_F'] = mask_F
#mat_dict['mat_G'] = mat_G
#mat_dict['mask_G'] = mask_G
mat_dict['mat_H'] = mat_H
mat_dict['mask_H'] = mask_H

#layout - original layout of the room
#loc - the location of the block to modify
#rotations - which rotations we want to test
#mask - mask for if we want to only check certain squares in the 3*3, default is all points are tested
def test_block(layout, loc, test_mat, rotations = [0,1,2,3], test_mask = np.array([[1,1,1],[1,1,1],[1,1,1]])):
    
    x = loc[0]
    y = loc[1]
    
    test_mat_list = [rotate_mat(test_mat,i) for i in rotations]
    test_mask_list = [rotate_mat(test_mask,i) for i in rotations]
    
    test_region = layout[x-1:x+2,y-1:y+2]/2
    
    
    comp_array = np.array([np.sum(np.abs(mat-test_region)*mask) for mat,mask in zip(test_mat_list,test_mask_list)])
    
    
    rotation = -1
    if any(comp_array == 0):
        rotation = np.argwhere(comp_array == 0)[:,0]
        rotation = np.random.choice(rotation, 1)[0] #get index of one valid rotation
        rotation = rotations[rotation]#get valid rotation
        
    return rotation
        
    
    
    
    

#first class of locations witihin the room, this is a two by two
def gen_im(layout, mat_dict, room_type):
    
    
    mat_A = mat_dict['mat_A']
    mask_A = mat_dict['mask_A']
    mat_B = mat_dict['mat_B']
    mask_B = mat_dict['mask_B']
    mat_C = mat_dict['mat_C']
    mask_C = mat_dict['mask_C']
    mat_H = mat_dict['mat_H']
    mask_H = mat_dict['mask_H']
    
    
    layout_2 = layout.copy()*2
    layout_1 = np.zeros(layout.shape)
    rot_map = np.zeros(layout.shape)
    
    #############Set up B blocks ##################
    locs = np.argwhere(layout_2==2)
    num_blocks = locs.shape[0]
    locs = [locs[i,:] for i in range(num_blocks)]
    
    B_locs = np.array([test_block(layout_2, loc, mat_B, rotations = [0,1,2,3], test_mask = mask_B) for loc in locs])
    valid = np.argwhere(B_locs != -1)[:,0]
    if len(valid) != 0:
        choice = np.random.choice(valid, 1)[0]
        
        rotation = B_locs[choice]
        
        loc = locs[choice]
        x = loc[0]
        y = loc[1]
        place_array = rotate_mat(mat_B,rotation)*rotate_mat(mask_B,rotation)
        
        layout_2[x-1:x+2,y-1:y+2] -= place_array
        layout_1[x-1:x+2,y-1:y+2] += place_array*4
        layout_2[x,y] = 4
        rot_map[x,y] = rotation
    
    #############Set up A blocks ##################
    locs = np.argwhere(layout_2==2)
    num_blocks = locs.shape[0]
    locs = [locs[i,:] for i in range(num_blocks)]
    
    A_locs = np.array([test_block(layout_2, loc, mat_A, rotations = [0], test_mask = mask_A) for loc in locs])
    valid = np.argwhere(A_locs != -1)[:,0]
    if len(valid) != 0:
        choice = np.random.choice(valid, 1)[0]
        
        loc = locs[choice]
        x = loc[0]
        y = loc[1]
        place_array = mat_A*mask_A
        
        layout_2[x-1:x+2,y-1:y+2] -= place_array
        layout_1[x-1:x+2,y-1:y+2] += place_array*3
        layout_2[x,y] = 3
        rot_map[x,y] = 0
    
    if len(locs) > 25:
        locs = np.argwhere(layout_2==2)
        num_blocks = locs.shape[0]
        locs = [locs[i,:] for i in range(num_blocks)]
        
        A_locs = np.array([test_block(layout_2, loc, mat_A, rotations = [0], test_mask = mask_A) for loc in locs])
        valid = np.argwhere(A_locs != -1)[:,0]
        if len(valid) != 0:
            choice = np.random.choice(valid, 1)[0]
            
            loc = locs[choice]
            x = loc[0]
            y = loc[1]
            place_array = mat_A*mask_A
            
            layout_2[x-1:x+2,y-1:y+2] -= place_array
            layout_1[x-1:x+2,y-1:y+2] += place_array*3
            layout_2[x,y] = 3
            rot_map[x,y] = 0
            
            
    #############Set up H blocks ##################
    locs = np.argwhere(layout_2==2)
    num_blocks = locs.shape[0]
    locs = [locs[i,:] for i in range(num_blocks)]
    
    H_locs = np.array([test_block(layout_2, loc, mat_H, rotations = [0,1,2,3], test_mask = mask_H) for loc in locs])
    valid = np.argwhere(H_locs != -1)[:,0]
    if len(valid) != 0:
        choice = np.random.choice(valid, 1)[0]
        
        rotation = H_locs[choice]
        
        loc = locs[choice]
        x = loc[0]
        y = loc[1]
        place_array = rotate_mat(mat_H,rotation)*rotate_mat(mask_H,rotation)
        
        layout_2[x-1:x+2,y-1:y+2] -= place_array
        layout_1[x-1:x+2,y-1:y+2] += place_array*6
        layout_2[x,y] = 6
        rot_map[x,y] = rotation
    
    '''
    ###########Set up I blocks ####################
    for i in range(2):
        locs = np.argwhere(layout_2==2)
        num_blocks = locs.shape[0]
        locs = [locs[i,:] for i in range(num_blocks)]
        
        H_locs = np.array([test_block(layout_2, loc, mat_H, rotations = [0,1,2,3], test_mask = mask_H) for loc in locs])
        valid = np.argwhere(H_locs != -1)[:,0]
        if len(valid) != 0:
            choice = np.random.choice(valid, 1)[0]
            
            rotation = H_locs[choice]
            
            loc = locs[choice]
            x = loc[0]
            y = loc[1]
            place_array = rotate_mat(mat_H,rotation)*rotate_mat(mask_H,rotation)
            
            layout_2[x-1:x+2,y-1:y+2] -= place_array
            layout_2[x,y] = 7
            rot_map[x,y] = rotation
    '''
    
    for i in range(6):
        #############Set up C blocks ##################
        locs = np.argwhere(layout_2==2)
        num_blocks = locs.shape[0]
        locs = [locs[i,:] for i in range(num_blocks)]
        
        C_locs = np.array([test_block(layout_2, loc, mat_C, rotations = [0,1,2,3], test_mask = mask_C) for loc in locs])
        valid = np.argwhere(C_locs != -1)[:,0]
        if len(valid) != 0:
            choice = np.random.choice(valid, 1)[0]
            
            rotation = C_locs[choice]
            
            loc = locs[choice]
            x = loc[0]
            y = loc[1]
            place_array = rotate_mat(mat_C,rotation)*rotate_mat(mask_C,rotation)
            
            layout_2[x-1:x+2,y-1:y+2] -= place_array
            layout_1[x-1:x+2,y-1:y+2] += place_array*5
            layout_2[x,y] = 5
            rot_map[x,y] = rotation
    
    
    
    #layout*=255/2
    layout_2[layout_2<0] = 0
    layout_2*=(255/20)
    
    layout_1[layout_1<2] = 2
    layout_1 *= layout
    layout_1*=(255/20)
    
    
    
    rot_map*=(255/4)
    
    
    alpha = np.ones(layout.shape)*255
    alpha[0,0] = int(room_type/5*255)
    floor_im = np.stack([layout_1, layout_2, rot_map, alpha], axis =2)
    
    pil_Image = Image.fromarray(np.uint8(floor_im))
    
    return pil_Image



room_list = [0]
room_trans_mat = np.array([[.1,.6,.1,.1,.1],
                          [.1,.1,.6,.1,.1],
                          [.1,.1,.1,.6,.1],
                          [.1,.1,.1,.1,.6],
                          [.6,.1,.1,.1,.1]])

#training set
np.random.seed(1234)

for i in range(1,10001):
    
    room_type = room_list[-1]
    room_list.append(room_type)
    layout = gen_layout()
    pil_Image = gen_im(layout, mat_dict, room_type)
    pil_Image.save(r'C:\Users\rwtx\Desktop\MLDS_comp_2021\floor_layouts/' + str(i).zfill(4) + '.png')
    
    
    room_list.append(np.random.choice(np.arange(5),p= room_trans_mat[room_list[-1],:]))


#testing set
np.random.seed(76145)

for i in range(10001,20000):
    
    room_type = room_list[-1]
    room_list.append(room_type)
    layout = gen_layout()
    pil_Image = gen_im(layout, mat_dict, room_type)
    pil_Image.save(r'C:\Users\rwtx\Desktop\MLDS_comp_2021\floor_layouts/' + str(i).zfill(4) + '.png')
    
    
    room_list.append(np.random.choice(np.arange(5),p= room_trans_mat[room_list[-1],:]))
