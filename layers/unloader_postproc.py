from .output_utils import postprocess
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn


from itertools import chain
from copy import deepcopy as dp
from PIL import Image as im

def unloader_pp(det_output,h ,w, top_k = 15, score_threshold = 0.5):
    

    t = postprocess(det_output, w, h, score_threshold = score_threshold)

    idx = t[1].argsort(0, descending=True)[:top_k]
    # Masks are drawn on the GPU, so don't copy    
    masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    
    # masks_np is numpy and for cpu
    masks_np = (masks).byte().cpu().numpy()
    
    """
    class change
    classes = np.where(classes == 'your_class', 1,classes)
    classes = np.where(classes == 'your_class2', 2,classes)
    """

    instance_ch = np.array(0 ,int)
    class_ch = np.array(0 ,int)
    all_new_list =[]

    tmp_list_for_Gp = []
    dict_for_Gp = {'num_objs': masks_np.shape[0], 'image_size': (masks_np.shape[1],masks_np.shape[2])}
    dict_for_Gp.setdefault('seg_length',[])
    dict_for_Gp.setdefault('scores',[])
    dict_for_Gp.setdefault('bboxes',[])
    
    # make new list for sort by size of mask and set index of all pixels
    for i in range(len(classes)):
        tmp_dict = {}
        tmp_dict = {'size': masks_np[i, :, :].sum(),'masks': masks_np[i, :, :], 'classes': classes[i],
       'scores': scores[i], 'bboxes': boxes[i]}
        all_new_list.append(tmp_dict.copy())

    all_new_list = sorted(all_new_list, key=lambda k: k['size'], reverse = True) 

    # intergrate images
    for i in range(len(classes)):
        #class channel 
        class_ch1 = np.array(0 ,int)
        tmp_class = int(all_new_list[i]['classes'])
        class_ch1 = all_new_list[i]['masks'] * tmp_class
        cond2 = class_ch1 == tmp_class
        class_ch = np.where(cond2,class_ch1,class_ch)

        #instance channel
        all_new_list[i].update(masks = all_new_list[i]['masks']* (i+1))
        
        cond = all_new_list[i]['masks'] == i+1
        
        instance_ch = np.where(cond, all_new_list[i]['masks'], instance_ch)
   
    for i in range(len(classes)):
      
        #seperate all mask by instance
        cond3 = np.where(instance_ch == i+1,1,0)
        cond3 = cond3.astype(np.uint8)
        #get contours for each mask
        contours, hierarchy = cv2.findContours(cond3, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        #get biggest mask contour
        contours.sort(key=cv2.contourArea, reverse=True)
       
      
        #flatten np.array
        tmp_list_for_Gp.append(contours[0].flatten('F'))
      
        #save seg_length
        dict_for_Gp['seg_length'].append(len(contours[0]))
        #save scores
        dict_for_Gp['scores'].append(all_new_list[i]['scores'])
        #save boxes
        dict_for_Gp['bboxes'].append(all_new_list[i]['bboxes'])
        '''
        if you wanna check the contour result

        img3 = cv2.drawContours(cond3, [contours[0]], -1, (255,0,0), 3)
        img3 = im.fromarray(img3*50)
        img3.save('output_images/output_contour'+str(i)+'.png')
        '''
    #flatten and save list of sementation and boxes
    flatten_list = [j for sub in tmp_list_for_Gp for j in sub]
    dict_for_Gp['segm_masks'] = flatten_list
    bbox_flatten_list = [j for sub in dict_for_Gp['bboxes'] for j in sub]
    dict_for_Gp['bboxes']= bbox_flatten_list
    

    # all_new_list =>[{'size: int(area of mask), 'masks': overlapped(np.adrray), 
    # 'classes': (numpy.int64), 'scores': (numpy.float32), 'boxes': (numpy.adarray)}]
    
    # np to image and save
    
    """
    save instance and class channel

    img = im.fromarray(instance_ch*10)
    img.save('output_images/output_instance.png')
    img2 = im.fromarray(class_ch*50)
    img2.save('output_images/output_class.png')
    """
    # dict_for_gp = > {'segm_masks':[int_list],'seg_length': int, 'scores' : [float_list]
    #                 , 'bboxes': [int_list], 'image_size' : (h, w) , 'num_objs': int}
    return instance_ch, class_ch, dict_for_Gp
