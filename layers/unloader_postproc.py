from .output_utils import postprocess
import cv2
import numpy as np
import torch
from itertools import chain
from copy import deepcopy as dp
from PIL import Image as im

def unloader_pp(det_output,h ,w, top_k = 15, score_threshold = 0.5):
    """ raw output data of Yolact -> Instance_channel, Class_channel, dictionary of Info
    Args:
        det_output (list): return of forward
        h (int): image height
        w (int): image width
        top_k (int, optional): maximum number of detected object. Defaults to 15.
        score_threshold (float, optional): score threshold. Defaults to 0.5.
    Returns:
        image_ch (numpy.ndarray): image channel with the size of (2, height, width), 2 is instance_ch and class_ch
        dict_for_YolactSegm (dictionary): {'segm_masks':List(np.int32),'seg_length': List(int), 
                                'scores' : List(np.float32), 'bboxes': List(np.int32), 
                                'image_size' : Tuple(int), 'num_objs': int, 'class_id' :List(np.int32)}
                                
                                instance_ch (numpy.ndarray): instance channel image with the size of (height, width), 
                                  and the type element is 'int'. called by image_ch[0]
                                class_ch (numpy.ndarray): class channel image with size of (height, width) 
                                  and the type element is 'int' called by image_ch[1]
                                segm_masks: contour x,y pixel
                                seg_length: the number of contour pixel point
                                scores: confidence score of each detected object
                                bboxes: 2 corners coordinate for bounding boxes
                                image_size: (height, width) of image
                                num_objs: the number of detected
                                class_id: class index of each object
    """
    # L36, see https://github.com/dbolya/yolact/blob/master/eval.py #L149
    t = postprocess(det_output, w, h, score_threshold = score_threshold)

    # L39-L42, see https://github.com/dbolya/yolact/blob/master/eval.py #L155-L160
    idx = t[1].argsort(0, descending=True)[:top_k]
    # Masks are drawn on the GPU, so don't copy    
    masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    
    # coppied from https://github.com/dbolya/yolact/blob/master/eval.py  end.
    
    #initial var
    instance_ch_tensor = torch.tensor(0 ,dtype = torch.float)
    class_ch_tensor = torch.tensor(0 ,dtype = torch.float)
    all_new_list_tensor= []
    tmp_list_for_YolactSegm_np = []

    #none instance case
    if masks.size()[0] == 0:
        tmp_dict_for_YolactSegm= {'seg_masks':[],'num_objs': 0, 'image_size': (h,w), 'class_id':[]
                    , 'seg_length':[], 'scores': [], 'bboxes': []}
        return instance_ch_tensor, class_ch_tensor, tmp_dict_for_YolactSegm

    dict_for_YolactSegm = {'num_objs': masks.size()[0], 'image_size': (masks.size()[1],masks.size()[2])}

    dict_for_YolactSegm.setdefault('seg_length',[])
    dict_for_YolactSegm.setdefault('scores',[])
    dict_for_YolactSegm.setdefault('bboxes',[])
    dict_for_YolactSegm.setdefault('class_id', [])
    
    # make new list and sort by size(area) of mask
    for i in range(len(classes)):
        all_new_list_tensor.append({'size': masks[i, :, :].sum(),
                            'class_id': (classes[i]+1).astype(np.int32),
                            'scores': scores[i], 
                            'bboxes': boxes[i].astype(np.int32),
                            'masks':masks[i,:,:]})
    all_new_list_tensor = sorted(all_new_list_tensor, key=lambda k: k['size'], reverse = True)

    # intergrate images
    for i in range(len(classes)):
        #class channel
        class_ch1_tensor = torch.tensor(0 ,dtype = torch.float)
        tmp_class_tensor = float(all_new_list_tensor[i]['class_id'])
        class_ch1_tensor = all_new_list_tensor[i]['masks'] * (tmp_class_tensor)
        cond2_tensor = class_ch1_tensor == tmp_class_tensor
        class_ch_tensor = torch.where(cond2_tensor,class_ch1_tensor,class_ch_tensor)

        #instance channel
        all_new_list_tensor[i].update(masks = all_new_list_tensor[i]['masks']* (i+1))
        cond_tensor = all_new_list_tensor[i]['masks'] == i+1
        instance_ch_tensor = torch.where(cond_tensor, all_new_list_tensor[i]['masks'], instance_ch_tensor)

    #torch.tensor -> numpy.ndarray
    instance_ch_np = (instance_ch_tensor).to(torch.int).cpu().numpy()
    class_ch_np = (class_ch_tensor).to(torch.int).cpu().numpy()  
  
    for i in range(len(classes)):

        #seperate all mask as a instance
        cond3_np = np.where(instance_ch_np == i+1,1,0)
        cond3_np = cond3_np.astype(np.uint8)
        
        #get contours for each mask
        contours_np, hierarchy = cv2.findContours(cond3_np, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        #get the biggest mask for each contour
        contours_np.sort(key=cv2.contourArea, reverse=True)

        #flatten np.array
        tmp_list_for_YolactSegm_np.append(contours_np[0].flatten(''))
        
        #save class_id
        dict_for_YolactSegm['class_id'].append(all_new_list_tensor[i]['class_id'])
        #save seg_length
        dict_for_YolactSegm['seg_length'].append(len(contours_np[0]))
        #save scores
        dict_for_YolactSegm['scores'].append(all_new_list_tensor[i]['scores'])
        #save boxes
        dict_for_YolactSegm['bboxes'].append(all_new_list_tensor[i]['bboxes'])

        """
        #Check contours
        img3 = cv2.drawContours(cond3_np, [contours_np[0]], -1, (255,0,0), 3)
        img3 = im.fromarray(img3*50)
        img3.save('output_images/output_contour_np'+str(i)+'.png')
        """

    #flatten and save list of segmentation and boxes
    flatten_list = list(chain.from_iterable(tmp_list_for_YolactSegm_np))
    dict_for_YolactSegm['segm_masks'] = flatten_list
    bbox_flatten_list = list(chain.from_iterable(dict_for_YolactSegm['bboxes']))
    dict_for_YolactSegm['bboxes']= bbox_flatten_list
    
    #combine two channels
    image_ch = np.stack((instance_ch_np,class_ch_np))
    
    """
    #check each channgel
    from PIL import Image as im
    img = im.fromarray(image_ch[0]*5000)
    img.save('output_images/output_instance.png')
    img2 = im.fromarray(image_ch[1]*10000)
    img2.save('output_images/output_class.png')
    """

    return image_ch, dict_for_YolactSegm


def unloader_pp_contour(det_output,h ,w, top_k = 15, score_threshold = 0.5):
    """ raw output data of Yolact -> dictionary for Cv Node, dictionary for grasp compute Node

    Args:
        det_output (list): return of forward
        h (int): image height
        w (int): image width
        top_k (int, optional): maximum number of detected object. Defaults to 15.
        score_threshold (float, optional): score threshold. Defaults to 0.5.

    Returns:
        dict_for_cv (dictionary): {'contours':List(np,int32), 'contour_length': List(int),
                                   'class_id' :List(np.int32)}
        dict_for_YolactSegm (dictionary): {'segm_masks':List(np.int32),'seg_length': List(int), 
                                'scores' : List(np.float32), 'bboxes': List(np.int32), 
                                'image_size' : Tuple(int), 'num_objs': int, 'class_id' :List(np.int32)}
                                

                                segm_masks, contours: contour x,y pixel
                                seg_length, contour_length: the number of contour pixel point
                                scores: confidence score of each detected object
                                bboxes: 2 corners coordinate for bounding boxes
                                image_size: (height, width) of image
                                num_objs: the number of detected
                                class_id: class index of each object
    """ 
       
    # L172, see https://github.com/dbolya/yolact/blob/master/eval.py #L149
    t = postprocess(det_output, w, h, score_threshold = score_threshold)

    # L175-L178, see https://github.com/dbolya/yolact/blob/master/eval.py #L155-L160
    idx = t[1].argsort(0, descending=True)[:top_k]
    # Masks are drawn on the GPU, so don't copy    
    masks = t[3][idx]
    classes, scores, boxes = [x[idx].cpu().numpy() for x in t[:3]]
    
    # coppied from https://github.com/dbolya/yolact/blob/master/eval.py  end.
    
    #torch.tensor -> numpy.ndarray
    masks_np = (masks).to(torch.int).cpu().numpy()
    
    #initial var
    all_new_list = []
    tmp_list_for_YolactSegm_np = []
    dict_for_cv = {}
    
    # make new list
    for i in range(len(classes)):
        all_new_list.append({'class_id': (classes[i]+1).astype(np.int32),
                            'scores': scores[i], 
                            'bboxes': boxes[i].astype(np.int32),
                            'masks':masks[i,:,:]})
    
    #none instance case
    if masks.size()[0] == 0:
        tmp_dict_for_YolactSegm= {'seg_masks':[],'num_objs': 0, 'image_size': (h,w), 'class_id':[]
                    , 'seg_length':[], 'scores': [], 'bboxes': []}
        tmp_dict_for_cv = {'class_id':[], 'instance_id':[], 'contours': [], 'contour_length':[]}

        return tmp_dict_for_cv, tmp_dict_for_YolactSegm
    
    #default setting
    dict_for_YolactSegm = {'num_objs': masks.size()[0], 'image_size': (masks.size()[1],masks.size()[2])}
    dict_for_YolactSegm.setdefault('seg_length',[])
    dict_for_YolactSegm.setdefault('scores',[])
    dict_for_YolactSegm.setdefault('bboxes',[])
    dict_for_YolactSegm.setdefault('class_id', [])
    
    dict_for_cv.setdefault('class_id',[])
    dict_for_cv.setdefault('contours',[])
    dict_for_cv.setdefault('contour_length',[])
    
    for i in range(len(classes)):
        
        cond = masks_np[i, :, :]
        cond = cond.astype(np.uint8)
        #get contours for each mask
        contours_np, hierarchy = cv2.findContours(cond, cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

        #get the biggest mask for each contour
        contours_np.sort(key=cv2.contourArea, reverse=True)

        #flatten np.array
        tmp_list_for_YolactSegm_np.append(contours_np[0].flatten('C'))
        
        #save class_id
        dict_for_YolactSegm['class_id'].append(all_new_list[i]['class_id'])
        dict_for_cv['class_id'].append(all_new_list[i]['class_id'])
        #save seg_length
        dict_for_YolactSegm['seg_length'].append(len(contours_np[0]))
        dict_for_cv['contour_length'].append(len(contours_np[0]))
        #save scores
        dict_for_YolactSegm['scores'].append(all_new_list[i]['scores'])
        #save boxes
        dict_for_YolactSegm['bboxes'].append(all_new_list[i]['bboxes'])
        
        """
        #Check contours
        img3 = cv2.drawContours(cond, [contours_np[0]], -1, (255,0,0), 3)
        img3 = im.fromarray(img3*50)
        img3.save('output_images/output_contour_np'+str(i)+'.png')
        """
       
    #flatten and save list of segmentation and boxes
    flatten_list = list(chain.from_iterable(tmp_list_for_YolactSegm_np))
    dict_for_YolactSegm['segm_masks'] = flatten_list
    dict_for_cv['contours'] = flatten_list
    bbox_flatten_list = list(chain.from_iterable(dict_for_YolactSegm['bboxes']))
    dict_for_YolactSegm['bboxes']= bbox_flatten_list

    return dict_for_cv, dict_for_YolactSegm