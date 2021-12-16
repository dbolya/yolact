# Batch Slip and Fall (Headless)

__version__ = 1.5
__author__ = 'Ajay Bhargava'

from utils import VideoReaders, DetectorLoader
import numpy as np, pandas as pd, glob
import cv2
from skimage import measure, color, segmentation
from SORT.sort import *
import warnings
from itertools import groupby

class ChainedAssignment:
    def __init__(self, chained=None):
        acceptable = [None, 'warn', 'raise']
        assert chained in acceptable, "chained must be in " + str(acceptable)
        self.swcw = chained

    def __enter__(self):
        self.saved_swcw = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = self.swcw
        return self

    def __exit__(self, *args):
        pd.options.mode.chained_assignment = self.saved_swcw

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = DetectorLoader.YOLACT('./weights/yolact_resnet50_54_800000.pth', threshold = 0.11)

def detect_fall(path, video_output = False):
    '''
    Docstring for detect_slip_fall()

    Arguments
    -----------
    str(path): Takes a path and sends it to the function for inference and subsequent action

    Returns
    -----------
    output(dict): A dictionary with the output of the model {"Frame": frame, "BBOX": bbox} 
    which is the first frame and person position where the person fell down. 
    '''

    frame_provider = VideoReaders.VideoReader(path)
    length, shape = frame_provider.properties()

    output_inference = []
    frames = []

    # Run Model
    for frame in frame_provider:
        frames.append(frame)
        c, s, bb, ma = model.predict(frame)
        idx = np.where(c == 0)
        pixelwise_arrays = []
        for item in idx:
            for n, qq in enumerate(item):
                pixelwise = np.zeros_like(ma[qq,...])
                pixelwise = ma[qq,...].astype(np.uint8)
                pixelwise[np.where(pixelwise == 1)] = n + 1
                pixelwise_arrays.append(pixelwise)

        filtered = []
        for array in pixelwise_arrays:
            if np.max(array) != 0:
                filtered.append(array)

        stacked = np.sum(filtered, axis = 0)
        output_inference.append(stacked)

    output = []
    for arry in output_inference:
        if len(arry.shape) < 2:
            output.append(np.zeros(shape).astype(np.uint8))
        else:
            output.append(arry)
    
    if len(output) != len(frames):
        raise ValueError("Frame Lengths are off.")

    if video_output:
        boundaries = []
        for im, mask in zip(frames, output):
            boundaries.append(segmentation.mark_boundaries(im, mask, mode = 'thick', color = (1,0,0)))

    # Derive measurement of angles
    minimal_dictionary = []
    for i in range(0, length):
        for region in measure.regionprops(output[i], color.rgb2gray(frames[i])):
            minimal_dictionary.append({
                "Frame": i, 
                "ID": region.label, 
                "BBOX": region.bbox
            })
    tracked_bboxes = []
    Sorter = Sort(max_age = length, min_hits = 1, iou_threshold = 0.01) # 1 Tunable parameter (iou_threshold)
    for instance, entry in groupby(minimal_dictionary, key = lambda x:x['Frame']):
        entry_list = []
        for item in entry:
            lst = list(item['BBOX'])
            lst.extend([0, item['ID']])
            entry_list.append(lst)
        track_bbs_ids = Sorter.update(np.array(entry_list))
        for objects in track_bbs_ids:
            r0, c0, r1, c1, ID, label = objects.tolist()
            for region in measure.regionprops(output[instance], color.rgb2gray(frames[instance])):
                if region.label == label:
                    this_normalized_moment = region.moments_normalized
                    angle = np.degrees((np.arctan2(2*this_normalized_moment[1,1], this_normalized_moment[2,0] - this_normalized_moment[0,2]))/2)
                    w = region.bbox[2] - region.bbox[0]
                    h = region.bbox[3] - region.bbox[1]
                    ar = w / float(h)
                    tracked_bboxes.append({'Frame': instance, 
                                            'ID': int(label), 
                                            'Track ID': int(ID), 
                                            'BBOX': region.bbox,
                                            'Angle': angle,
                                            'Area': region.area,
                                            'Aspect Ratio': ar, 
                                            'Eccentricity': region.eccentricity, 
                                            'Perimeter': region.perimeter})
    tracked_dataframe = pd.DataFrame(tracked_bboxes)
    # Note frames where fall was detected
    super_output = []
    if len(tracked_dataframe) == 0:
        super_output.append({'File': None, 'Frame': None, 'Bounding Box': None})
    else:
        for tracks, track_df in tracked_dataframe.groupby('Track ID'):
            current_track = track_df.loc[(track_df['Track ID'] == tracks), :]
            with ChainedAssignment():
                current_track['Rate of Change'] = current_track.loc[(current_track['Track ID'] == tracks), :]['Angle'].pct_change(5, fill_method = 'ffill') # 1 Tunable parameters (Periodicity for rate of change in angle)
                current_track['Slip-Fall'] = np.where(current_track['Rate of Change'] < -10, 'Slip', 'Stand') # 1 Tunable Parameter (Hard Cutoff for % change in angle)
            if len(current_track.loc[current_track['Slip-Fall'] == 'Slip']) != 0:
                super_output.append({'Track ID': tracks, 'File': os.path.basename(path), 'Frame': current_track.loc[current_track['Slip-Fall'] == 'Slip'].iloc[0]['Frame'], 'Bounding Box': current_track.loc[current_track['Slip-Fall'] == 'Slip'].iloc[0]['BBOX']})
            else:
                super_output.append({'Track ID': None, 'File': None, 'Frame': None, 'Bounding Box': None})
    
    fall_ids = pd.DataFrame(super_output)
    fall_ids = fall_ids.dropna()


    if video_output:
        aggregate_bbox_df = []
        
        if len(fall_ids) == 0:
            return boundaries, fall_ids
        else:
            for track, data in fall_ids.groupby('Track ID'):
                investigate_frame = int(data['Frame'].values)
                investigate_track = track
                cv2_dataframe = tracked_dataframe[(tracked_dataframe['Track ID'] == investigate_track) & (tracked_dataframe['Frame'] >= investigate_frame)][['Track ID', 'Frame', 'BBOX', 'Angle']]
                aggregate_bbox_df.append(cv2_dataframe)
            
            if len(aggregate_bbox_df) == 0:
                return boundaries, fall_ids
            else:
                concat_bbox_df = pd.concat(aggregate_bbox_df)
                bbox_dictionary = concat_bbox_df.groupby('Frame').agg(tuple).applymap(list).reset_index()
                cv2_final_dictionary = pd.concat([bbox_dictionary.set_index('Frame').reindex(range(0, bbox_dictionary.Frame.min())).ffill().reset_index(), bbox_dictionary])
                
                final_frames = []
                for frame, info in cv2_final_dictionary.groupby('Frame'):
                    if info.loc[info['Frame'] == frame]['Track ID'].isnull().values:
                        final_frames.append(boundaries[frame])
                    else:
                        for track, angle, item in zip(info['Track ID'], info['Angle'], info['BBOX']):
                            for t, an, tuple_object in zip(track, angle, item):
                                text = 'Track:' + str(t) + ' ' + str(an)[:4] + 'Degrees'
                                x = tuple_object[0]
                                y = tuple_object[1]
                                w = tuple_object[2] - tuple_object[0]
                                h = tuple_object[3] - tuple_object[1]
                                image_output = cv2.rectangle(boundaries[frame], (y,x), (y + h, x + w), (36,255,12), 1)
                                image_output = cv2.putText(image_output, str(text), (y, x - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                        final_frames.append(image_output)
                return final_frames, fall_ids
    else:
        return fall_ids

# confidence_list = [0.1, 0.15, 0.20, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# pathlist = sorted(glob.glob('../datasets/slip-fall/fall-database/*.mp4'))
# for confidence_value in confidence_list:
#     for path in pathlist:
#         with warnings.catch_warnings():
#             warnings.simplefilter("ignore")
#             model = DetectorLoader.YOLACT('./weights/yolact_resnet50_54_800000.pth', threshold = confidence_value)
#         path_out = os.path.basename(path)
#         final_frames, fall_ids = detect_fall(path, video_output = True)
#         fourcc = cv2.VideoWriter_fourcc(*'MP4V')
#         converted_confidence_value = str(confidence_value).replace('.', '-')
#         if not os.path.exists("./results/slip-videos/{}-confidence-threshold/".format(converted_confidence_value)):
#             os.makedirs("./results/slip-videos/{}-confidence-threshold/".format(converted_confidence_value))
#         writer = cv2.VideoWriter("./results/slip-videos/{}-confidence-threshold/{}".format(converted_confidence_value, path_out), 0x7634706d, 5.0, max([x.shape for x in final_frames])[:-1][::-1])
#         for capture in final_frames:
#             out_capture = (capture * 255).astype(np.uint8)
#             writer.write(out_capture)
#         writer.release()
#         if not os.path.exists("./results/charts/slip-videos/{}-confidence-threshold/".format(converted_confidence_value)):
#             os.makedirs("./results/charts/slip-videos/{}-confidence-threshold/".format(converted_confidence_value))
#         fall_ids.to_csv('./results/charts/slip-videos/{}-confidence-threshold/{}.csv'.format(converted_confidence_value, os.path.splitext(path_out)[0]))

confidence_list = [0.20, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
pathlist = sorted(glob.glob('../datasets/slip-fall/no-fall-database/*.mp4'))
for confidence_value in confidence_list:
    for path in pathlist:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DetectorLoader.YOLACT('./weights/yolact_resnet50_54_800000.pth', threshold = confidence_value)
        path_out = os.path.basename(path)
        final_frames, fall_ids = detect_fall(path, video_output = True)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        converted_confidence_value = str(confidence_value).replace('.', '-')
        if not os.path.exists("./results/no-slip-videos/{}-confidence-threshold/".format(converted_confidence_value)):
            os.makedirs("./results/no-slip-videos/{}-confidence-threshold/".format(converted_confidence_value))
        writer = cv2.VideoWriter("./results/no-slip-videos/{}-confidence-threshold/{}".format(converted_confidence_value, path_out), 0x7634706d, 5.0, max([x.shape for x in final_frames])[:-1][::-1])
        for capture in final_frames:
            out_capture = (capture * 255).astype(np.uint8)
            writer.write(out_capture)
        writer.release()
        if not os.path.exists("./results/charts/no-slip-videos/{}-confidence-threshold/".format(converted_confidence_value)):
            os.makedirs("./results/charts/no-slip-videos/{}-confidence-threshold/".format(converted_confidence_value))
        fall_ids.to_csv('./results/charts/no-slip-videos/{}-confidence-threshold/{}.csv'.format(converted_confidence_value, os.path.splitext(path_out)[0]))

        vars = ['final_frames', 'fall_ids', 'model']
        for v in vars:
            del v
