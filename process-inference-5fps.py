from utils import VideoReaders, DetectorLoader
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from itertools import groupby
import warnings
from SORT.sort import *
from skimage import measure, color, draw
from scipy.spatial.distance import cdist
import seaborn as sns
from tqdm.auto import tqdm
tqdm.pandas()

ground_truths = pd.read_csv('datasets/slip-fall/original-holdout/ground-truths-updated.csv')
dataset = []
for flag, df in tqdm(ground_truths.groupby('Condition')):
    pathlist = [os.path.join(
        '../datasets/slip-fall/holdout-videos/',
        x
    ) for x in df['File ID']]
    df['Paths'] = pathlist
    dataset.append(df)

dataset = pd.concat(dataset)

def check_video_length(x):
    '''
    Uses the VideoLoader to detect if the frames are too long for use in our dataset on a series of data in a dataframe. 
    '''
    frame_provider = VideoReaders.VideoReader(x)
    length, shape = frame_provider.properties()
    if length > 100:
        return "Too Big."
    else:
        return "Just Right."

dataset['Usage'] = dataset['Paths'].apply(check_video_length)
usable_data = dataset[(dataset['Usage'] == 'Too Big.')]
display(usable_data[usable_data['Condition'] == 'Fall'].sort_values('File ID'))

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

def instantaneous_time(x):
    frame_interval = 0.2
    return (x[1] - x[0]) * frame_interval

def angular_displacement(x):
    return (x[1] - x[0])

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model = DetectorLoader.YOLACT('./weights/slip-fall-chosen-weights/yolact_resnet50_432_74900.pth', threshold = 0.25)

pbar = tqdm(usable_data.groupby('Paths'))
for path, _ in pbar:
    if os.path.basename(path) == '0301.mp4':
        frame_provider = VideoReaders.VideoReader(path)
        length, shape = frame_provider.properties()
        print('{} is {} frames.'.format(os.path.basename(path),length))

        frames = []
        inference = []
        for frame in frame_provider:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            c, s, bb, ma = model.predict(frame[:,:,[2,1,0]])
            idx = np.where(c == 0) # Person has a Class ID of 0
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

            stacked = np.sum(filtered, axis = 0) # Will cause the array pixel values to increase in numerical value. This also maintains multiple people in the frame. 
            inference.append(stacked)

        output = []
        for arry in inference: # Check if the shape coming out of the model is OK.
            if len(arry.shape) < 2:
                output.append(np.zeros(shape).astype(np.uint8))
            else:
                output.append(arry)

        def check(list1, val):
            return(any(x < val for x in list1))

        # IGNORE ZONES CODE: Remove anything that's not in the frame
        for frame_position in range(0, length):
            generated_image = np.zeros_like(output[frame_position])
            for pixel in np.unique(output[frame_position]):
                if pixel == 0: 
                    pass
                else:
                    ignore_perimeter = 5 # Tunable parameter for ignore zone
                    end_extent = (generated_image.shape[0] - ignore_perimeter, generated_image.shape[1] - ignore_perimeter)
                    perim_rr, perim_cc = draw.rectangle_perimeter(start = (ignore_perimeter, ignore_perimeter), end = end_extent)
                    rectangle_coords = np.stack((perim_rr, perim_cc), axis = 1)
                    generated_image[rectangle_coords[:,0], rectangle_coords[:,1]] = 1 # Debugging
                    for otherregion in measure.regionprops((output[frame_position] == pixel).astype(np.uint8)):
                        mask_coordinates = otherregion.coords
                        generated_image[mask_coordinates[:,0], mask_coordinates[:,1]] = 1
                    distances = cdist(mask_coordinates, rectangle_coords, metric = 'euclidean') # Compute the euclidean distance between points and imposed boundary 
                    minimum_distance = np.min(distances, axis = 1).tolist()
                    integer_minimum_distances = [int(item) for item in minimum_distance]
                    if check(integer_minimum_distances, 1):
                        output[frame_position][output[frame_position] == pixel] = 0

        # Derive measurement of angles and link pixelwise masks using SORT
        minimal_dictionary = []
        for i in range(0, length):
            for region in measure.regionprops(output[i], color.rgb2gray(frames[i])):
                minimal_dictionary.append({
                    "Frame": i, 
                    "ID": region.label, 
                    "BBOX": region.bbox
                })
        tracked_bboxes = []
        Sorter = Sort(max_age = 5, min_hits = 1, iou_threshold = 0.001) # 1 Tunable parameters for SORT
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
                        angle = (np.arctan2(2*this_normalized_moment[1,1], this_normalized_moment[2,0] - this_normalized_moment[0,2]))/2 # Normalized image moments 
                        w = region.bbox[2] - region.bbox[0]
                        h = region.bbox[3] - region.bbox[1]
                        ar = w / float(h)
                        tracked_bboxes.append({'Frame': instance, 
                                                'ID': int(label), 
                                                'Track ID': int(ID), 
                                                'BBOX': region.bbox,
                                                'θ': angle,
                                                'Area': region.area,
                                                'Aspect Ratio': ar, 
                                                'Eccentricity': region.eccentricity, 
                                                'Perimeter': region.perimeter})
        tracked_dataframe = pd.DataFrame(tracked_bboxes)
        
        # Calculate angular acceleration. 
        super_output = []
        if len(tracked_dataframe) == 0:
            super_output.append({'File': None, 'Frame': None, 'Bounding Box': None, 'ID': None, 'θ': None, 'dθ': None, 'Δt': None, '⍵': None, '⍵^2': None})
        else:
            for tracks, track_df in tracked_dataframe.groupby('Track ID'):
                # TODO Might want to add an if statement to suppress tracks with length less than track age. 
                current_track = track_df.loc[(track_df['Track ID'] == tracks), :]
                with ChainedAssignment():
                    current_track['dθ'] = current_track.loc[(current_track['Track ID'] == tracks), :]['θ'].rolling(window = 2).apply(angular_displacement, raw = True).fillna(0)
                    current_track['Δt'] = current_track.loc[(current_track['Track ID'] == tracks), :]['Frame'].rolling(window = 2).apply(instantaneous_time, raw = True).fillna(0).cumsum()
                    current_track['⍵'] = current_track.loc[(current_track['Track ID'] == tracks), :]['dθ'].diff() / current_track.loc[(current_track['Track ID'] == tracks), :]['Δt'].diff().fillna(0)
                    current_track['⍵^2'] = current_track.loc[(current_track['Track ID'] == tracks), :]['⍵'].diff() / current_track.loc[(current_track['Track ID'] == tracks), :]['Δt'].diff()
                    current_track['Flag'] = np.where(current_track['⍵^2'].abs() > 40, 'Fallen-Human', 'Human')
                    super_output.append(current_track)
        fall_identities = pd.concat(super_output)

        # Export plots. 
        try:
            fall_identities = fall_identities.dropna()
            df = pd.melt(fall_identities[['ID', 'Track ID', 'Frame', 'θ', 'dθ', 'Δt', '⍵', '⍵^2']], 
                    id_vars = ['Track ID', 'Frame'], 
                    value_vars = ['dθ', '⍵', '⍵^2'],
                    var_name = 'Parameter',
                    value_name = 'Value')
            if len(np.unique(df['Track ID'])) > 1:
                with sns.axes_style('whitegrid'):
                    sns.despine()
                    g = sns.relplot(data = df, x = 'Frame', y = 'Value', hue = 'Parameter', col = 'Track ID', kind = 'line', aspect = 1.5)
                    g.set(ylabel = 'θ (radians)', ylim = (-150, 150))
                g.tight_layout()
                g.savefig('datasets/slip-fall/original-holdout/charts/{}'.format(os.path.basename(path).replace('.mp4', '.png')), dpi = 300)
                plt.close(g.fig)
            else:
                f,ax = plt.subplots(1)
                with sns.axes_style('whitegrid'):
                    sns.despine()
                    sns.lineplot(data = df, x = 'Frame', y = 'Value', hue = 'Parameter', ax = ax)
                    ax.set(ylabel = 'θ (radians)', ylim = (-150, 150))
                f.tight_layout()
                f.savefig('datasets/slip-fall/original-holdout/charts/{}'.format(os.path.basename(path).replace('.mp4', '.png')), dpi = 300)
                plt.close(f)
            print('{}, chart complete!'.format(os.path.basename(path)))
        except TypeError:
            print('{} did not produce a chart.'.format(os.path.basename(path)))
            continue

        # Produce annotated video. 
        aggregate_bbox_df = []
        fall_identities = fall_identities.dropna()
        for track, data in fall_identities.groupby('Track ID'):
            investigate_frame = data['Frame'].values
            to_merge = tracked_dataframe[(tracked_dataframe['Track ID'] == track) & (tracked_dataframe['Frame'].isin(investigate_frame))][['Frame', 'ID', 'Track ID', 'BBOX']]
            cv2_dataframe = data.merge(to_merge[['Frame', 'ID', 'Track ID', 'BBOX']], on = ['Frame', 'ID', 'Track ID', 'BBOX'])
            aggregate_bbox_df.append(cv2_dataframe)

            final_frames = []
            if len(aggregate_bbox_df) == 0:
                final_frames.append(frames)
            else:
                concat_bbox_df = pd.concat(aggregate_bbox_df)
                bbox_dictionary = concat_bbox_df.groupby('Frame').agg(tuple).applymap(list).reset_index()
                cv2_final_xx = pd.concat([bbox_dictionary.set_index('Frame').reindex(range(0, bbox_dictionary.Frame.min())).ffill().reset_index(), bbox_dictionary])
                cv2_final_dictionary = cv2_final_xx.set_index('Frame').reindex(range(0, len(frames)), fill_value = np.NaN).reset_index()

                for frame, info in cv2_final_dictionary.groupby('Frame'):
                        if info.loc[info['Frame'] == frame]['Track ID'].isnull().values:
                            final_frames.append(frames[frame])
                        else:
                            for track, acceleration, bounds, angle, flagging in zip(info['Track ID'], info['⍵^2'], info['BBOX'], info['θ'], info['Flag']):
                                    for t, acc, tuple_object, theta, flag in zip(track, acceleration, bounds, angle, flagging):
                                            text = '{:.0f} radians/s^2'.format(int(abs(acc)))
                                            x = tuple_object[0]
                                            y = tuple_object[1]
                                            w = tuple_object[2] - tuple_object[0]
                                            h = tuple_object[3] - tuple_object[1]

                                            # Calculate Angle line for slipping/falling object
                                            y0c, x0c = int(y + h/2), int(x + w/2)
                                            y1c, x1c = int(x0c + 50 * np.cos(theta)), int(y0c + 50 * np.sin(theta))
                                            
                                            # Set colors
                                            if 0 < abs(acc) < 10:
                                                    color_value = (118,238,0)
                                            elif 11 < abs(acc) < 50:
                                                    color_value = (255,211,67)
                                            elif abs(acc) > 51: 
                                                    color_value = (255,48,48)
                                            else:
                                                    pass
                                            image_output = cv2.rectangle(frames[frame], (y,x), (y + h, x + w), color_value, 1)
                                            image_output = cv2.putText(image_output, str(text), (y, x - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_value, 1)
                            final_frames.append(image_output)

        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        writer = cv2.VideoWriter("datasets/slip-fall/original-holdout/videos/{}".format(os.path.basename(path)), 0x7634706d, 5.0, max([x.shape for x in final_frames])[:-1][::-1])
        for capture in final_frames:
            out_capture = cv2.cvtColor(capture, cv2.COLOR_BGR2RGB)
            writer.write(out_capture)
        writer.release()
        print('Output processed video for {}'.format(os.path.basename(path)))
