import json
import torch
import os, glob
from tqdm import tqdm
import multiprocessing

def get_feature_indices(start_time, end_time):
    """
    Find indices of features that correspond to a given time period.
    
    Args:
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
            
    Returns:
        list: List of feature indices whose windows overlap with the given time period
        
    Note:
        - Stride is 16/30 seconds (≈ 0.533 seconds)
        - Window size is 32/30 seconds (≈ 1.067 seconds)
        - A feature at index i corresponds to window [i*stride, i*stride + window_size]
    """
    STRIDE = 16/30  # seconds
    WINDOW = 32/30  # seconds
    
    # Find the first feature index whose window might overlap with start_time
    # Window [t, t + 32/30] overlaps with start_time if t + 32/30 > start_time
    # Solving: idx * 16/30 + 32/30 > start_time
    # idx > (30*start_time - 32)/16
    first_idx = max(0, int((30 * start_time - 32) // 16))
    
    # Find the last index whose window starts before end_time
    # Window [t, t + 32/30] overlaps with end_time if t < end_time
    # Solving: idx * 16/30 < end_time
    # idx < 30*end_time/16
    last_idx = int((30 * end_time) // 16)
    
    # Return range of indices
    return first_idx, last_idx

def process_one_take_by_segment(take_json):
    take_uid = take_json['take_uid']
    take_name = take_json['take_name']

    feature_dict = f'/home/ubuntu/AutomataLearning/egoexo4d/data/features/egovlpv2'
    pattern = os.path.join(feature_dict, f'{take_uid}.pt')
    feature_file = glob.glob(pattern)
    if not feature_file:
        raise FileExistsError(f'No feature file found: {take_name}')
    feature = torch.load(feature_file[0], weights_only=True).squeeze() # (num_windows, features_dim)

    video_embedding_trace = []
    for segment_json in take_json['segments']:
        first_idx, last_idx = get_feature_indices(segment_json['start_time'], segment_json['end_time'])
        video_embedding_trace.append(
            torch.mean(feature[first_idx:last_idx], dim=0)
        )
    del feature
    return torch.stack(video_embedding_trace)

def process_by_segment(takes, worker_id):
    meta_data = []
    video_embedding_data = []

    for take_json in tqdm(takes):
        try:
            video_embedding_trace = process_one_take_by_segment(take_json)
        except (FileExistsError, ValueError) as e:
            print(f"Error processing {take_json['take_name']}: {e}")
            continue
        meta_data.append(take_json)
        video_embedding_data.append(video_embedding_trace)
        
    print(f'worker {worker_id} processed {len(video_embedding_data)}')
    return meta_data, video_embedding_data

def split_takes(takes, num_workers):
    """Split the data into num_workers parts."""
    return [takes[i::num_workers] for i in range(num_workers)]

def multi_process_by_segment(set):
    num_workers = 2
    with open(f'data/annotations/keystep_{set}.json', 'r') as f:
        meta_json = json.load(f)
    take_splits = split_takes(list(meta_json['annotations'].values()), num_workers)

    pool = multiprocessing.Pool(num_workers)
    results = pool.starmap(process_by_segment, [(takes, i) for i, takes in enumerate(take_splits)])
    pool.close()
    pool.join()
    
    meta_data = []
    video_embedding_data = []
    for result in results:
        meta_data.extend(result[0])
        video_embedding_data.extend(result[1])

    torch.save(meta_data, f"data_embedding/egovlpv2_segment_metadata_{set}.pt")
    torch.save(video_embedding_data, f"data_embedding/egovlpv2_segment_video_embedding_{set}.pt")

if __name__=='__main__':
    multi_process_by_segment('train')
    multi_process_by_segment('val')


