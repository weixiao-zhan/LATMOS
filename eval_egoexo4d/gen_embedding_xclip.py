import json
import multiprocessing
import numpy as np
import torch
import torchvision
from transformers import AutoProcessor, AutoModel
import glob
import os
from tqdm import tqdm

def setup_model(worker_id):
    gpu_id = worker_id % torch.cuda.device_count()
    device = torch.device(f"cuda:{gpu_id}")
    processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
    model = AutoModel.from_pretrained("microsoft/xclip-base-patch32").to(device)
    return device, processor, model
    
def load_video(take_name):
    """
    Decode the entire video with torchvision decoder.
    Args:
    take_name (str): take name
    Returns:
    video (torch.Tensor): Tensor of decoded frames of shape (num_frames, height, width, 3).
    fps (float): Frames per second of the video.
    """
    video_dict = f'/home/ubuntu/AutomataLearning/egoexo4d/data/takes/{take_name}/frame_aligned_videos/downscaled/448/'
    pattern = os.path.join(video_dict, 'aria*_214-1.mp4')
    video_files = glob.glob(pattern)
    if not video_files:
        raise FileExistsError(f'No video file found: {take_name}')

    video_file = video_files[0]
    video, _, metadata = torchvision.io.read_video(video_file)
    fps = metadata["video_fps"]
    if video.shape[0] == 0:
        raise FileExistsError(f'Video load error: {take_name}')
        
    return video, fps

def gen_embedding(frames, texts, device, processor, model):
    '''
    texts = [step_name, step_description]
    frames = [8, 448, 448, 3] shaped
    return: video embedding, [step_name.embedding, step_description.embedding]
    '''
    inputs = processor(
        videos=list(frames),
        text=texts,
        return_tensors="pt",
        # padding=True,
    )
    with torch.no_grad():
        outputs = model(**inputs.to(device))
    return outputs['video_embeds'].squeeze(), outputs['text_embeds'].squeeze()

# ----- process one take ----- #

def process_one_take_by_segment(take_json, 
                             device, processor, model):
    video, fps = load_video(take_json['take_name'])
    total_frames = video.shape[0]

    video_embedding_trace = []
    text_embedding_trace = []
    for segment_json in take_json['segments']:
        # get frames from video slices
        start_frame = min(int(segment_json['start_time'] * fps), total_frames - 1)
        end_frame = min(int(segment_json['end_time'] * fps), total_frames - 1)    
        indices = torch.linspace(start_frame, end_frame, 8).long()
        frames = video[indices]
        # gen embedding
        texts = segment_json['step_description']
        video_embedding, text_embedding = gen_embedding(frames, texts, device, processor, model)
        # store embedding
        video_embedding_trace.append(video_embedding)
        text_embedding_trace.append(text_embedding)
    
    # if not video_embedding_trace:
    #     raise ValueError(f'No valid segments found for {take_json["take_name"]}')
    
    # hint GC to free memory
    del video
    
    return torch.stack(video_embedding_trace), \
           torch.stack(text_embedding_trace)

def process_one_take_by_window(take_json, window_length,
                            device, processor, model):
    video, fps = load_video(take_json['take_name'])
    total_frames = video.shape[0]
    frames_per_window = int(window_length * fps)

    video_embedding_trace = []
    for start_frame in range(0, total_frames, frames_per_window):
        # get frames from video slices
        end_frame = min(start_frame + frames_per_window, total_frames-1)
        indices = torch.linspace(start_frame, end_frame, 8).long()
        frames = video[indices]
        # gen embedding        
        video_embedding, _ = gen_embedding(frames, [''], device, processor, model)
        # store embedding
        video_embedding_trace.append(video_embedding)
    
    # if not video_embedding_trace:
    #     raise ValueError(f'No valid segments found for {take_json["take_name"]}')
    # hint GC to free memory
    del video
    return torch.stack(video_embedding_trace)

# ----- process ----- #

def process_by_segment(takes, worker_id):
    device, processor, model = setup_model(worker_id)

    meta_data = []
    video_embedding_data = []
    text_embedding_data = []

    for take_json in tqdm(takes):
        try:
            video_embedding_trace, text_embedding_trace = \
                process_one_take_by_segment(take_json, device, processor, model)
        except (FileExistsError, ValueError) as e:
            print(f"Error processing {take_json['take_name']}: {e}")
            continue
        meta_data.append(take_json)
        video_embedding_data.append(video_embedding_trace)
        text_embedding_data.append(text_embedding_trace)
    print(f'worker {worker_id} processed {len(video_embedding_data)}')
    return meta_data, video_embedding_data, text_embedding_data

def process_by_window(takes, worker_id, window_length):
    device, processor, model = setup_model(worker_id)

    meta_data = []
    video_embedding_data = []

    for take_json in tqdm(takes):
        try:
            video_embedding_trace = \
                process_one_take_by_window(take_json, window_length, device, processor, model)
        except (FileExistsError, ValueError) as e:
            print(f"Error processing {take_json['take_name']}: {str(e)}")
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
    text_embedding_data = []
    for result in results:
        meta_data.extend(result[0])
        video_embedding_data.extend(result[1])
        text_embedding_data.extend(result[2])

    torch.save(meta_data, f"data_embedding/segment_metadata_{set}.pt")
    torch.save(video_embedding_data, f"data_embedding/segment_video_embedding_{set}.pt")
    torch.save(text_embedding_data, f"data_embedding/segment_text_embedding_{set}.pt")

def multi_process_by_window(set):
    num_workers = 4
    with open(f'data/annotations/keystep_{set}.json', 'r') as f:
        meta_json = json.load(f)
    take_splits = split_takes(list(meta_json['annotations'].values()), num_workers)

    window_length = 10
    pool = multiprocessing.Pool(num_workers)
    results = pool.starmap(process_by_window, [(takes, i, window_length) for i, takes in enumerate(take_splits)])
    pool.close()
    pool.join()
    
    meta_data = []
    video_embedding_data = []
    for result in results:
        meta_data.extend(result[0])
        video_embedding_data.extend(result[1])

    torch.save(meta_data, f"data_embedding/window{window_length}_metadata_{set}.pt")
    torch.save(video_embedding_data, f"data_embedding/window{window_length}_video_embedding_{set}.pt")

if __name__=='__main__':
    multi_process_by_segment('train')
