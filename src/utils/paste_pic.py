import cv2
import os
import numpy as np
from tqdm import tqdm
import uuid
import concurrent.futures
from src.utils.videoio import save_video_with_watermark

def paste_pic(video_path, pic_path, crop_info, new_audio_path, full_video_path, extended_crop=False):

    if not os.path.isfile(pic_path):
        raise ValueError('pic_path must be a valid path to video/image file')
    elif pic_path.split('.')[-1] in ['jpg', 'png', 'jpeg']:
        full_img = cv2.imread(pic_path)
    else:
        video_stream = cv2.VideoCapture(pic_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        still_reading, frame = video_stream.read()
        video_stream.release()
        full_img = frame
    frame_h = full_img.shape[0]
    frame_w = full_img.shape[1]

    video_stream = cv2.VideoCapture(video_path)
    fps = video_stream.get(cv2.CAP_PROP_FPS)
    crop_frames = []
    while True:
        still_reading, frame = video_stream.read()
        if not still_reading:
            video_stream.release()
            break
        crop_frames.append(frame)
    
    # Provide crop_info and other parameters
    if extended_crop:
        # Define oy1, oy2, ox1, ox2 based on extended crop
        oy1, oy2, ox1, ox2 = ...
    else:
        # Define oy1, oy2, ox1, ox2 based on regular crop
        oy1, oy2, ox1, ox2 = ...

    tmp_path = str(uuid.uuid4())+'.mp4'
    
    def process_image(crop_frame):
        p = cv2.resize(crop_frame.astype(np.uint8), (ox2-ox1, oy2 - oy1))
        mask = 255*np.ones(p.shape, p.dtype)
        location = ((ox1+ox2) // 2, (oy1+oy2) // 2)
        gen_img = cv2.seamlessClone(p, full_img, mask, location, cv2.NORMAL_CLONE)
        return gen_img

    out_tmp = cv2.VideoWriter(tmp_path, cv2.VideoWriter_fourcc(*'mp4'), fps, (frame_w, frame_h))

    max_threads = 5

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_threads) as executor:
        processed_frames = []
        for gen_img in tqdm(executor.map(process_image, crop_frames), total=len(crop_frames), desc='seamlessClone:'):
            processed_frames.append(gen_img)

    for frame in processed_frames:
        out_tmp.write(frame)

    out_tmp.release()

    save_video_with_watermark(tmp_path, new_audio_path, full_video_path, watermark=False)
    os.remove(tmp_path)

