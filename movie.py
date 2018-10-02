from collections import deque

import moviepy.editor as mp
from moviepy.editor import VideoFileClip

from classify import *

dist_pickle = pickle.load(open("svc.sav", "rb"))
print(dist_pickle)

svc = dist_pickle["svc"]
rf = dist_pickle["rf"]

scaler = dist_pickle["scaler"]
config = dist_pickle["config"]

img = mpimg.imread("./test_images/test1.jpg")

config["y_start"] = 400  # Min and max in y to search in slide_window()
config["y_stop"] = 656  # Min and max in y to search in slide_window()
config["scale"] = 1.5
config["threshold"] = 1

bins = 10
img_size = (720, 1280)
weights = np.arange(1, bins + 1) / bins

heat_map_q = deque(np.array([np.zeros(img_size).astype(np.float)]), maxlen=bins)


def weighted_average(points, weights):
    return np.average(points, 0, weights[-len(points):])

def process_frame_svm(img):
    return process_frame(img, svc)

def process_frame_rf(img):
    return process_frame(img, rf)

def process_frame(img, model):
    # Window searching

    # import time
    #
    # start_p = time.time()

    boxes = find_cars(img, scaler, model, config)
    # Heat mapping
    # print("find cars = ", time.time() - start_p)
    heat = np.zeros_like(img[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, boxes)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, config["threshold"])

    if np.all(heat <= config["threshold"]):
        heat_map_q.extend(np.array([heat_map_q[-1]]))
    else:
        heat_map_q.extend(np.array([heat]))

    # Apply threshold
    heat_map_avg = weighted_average(heat_map_q, weights)
    heat_map_avg = apply_threshold(heat_map_avg, config["threshold"])

    # Final annotated image
    # Visualize the heatmap when displaying
    heat_map = np.clip(heat_map_avg, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heat_map)
    heat_img = draw_labeled_bboxes(np.copy(img), labels)
    # print("heatmap = ", time.time() - start_p)
    return heat_img


def movie(file, output_path="output_videos", subclip=None, start=0):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    original_video = VideoFileClip(file)
    print("segment_length=", subclip)
    print("duration=", original_video.duration)
    if subclip is not None:
        duration = original_video.duration

        segment_length = subclip

        clips_rf = []
        clips_svc = []
        # the first segment starts at 0 seconds
        clip_start = start

        # make new segments as long as clip_start is
        # less than the duration of the video
        while clip_start < duration:
            clip_end = clip_start + segment_length

            # make sure the the end of the clip doesn't exceed the length of the original video
            if clip_end > duration:
                clip_end = duration

            # create a new moviepy videoclip, and add it to our clips list
            clip = original_video.subclip(clip_start, clip_end)
            processed_clip = clip.fl_image(process_frame_rf)
            processed_clip.write_videofile('output_videos/%s-%s-%s-%s' % ("rf", clip_start, clip_end, file),
                                           audio=False)
            clips_rf.append(clip)

            processed_clip = clip.fl_image(process_frame_svm)
            processed_clip.write_videofile('output_videos/%s-%s-%s-%s' % ("svc", clip_start, clip_end, file),
                                           audio=False)
            clips_svc.append(clip)


            clip_start = clip_end

        final_video_rf = mp.concatenate_videoclips(clips_rf)
        final_video_rf.write_videofile('output_videos/rf-%s' % (file), audio=False)

        final_video_svc = mp.concatenate_videoclips(clips_svc)
        final_video_svc.write_videofile('output_videos/rf-%s' % (file), audio=False)

    else:
        processed_clip = original_video.fl_image(process_frame)
        processed_clip.write_videofile('output_videos/%s' % (file), audio=False)


# movie("test_video.mp4", subclip = .5)
movie("project_video.mp4", subclip=1)
