from collections import deque

from classify import *

dist_pickle = pickle.load(open("svc.sav", "rb"))
print(dist_pickle)

svc = dist_pickle["svc"]
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


def process_frame(img):
    # Window searching

    # import time
    #
    # start_p = time.time()

    boxes = find_cars(img, scaler, svc, config)
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


def movie(file, output_path="output_videos", subclip=None):
    from moviepy.editor import VideoFileClip

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    output = 'output_videos/%s' % file
    clip = VideoFileClip(file)
    if subclip is not None:
        clip = clip.subclip(subclip)
    processed_clip = clip.fl_image(process_frame)
    processed_clip.write_videofile(output, audio=False)

movie("test_video.mp4")
# movie("project_video.mp4")
