from moviepy.video.io.VideoFileClip import VideoFileClip

from search import *

with open('classifier.p', 'rb') as f:
    classifier = pickle.load(f)
finder = CarFinder()
find_cars = finder.find_cars


def pipe(img):
    draw_img = np.copy(img)
    box_list = []
    out_img, _list = find_cars(img, draw_img, 1, classifier.svc, classifier.X_scaler, classifier.orient,
                               classifier.pix_per_cell, classifier.cell_per_block, classifier.spatial,
                               classifier.histbin)
    box_list = box_list + _list

    out_img, _list = find_cars(img, draw_img, 1.5, classifier.svc, classifier.X_scaler, classifier.orient,
                               classifier.pix_per_cell, classifier.cell_per_block, classifier.spatial,
                               classifier.histbin)
    box_list = box_list + _list

    out_img, _list = find_cars(img, draw_img, 2, classifier.svc, classifier.X_scaler, classifier.orient,
                               classifier.pix_per_cell, classifier.cell_per_block, classifier.spatial,
                               classifier.histbin)
    box_list = box_list + _list

    draw_img, heatmap = heat_map(img, box_list)
    return draw_img

if __name__ == '__main__':
    # clip = VideoFileClip("test_video.mp4")
    # processed_clip = clip.fl_image(pipe)  # NOTE: this function expects color images!!
    # processed_clip.write_videofile("test_video_out.mp4", audio=False)

    clip = VideoFileClip("project_video.mp4")
    processed_clip = clip.fl_image(pipe)  # NOTE: this function expects color images!!
    processed_clip.write_videofile("project_video_out.mp4", audio=False)

