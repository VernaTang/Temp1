# Setting the duration to be 1 when the duration of video is 0
import os
import cv2
import argparse
import shutil
import math

#python image_process.py -p "./data/original/MELD/dev_video" -f 1
#python image_process.py -p "./data/original/MELD/train_video" -f 1
#python image_process.py -p "./data/original/MELD/test_video" -f 1
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--path', help = "please set a path", type=str)
parser.add_argument('-f', '--frames', help="set frames", type=int)
parser.add_argument('-t', '--times', help="set times", type=float)
args = parser.parse_args()

videos_src_path = args.path
print('now processing videos from {}'.format(videos_src_path))
# The output dir name
videos_save_path = os.path.abspath(os.path.dirname(videos_src_path)) + "/img_complete/" + videos_src_path.split("/")[-1]
if not os.path.exists(videos_save_path):
    os.makedirs(videos_save_path)

file_cnt = 0
videos = os.listdir(videos_src_path)
for each_video in videos:
    file_cnt += 1
    each_video_name = each_video.split('.')[0]
    each_video_full_path = os.path.join(videos_src_path, each_video)
    img_path = videos_save_path
    if not os.path.exists(img_path):
        os.mkdir(img_path)
    #If you want to make dir for each video
    # img_path = os.path.join(videos_save_path, each_video_name)
    # if os.path.exists(img_path):
    #     shutil.rmtree(img_path)
    # os.mkdir(img_path)

    cap = cv2.VideoCapture(each_video_full_path)

    if cap and cap.get(5) > 0:
        frames_num = cap.get(7)
        rate = cap.get(5)
        duration = round(frames_num / rate)
        print("video {} is setting\ntime length:{}s".format(each_video_name,duration))

        c = 0
        j = 0
        ret = True
        if args.frames:
            if frames_num >= args.frames:
                numbers = math.floor(frames_num / args.frames)
            else:
                numbers = 1
                print("video {} frames is out of range\nall frames:{}".format(each_video_name, int(frames_num)))
            while ret:
                ret, frame = cap.read()
                if frame is None:
                    continue
                c += 1
                if c % numbers == 0:
                    j += 1
                    if j <= args.frames:

                        cv2.imencode('.jpg', frame)[1].tofile(img_path + '/' + '{}.jpg'.format(each_video_name))
        else:
            if duration == 0:
                numbers = 1
            else:
                numbers = int(frames_num/duration*args.times)
            while ret:
                ret, frame = cap.read()
                if frame is None:
                    continue
                c += 1
                if c % numbers == 0:
                    cv2.imencode('.jpg', frame)[1].tofile(img_path + '/' + '{}.jpg'.format(each_video_name))

print('\nvideo numbers:', str(file_cnt), '\nAll is done')
