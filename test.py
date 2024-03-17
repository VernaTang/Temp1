# from transformers import AutoTokenizer, T5ForConditionalGeneration
#
# tokenizer = AutoTokenizer.from_pretrained("t5-small")
# model = T5ForConditionalGeneration.from_pretrained("t5-small")
#
# # training
# input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
# labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
# outputs = model(input_ids=input_ids, labels=labels)
# loss = outputs.loss
# logits = outputs.logits
#
# # inference
# input_ids = tokenizer(
#     "The <extra_id_0> walks <extra_id_1> home", return_tensors="pt"
# ).input_ids  # Batch size 1
# outputs = model.generate(input_ids)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
# # studies have shown that owning a dog is good for you.
# import torch
# a = torch.tensor([[2,3],[2,1],[1,3]])
# print(a)
# b = torch.logsumexp(a, -1)
# print(b)
# torch.dist(torch.logsumexp(a, 1), torch.log(torch.sum(torch.exp(a), 1)))

import os
import cv2
import argparse
import shutil
import math

test_video = os.path.join("D:\PycharmProjects\Temp1\data\original\MELD\train_video", "dia1038_utt15.mp4")

cap = cv2.VideoCapture(test_video)
print(cap.get(5))

if cap and cap.get(5) > 0:
    frames_num = cap.get(7)
    rate = cap.get(5)
    duration = round(frames_num / rate)
    print("The video is normal!!!!!!!")


    c = 0
    j = 0
    ret = True
