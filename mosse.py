import numpy as np
import cv2
from until import *
import os
import argparse
class mosse:
    def __init__(self,args,path):
        self.args = args
        self.img_path = path
        self.img_list = self.get_img_lists(self.img_path)
    def track(self):
        init_image = cv2.imread(self.img_list[0])
        init_frame = cv2.cvtColor(init_image,cv2.COLOR_BGR2GRAY)
        init_frame = init_frame.astype(np.float32)
        gt = cv2.selectROI('mosse', init_image, False, False)
        gt = np.array(gt).astype(np.int64)
        gauss_img = gauss(init_frame,gt,self.args.sigma)
        init_gt = gauss_img[gt[1]:gt[1]+gt[3],gt[0]:gt[0]+gt[2]]
        init_gt = np.uint8(init_gt * 255)
        cv2.imshow('1', init_gt)
        # cv2.waitKey()
        f = init_frame[gt[1]:gt[1]+gt[3],gt[0]:gt[0]+gt[2]]

        Ai, Bi = self.pro_training( f ,init_gt,gt)

        for i in range(len(self.img_list)):
            image = cv2.imread(self.img_list[i])
            image_frame = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
            image_frame = image_frame.astype(np.float32)
            if i == 0:
                Ai = self.args.lr * Ai
                Bi = self.args.lr * Bi
                plot = gt.copy()
                clip = np.array([plot[0],plot[1],plot[0]+plot[2],plot[1]+plot[3]])
            else:
                Hi = Ai / Bi
                f = image_frame[clip[1]:clip[3],clip[0]:clip[2]]
                f = cv2.resize(f, (gt[2],gt[3]))
                f = preprocessing(f)
                G = np.fft.fft2(f)*(np.conjugate(Hi))
                gi = normal(np.fft.ifft2(G))
                a = gi
                a = np.uint8(a * 255)
                cv2.imshow('1',a)

                g_max = np.max(gi)
                g_plot = np.where(gi == g_max)
                dx = int(g_plot[0] - gi.shape[0]/2)
                dy = int(g_plot[1] - gi.shape[1]/2)

                plot[0] = plot[0] + dx
                plot[1] = plot[1] + dy
                clip[0] = np.clip(plot[0],0,image.shape[1])
                clip[1] = np.clip(plot[1], 0, image.shape[0])
                clip[2] = np.clip(plot[0]+plot[2], 0, image.shape[1])
                clip[3] = np.clip(plot[1]+plot[3], 0, image.shape[0])

                f = image_frame[clip[1]:clip[3],clip[0]:clip[2]]
                try:
                    f = cv2.resize(f, (gt[2], gt[3]))
                    Ai = self.args.lr * (G * np.conjugate(np.fft.fft2(f))) + (1 - self.args.lr) * Ai
                    Bi = self.args.lr * (np.fft.fft2(f) * np.conjugate(np.fft.fft2(f))) + (1 - self.args.lr) * Bi
                except:
                    print('object disappear!')
                    break

            cv2.rectangle(image,(plot[0],plot[1]),(plot[0]+plot[2],plot[1]+plot[3]),(255, 0, 0), 3)
            cv2.imshow('mosse',image)
            cv2.waitKey(1000)
    def pro_training(self, f ,g,gt):
        height, weight = g.shape
        f = cv2.resize(f, (weight,height))
        f = preprocessing(f)
        F = np.fft.fft2(f)
        G = np.fft.fft2(g)
        Ai = G*np.conjugate(F)
        Bi = F*np.conjugate(F)
        for _ in range(self.args.number_xuanzhuang):
            f= preprocessing(random_xuanzhuang(f))
            Ai = Ai + G* np.conjugate(np.fft.fft2(f))
            Bi = Bi + np.fft.fft2(f)* np.conjugate(np.fft.fft2(f))
        return Ai,Bi


    def get_img_lists(self,img_path):
        img_list = []
        images_path = os.listdir(img_path)
        for image in images_path:
            if os.path.splitext(image)[1] == '.jpg':
                img_list.append(os.path.join(img_path,image))
        img_list.sort()
        return img_list

if __name__== '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('--sigma',type=float, default=10 ,help='the sigma')
    parse.add_argument('--lr',type=float,default=0.125,help='the learning rate')
    parse.add_argument('--number_xuanzhuang',type=float,default=128,help='the number of pretrain')
    args = parse.parse_args()
    image_path = '/home/lyc/mosse-object-tracking/datasets/surfer'
    tracker = mosse(args,image_path)
    tracker.track()