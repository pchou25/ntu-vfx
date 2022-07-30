import cv2
import numpy as np
import sys,os
import math
import logging,argparse
import random
from msop import *
from utils import *
from matching import *
import glob
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-log", "--log", default="info")
    parser.add_argument("-p", "--path", default="./data/")
    parser.add_argument("-f", "--focal", default=650)
    parser.add_argument("-o", "--out", default='./')
    parser.add_argument("-n", "--num", default=99)
    parser.add_argument("-m", "--match", default=False,action='store_true')
    parser.add_argument("-r", "--rev", default=False,action='store_true')
    parser.add_argument("-s", "--scale", default=1)
    parser.add_argument("-k", "--draw_key", default=False,action='store_true')
    logging.basicConfig(level=parser.parse_args().log.upper(),format='')


    path = parser.parse_args().path
    focal = int(parser.parse_args().focal)
    outdir = parser.parse_args().out
    outmatch = bool(parser.parse_args().match)
    scale = int(parser.parse_args().scale)
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    file_list = glob.glob(os.path.join(path,'*.jpg'))
    imgnum = min(len(file_list),int(parser.parse_args().num))
    file_list.sort(reverse = parser.parse_args().rev)
    
    img_gray = [cv2.imread(file_list[i],cv2.IMREAD_GRAYSCALE) for i in range(imgnum)]
    img_rgb = [cv2.imread(file_list[i]) for i in range(imgnum)]
    


    
    for i in range(len(img_gray)):
        img_gray[i] = proj(img_gray[i],focal = focal)
        img_rgb[i] = proj(img_rgb[i],focal = focal)
        logging.info(f'proj {file_list[i]} size {img_gray[i].shape}')


    if scale > 1 :
        for i in range(imgnum):
            img_gray[i] = cv2.resize(img_gray[i] , np.array(img_gray[i].shape[::-1])//scale)
            img_rgb[i] =  cv2.resize(img_rgb[i] , np.array(img_rgb[i].shape[1::-1])//scale)
            logging.info(f'{file_list[i]} size {img_gray[i].shape}')

    features = []
    pos = []

    for i in range(len(img_gray)):

        f,p = msop(img_gray[i].copy() , debug=False and i == 0)
        logging.info(f'msop img {i} , key points number {len(f)} , ')
        features.append(f)
        pos.append(p)

    matches =[]
    scores=[]
    offsets=[(0,0)]
    draws = []
    for i in range(1,len(img_gray)):

        m , sc = feature_bf(features[i-1],features[i] )

        thres = []# remove matches too far away
        for j in range(len(pos[i-1])):
            p1 = pos[i-1][j]
            p2 = list(pos[i][m[j]] )
            p2[1]+=img_rgb[0].shape[1]
            thres.append((dis(p1,p2) , j))
        thres.sort()
        for d,j in thres[len(thres)//4:]:
            feature_bf.ind2[j]=-2


        off = ransac(pos[i-1],pos[i],m)

        logging.info(f'offset {i-1},{i} : {off}')

        if outmatch:
            draw = draw_match(img_rgb[i-1],img_rgb[i],pos[i-1],pos[i],m,sc)
            cv2.imwrite(os.path.join(outdir,f'draw{i-1}_{i}.jpg'),draw)

        matches.append(m)
        scores.append(sc)
        offsets.append(off)

    offsets  = list(map(list,offsets))
    output = blending(img_rgb,offsets)

    
    cv2.imwrite(os.path.join(outdir,'output.jpg'),output)
    cv2.imwrite(os.path.join(outdir,'output2.jpg'),blending_twoimg(img_rgb[i-1],img_rgb[i],offsets[i][0],offsets[i][1]))