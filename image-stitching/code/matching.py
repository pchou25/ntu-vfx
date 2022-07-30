import cv2
import numpy as np
import sys,os
import math
import logging,argparse
import random
from msop import *
from utils import *
def feature_bf(f1,f2):
    ind = []
    sc = []
    for i in range(len(f1)):
        best = 0
        loss = np.inf
        for j in range(len(f2)):
            L = ((f1[i] - f2[j])**2).sum()
            if loss > L:
                loss = L
                best = j
        ind.append(best)
        sc.append(loss)
    t = sorted(sc)[-1]
    feature_bf.ind2 = ind.copy()
    for i in range(len(ind)):
        if sc[i] > t:
            feature_bf.ind2[i] =  -1
    return ind,sc


def ransac(pos1,pos2,match,it = 900 , thres = 1000):
    best , ret_mat = 0, (0,0)
    # random.seed(0)
    for i in range(len(pos1)):

        a=i

        xs = pos1[a][0] - pos2[match[a]][0]
        ys = pos1[a][1] - pos2[match[a]][1]

        inlier = 0
        for i,p1 in enumerate(pos1):
            if feature_bf.ind2[i] <0 :
                continue
            x,y = p1
            x2,y2 = pos2[match[i]]
            x2 += xs
            y2 += ys

            loss = (x2-x,y2-y)
            loss = (np.array(loss)**2).sum()
            if loss < thres:
                inlier +=1
        if inlier > best:
            best = inlier
            ret_mat = (xs,ys)

    return ret_mat

def draw_match(img1,img2,p1,p2,match,sc):
    new_img = np.concatenate((img1,img2),axis=1)
    for i,j in enumerate(match):
        p2cat = (p2[j][0],p2[j][1]+img1.shape[1])
        color = (0, 255, 0)
        pcolor = (0, 0, 255)
        if feature_bf.ind2[i] ==-1  :
            continue
        elif  feature_bf.ind2[i] ==-2:
            color = (20,20,0)
            color = (0,0,0)
            pcolor = (0,0,0)
            continue
        cv2.line(new_img , p1[i][::-1],  p2cat[::-1] , color , 1)
        cv2.circle(new_img, p1[i][::-1], 3, color = pcolor)
        cv2.circle(new_img, p2cat[::-1], 3, color = pcolor)
    return new_img

def blending_twoimg(imga,imgb,x,y):
    if x <0:
        imga = np.concatenate([np.zeros((abs(x),imga.shape[1],3)),imga],axis=0)
        x=0
    if y<0:
        imga = np.concatenate((np.zeros((imga.shape[0],abs(y),3)),imga),axis=1)
        y=0
    siz = [imga.shape[0]+x,imga.shape[1]+y]

    imgb = shift(imgb,y,x,siz = siz[::-1])


    ret = np.zeros((imga.shape[0] +x ,imga.shape[1]+y ,3))
    ret[:imga.shape[0] , :imga.shape[1]] = imga
    ret+= imgb

    rng = imga.shape[1]-y
    for j in range(y,imga.shape[1]):
        alpha = (j-y)/rng
        ret[x:imga.shape[0],j] = (1-alpha) * imga[x:,j]  + alpha * imgb[x:imga.shape[0],j] 

    return ret

def blending(imgs,offsets):
    assert len(imgs) == len(offsets)
    assert len(imgs) > 1
    ret = imgs[0].copy()
    x0,y0 = offsets[0]
    offset_sum = np.array(offsets)
    offset_sum = [offset_sum[:i+1].sum(axis = 0) for i in range(0,len(offset_sum))]
    logging.info(f'offsetsum : {offset_sum},oofset:\n{offsets}')
    offsets 

    for i in range(len(imgs)):
        xs,ys = offsets[i]
        img_t = imgs[i]
        logging.info(f'padding:{i},{ret.shape}')
        ret = np.pad(ret, [(max(0,-xs),max(0,xs)),(max(0,-ys),max(0,ys)),(0,0)] )
        logging.info(f'stiching:{i},{ret.shape}')
        logging.debug(f'offsets : {offset_sum},\n{offsets}')
        if xs < 0:
            for j in range(len(imgs)):
                offset_sum[j][0]-=xs
        if ys < 0:
            for j in range(len(imgs)):
                offset_sum[j][0]-=ys
        xs,ys = offset_sum[i]
        

        for x in range(xs,xs + img_t.shape[0]):
            for y in range(ys,ys+ img_t.shape[1]):
                if img_t[x-xs,y-ys].sum() !=0:
                    try:
                        ret[x,y] = img_t[x-xs,y-ys]
                    except:
                        print(x,y,img_t.shape,xs,ys)
                        raise
        if i!=0:

            for x in range(xs,offset_sum[i-1][0] + imgs[i-1].shape[0]):
                if x-xs >= imgs[i].shape[0]:
                    break
                for y in range(ys,offset_sum[i-1][1] + imgs[i-1].shape[1]):
                    if y-ys >= imgs[i].shape[1]:
                        break

                    alpha = (y-ys)/(offset_sum[i-1][1] + imgs[i-1].shape[1]-ys)

                    if imgs[i-1][x - offset_sum[i-1][0]].sum()>1 and  imgs[i][x-xs,y-ys].sum()>1:
                        ret[x,y] = (1-alpha) * imgs[i-1][x - offset_sum[i-1][0] , y-offset_sum[i-1][1]]+ alpha* imgs[i][x-xs,y-ys]
                        
                    elif imgs[i-1][x - offset_sum[i-1][0]].sum()>1:
                        ret[x,y] = imgs[i-1][x - offset_sum[i-1][0] , y-offset_sum[i-1][1]]
                        
                    elif imgs[i][x-xs,y-ys].sum()>1:
                        ret[x,y] = imgs[i][x-xs,y-ys]

                    else:
                        pass


    return ret

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
       "-log", 
       "--log", 
       default="info",
       )
    logging.basicConfig(level=parser.parse_args().log.upper(),format='')


    path = './data/denny/'
    file_list = os.listdir(path)
    file_list.sort()
    testimg = [cv2.imread(os.path.join(path,file_list[i]),cv2.IMREAD_GRAYSCALE) for i in range(4)]
    testrgb = [cv2.imread(os.path.join(path,file_list[i])) for i in range(len(testimg))]
    
    focal=650
    for i in range(len(testimg)):
        testimg[i] = proj(testimg[i],focal = focal)
        testrgb[i] = proj(testrgb[i],focal = focal)

    c1,c2 = 2,3
    f1 , p1 = msop(testimg[c1] , debug=True)
    f2 , p2 = msop(testimg[c2] , debug=False)

    match , sc= feature_bf(f1,f2)
    print(f1.shape,p1.shape,f2.shape,p2.shape)
    draw_match(testrgb[c1],testrgb[c2],p1,p2,match,sc)

    sh = ransac(p1,p2,match)

    print(sh)

    imgb = shift(testrgb[c2],sh[1],sh[0])
    cv2.imwrite('shift.jpg',imgb)

    cv2.imwrite('stack1.jpg',T)
    cv2.imwrite('stack.jpg',blending(testrgb[c1],testrgb[c2],sh[0],sh[1]))
