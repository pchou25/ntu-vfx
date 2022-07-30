import cv2
import numpy as np
import sys,os
import math
import logging,argparse
    
def proj(img,focal=100):
    hi,wi = np.indices(img.shape[:2]).astype(float)
    hi -= img.shape[0]/2
    wi -= img.shape[1]/2
    logging.debug(f'{hi.shape}{img.shape}')
    s = focal

    hp = s*hi/np.sqrt(wi**2+focal**2)
    wp = s*np.arctan(wi/focal)

    hp = np.round(hp+img.shape[0]/2).astype(int)
    wp = np.round(wp+img.shape[1]/2).astype(int)

    cyn = np.zeros(shape=img.shape, dtype=np.uint8)
    cyn[hp,wp] = img
    cyn = cyn[hp.min():hp.max()+1,wp.min():wp.max()+1]
    return cyn

def shift(img,x,y, siz=None):
    m = np.float32([[1,0,x],[0,1,y]])
    if siz!=None:
        ret = cv2.warpAffine(img,m,siz)
    else:
        ret = cv2.warpAffine(img,m,img.shape[:2][::-1])
    return ret

def dis(p1,p2):
    return np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )
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
    testimg = cv2.imread(os.path.join(path,file_list[0]),cv2.IMREAD_GRAYSCALE)
    f=100
    p = proj(testimg,f)
    cv2.imwrite(f'proj.jpg',p )
