import cv2
import numpy as np
import sys,os
import math
import logging,argparse
 
def non_maximal_sup(corner_func ,local_max, anms = 400 , thres=10):

	H,W = corner_func.shape
	points = [(corner_func[i][j],i,j) for i in range(H) for j in range(W)  if (local_max[i][j] == corner_func[i][j] and local_max[i][j] >= thres)]
	points.sort(reverse=True)
	logging.debug(local_max.shape)
	logging.info(f"anms candidates:{len(points)}")
	R = (H**2+W**2)/32

	keypts=[points[0]]
	it = 0
	valid = [1 for i in range(len(points))]

	while len(keypts) < anms:

		for i,p in enumerate(points):
			if valid[i] != 1 :
				continue
			add = True
			for k in keypts:
				if (k[1] - p[1])**2 + (k[2] - p[2])**2  <R:
					add = False
					break
			if add:
				keypts.append(p)
				valid[i] = 0
				if len(keypts) > anms:
					break
		logging.debug(f'ANMS　it:{it} , key:{len(keypts)} , R:{R}')

		it+=1
		R/=1.5
		if R < 1:
			break
	return keypts

def sub_pixel_refinement(corner_func , keys):
	refine_point = []
	cf = corner_func
	for _,i,j in keys:
		if min(i,j)==0 or i+1 == corner_func.shape[0] or j+1 == corner_func.shape[1]:
			continue
		f_x = (cf[i][j+1] - cf[i][j-1])/2
		f_y = (cf[i+1][j] - cf[i-1][j])/2
		ff_xx = cf[i][j+1] - 2*cf[i][j] + cf[i][j-1]
		ff_yy = cf[i+1][j] - 2*cf[i][j] + cf[i-1][j]
		ff_xy =  (cf[i+1][j+1] - cf[i+1][j-1] - cf[i-1][j+1] + cf[i-1][j-1] )/4 + 1e-5

		shift = - (np.mat([[ff_xx, ff_xy],[ff_xy, ff_yy]])**-1) * np.mat([[f_x],[f_y]])

		xm,ym = shift[0,0],shift[1,0]

		X = np.mat([[i],[j]])
		fx = cf[i][j] + np.mat([[f_x],[f_y]]).T *  X+  X.T*np.mat([[ff_xx, ff_xy],[ff_xy, ff_yy]])*X/2

		if (cf[i][j]+np.mat([[f_x],[f_y]]).T *  X/2) < 0.03 or fx < 0.03:
			continue
		else:
			refine_point.append((round(i+xm),round(j+ym)))

	return refine_point

def orientation(pl):
	ori_blured = cv2.GaussianBlur(pl, (5,5) , 4.5, 4.5)
	ori_gradx = cv2.Sobel(ori_blured , cv2.CV_32F , 1, 0, ksize=3,  borderType=cv2.BORDER_DEFAULT)
	ori_grady =  cv2.Sobel(ori_blured , cv2.CV_32F , 0, 1, ksize=3,  borderType=cv2.BORDER_DEFAULT)
	ulen = (ori_gradx **2 + ori_grady **2)**0.5 + 1e-8

	ori = np.stack([ori_gradx/ulen, ori_grady/ulen] ,axis=-1)

	return ori

def subsample(img,x,y,r=40):
	Lx ,Ly= max(0,x-r//2) , max(0,y-r//2)
	Rx,Ry = min(img.shape[0]-1,x+r//2) , min(img.shape[1]-1,y+r//2)

	patch = img[Lx:Rx,Ly:Ry]
	patch = patch[::5,::5]
	
	if patch.shape!=(8,8):
		patch = cv2.resize(img[Lx:Rx,Ly:Ry], (8, 8))
	return patch

def desciptors(pl , ori , x , y):
	cos ,sin = ori[x][y]

	theta = math.acos(cos)

	M = cv2.getRotationMatrix2D((y,x), theta, 1.0)

	rotate_img = pl
	rotate_img = cv2.warpAffine(pl,M,pl.shape[::-1])
	
	patch = subsample(rotate_img,x,y)
	patch = (patch -patch.mean())/(patch.std()+1e-8)
	return patch.flatten()

def msop(img_src , debug=True, debug_path='.' , anms_mul = 1):
	
	pl = img_src.copy()
	pl_all = []
	h , w = img_src.shape

	#generate all levels to pl_all
	for _ in range(4):
		pl_all.append(pl.copy())
		
		pl = cv2.GaussianBlur(pl , (3,3) , 1,1)
		pl = cv2.resize(pl , (pl.shape[1]//2,pl.shape[0]//2))


	#detector
	features=[]
	pos = []
	for i,pl in enumerate(pl_all):
		logging.info(f'level:{i}, pl size:{pl.shape}')
		Ix =  cv2.Sobel(pl, cv2.CV_32F , 1, 0, ksize=3,  borderType=cv2.BORDER_DEFAULT)
		Iy =  cv2.Sobel(pl, cv2.CV_32F , 0, 1, ksize=3,  borderType=cv2.BORDER_DEFAULT)
		a = cv2.GaussianBlur(Ix**2 , (3,3) , 1.5,1.5)
		b = cv2.GaussianBlur(Ix*Iy , (3,3) , 1.5,1.5)
		d = cv2.GaussianBlur(Iy**2 , (3,3) , 1.5,1.5)

		corner_func = (a*d-b*b)/(a+d+1e-8) # Fhm ,function of harris corner deteciton

		local_maxima =  cv2.dilate(corner_func , np.ones((3,3), np.uint8) )

		if debug:
			heatmapshow = cv2.normalize(local_maxima, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
			heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
			cv2.imwrite(f'gradient_heat{i}.jpg',cv2.resize(heatmapshow,(pl_all[0].shape[::-1])) )


		sup = non_maximal_sup(corner_func,local_maxima,anms = int(min(pl.shape)/anms_mul) ) #shape (f,x,y)
		sup = np.array(sup,dtype=int)
		logging.debug(f'{sup.shape,sup[:,1].max()},{sup[:,2].max()}')

		refined = sub_pixel_refinement(local_maxima , sup) # (x,y)
		

		ori = orientation(pl_all[0])
		

		

		if debug:
			draw = cv2.cvtColor(pl, cv2.COLOR_GRAY2BGR)
			for _,x,y in sup:
				cv2.circle(draw, (y,x), 1, color = (0, 0, 255))
			cv2.imwrite(f'keypoints{i}.jpg',cv2.resize(draw,(pl_all[0].shape[::-1])) )

		logging.info(f'refined key points number:{len(refined)}\n')
		
		for x,y in refined:
			if pl_all[0].shape[0] >int(x*(2**i)) >=0 and  pl_all[0].shape[1] >int(y*(2**i)) >=0:
				pos.append([int(x*(2**i)),int(y*(2**i))])
				features.append(desciptors(pl_all[0] , ori , int(x*(2**i)) , int(y*(2**i))) )

	
	return np.array(features),np.array(pos)


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
	featrues , pos = msop(cv2.imread(os.path.join(path,file_list[0]),cv2.IMREAD_GRAYSCALE) , debug=True)
	logging.info(f'feature array shape{featrues.shape}, position shape:{pos.shape}')
