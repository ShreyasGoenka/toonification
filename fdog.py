import numpy as np
import cv2


def gaussian(t, sigma):
    return (1/np.sqrt(2*np.pi*sigma))*np.exp(-t**2/(2*sigma**2))

def neibhour(angle):

    px_dirs = [(0,1), (-1,1), (-1,0), (-1,-1), (0,-1), (1,-1), (1,0), (1,1)]
    angles = [0, 45, 90, 135, 180, 225, 270, 315]
    min_angle = 360
    index = 0
    for i in range(8):
        if abs(angles[i] - angle) < min_angle:
            min_angle = abs(angles[i] - angle)
            index = i

    return px_dirs[index]

def angle(x,y):
    if x == 0:
        return np.pi/2
    theta = np.arctan(abs(y)/abs(x))
    
    if x >= 0 and y >= 0:
        return 2*np.pi - theta
    if x >= 0 and y < 0:
        return theta
    if x < 0 and y >= 0:
        return np.pi + theta
    if x < 0 and y < 0:
        return np.pi - theta

def fdog_iter(image,
flow, 
sig_m = 3, 
sig_c = 1, 
tau = 0.7,
alpha = 3,
beta = 3,
p = 0.9761):

    sig_s  = 1.05*sig_c
    size = image.shape
    # px_dirs = np.array([(0,1), (-1,1), (-1,0), (-1,-1),(0,-1), (1,-1), (1,0), (1,1)])

    Hg = np.zeros(size)
    for i in range(size[0]):
        if (i%25 == 0):
            print(i//25)
        for j in range(size[1]):

            angle_p = (angle(flow[i][j][0],flow[i][j][1]) + np.pi/2)
            total_wt = 0
            if angle_p > 2*np.pi:
                angle_p -= 2*np.pi

            pix = neibhour(angle_p*180/np.pi)
            dx = pix[0]
            dy = pix[1]
            
            for k in range(-alpha, alpha+1):
                if i+dx*k < 0 or i+dx*k >= size[0] or j+dy*k < 0 or j+dy*k >= size[1]:
                    continue
                total_wt += abs(gaussian(k, sig_c) - p*gaussian(k, sig_s))
                Hg[i][j] += image[i+dx*k][j+dy*k]*(gaussian(k, sig_c) - p*gaussian(k, sig_s))

            if total_wt != 0:
                Hg[i][j] /= total_wt
        
    He = np.zeros(size)

    for i in range(size[0]):
        if (i%25 == 0):
            print(i//25)
        for j in range(size[1]):
            
            total_wt = 0
            He[i][j] += Hg[i][j]*gaussian(0, sig_m)
            total_wt += gaussian(0, sig_m)
            li = i
            lj = j

            for l in range(1, beta+1):

                angle_t = angle(flow[i][j][0],flow[i][j][1])
                pix = neibhour(angle_t*180/np.pi)
                dx = pix[0]
                dy = pix[1]

                li += dx
                lj += dy
                if li < 0 or li >= size[0]:
                    break
                if lj < 0 or lj >= size[1]:
                    break
                He[i][j] += Hg[li][lj]*gaussian(l, sig_m)
                total_wt += gaussian(l, sig_m)

            li = i
            lj = j

            for l in range(-beta, 0):

                angle_t = angle(flow[i][j][0],flow[i][j][1])
                pix = neibhour(angle_t*180/np.pi)
                dx = pix[0]
                dy = pix[1]
                
                li -= dx
                lj -= dy
                if li < 0 or li >= size[0]:
                    break
                if lj < 0 or lj >= size[1]:
                    break
                He[i][j] += Hg[li][lj]*gaussian(l, sig_m)
                total_wt += gaussian(l, sig_m)

            if total_wt != 0:
                He[i][j] /= total_wt
    

    He[(1+np.tanh(He) < tau)*(He < 0)] = 0
    He[He != 0] = 255
    return He

def fdog(image,
flow, 
iters = 1, 
sig_m = 3, 
sig_c = 1, 
tau = 0.3,
alpha = 3,
beta = 3,
p = 0.9761):

    input_image = image
    edges = np.zeros(image.shape)

    for iter in range(iters):
        edges = fdog_iter(input_image, flow, sig_m, sig_c, tau, alpha, beta, p)
        input_image = np.minimum(input_image, edges)
    
    return edges