import numpy as np

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


def fbl(image, flow, iters = 1, at = 5, ap = 5, sig_space = 3, sig_it = 100, sig_ip = 25):
    size = image.shape
    # px_dirs = np.array([(0,1), (-1,1), (-1,0), (-1,-1),(0,-1), (1,-1), (1,0), (1,1)])
    # px_dirs_norm = np.linalg.norm(px_dirs, axis=1)
    # unit_dirs = px_dirs/np.stack([px_dirs_norm,px_dirs_norm], axis=1)

    image_t = np.zeros(size)
    image_p = np.zeros(size)

    for iter in range(iters):
        for i in range(size[0]):
            if i%25 == 0:
                print(i//25)
            for j in range(size[1]):
                total_wt = 0
                
                angle_t = angle(flow[i][j][0],flow[i][j][1])
                pix = neibhour(angle_t*180/np.pi)
                dx = pix[0]
                dy = pix[1]
                
                for k in range(-at, at+1):
                    if i+dx*k < 0  or i+dx*k >= size[0] or j+dy*k < 0 or j+dy*k >= size[1]:
                        continue
                    ws = gaussian(k, sig_space)
                    I = np.dot(image[i][j] - image[i+dx*k][j+dy*k], image[i][j] - image[i+dx*k][j+dy*k])
                    wi = gaussian(I, sig_it)
                    total_wt += wi*ws
                    image_t[i][j] += image[i+k*dx][j+k*dy]*ws*wi
                
                if (total_wt != 0):
                    image_t[i][j] /= total_wt

        for i in range(size[0]):
            for j in range(size[1]):
                total_wt = 0

                angle_p = (angle(flow[i][j][0],flow[i][j][1]) + np.pi/2)
                if angle_p > 2*np.pi:
                    angle_p -= 2*np.pi

                pix = neibhour(angle_p*180/np.pi)
                dx = pix[0]
                dy = pix[1]

                for k in range(-ap, ap+1):
                    if i+dx*k < 0 or i+dx*k >= size[0] or j+dy*k < 0 or j+dy*k >= size[1]:
                        continue
                    ws = gaussian(k, sig_space)
                    I = np.dot(image_t[i][j] - image_t[i+dx*k][j+dy*k], image_t[i][j] - image_t[i+dx*k][j+dy*k])
                    wi = gaussian(I, sig_ip)
                    total_wt += wi*ws
                    image_p[i][j] += image_t[i+dx*k][j+dy*k]*wi*ws
                
                if (total_wt != 0):
                    image_p[i][j] /= total_wt

        image = image_p

    return image
        
        

