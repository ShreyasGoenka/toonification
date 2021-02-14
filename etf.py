import numpy as np
import cv2
import sys
import matplotlib.pylab as plt
import os
import pickle
# import lic.lic_internal as lic_internal

"""
    This function implements the separable edge tangent flow 
    for an input image.

    image: 2D numpy array with intensity values
    iters: number of times ETF is to be performed
    mu: size of the kernal
"""

def etf(image, mu, iters = 2):
    # gradients
    image = image/255
    g_x = np.array(cv2.Sobel(image, cv2.CV_64F,1,0,ksize=5))
    g_y = np.array(cv2.Sobel(image, cv2.CV_64F,0,1,ksize=5))
    g_mag = np.sqrt(g_x*g_x + g_y*g_y)
    g_norm = g_mag/np.max(g_mag) # normalized magnitudes

    # tangents (first iteration)
    t_x = -1*g_y
    t_y = g_x
    t = np.stack([t_x, t_y], axis=2)
    t_mag = np.sqrt(t_x*t_x + t_y*t_y)
    t_mag[t_mag == 0] = 1
    t_norm = t/np.stack([t_mag, t_mag], axis=2)
    size = t.shape
    print(size)

    for iter in range(iters):

        t_temp = np.zeros(size)

        for i in range(size[0]):
            if (i%25 == 0):
                print(i//25)
            for j in range(size[1]):
                for x in range(max(0, i-mu), min(i+mu+1, size[0])):
                    for y in range(max(0, j-mu), min(size[1], j+mu+1)):
                        # if np.linalg.norm(np.array([i,j])-np.array([x,y])) <= mu:
                        #     ws = 1
                        # else:
                        #     ws = 0
                        wm = ((g_norm[x][y]-g_norm[i][j]+1)/2)
                        # wm = (np.tanh(g_norm[x][y]-g_norm[i][j]+1)/2)
                        # wm = (np.tanh(g_mag[x][y]-g_mag[i][j]+1)/2)
                        wd = np.dot(t[i][j], t[x][y])
                        t_temp[i][j] += t[i][j]*wm*wd


        t = t_temp
        t_mag = np.linalg.norm(t, axis=2)
        t_mag[t_mag == 0] = 1
        t = t/np.stack([t_mag, t_mag], axis=2)

    return t

if __name__ == "__main__":
    input_path = sys.argv[1]
    image = cv2.imread(input_path, 1)
    image = cv2.resize(image, (256, 256))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    size = image_gray.shape

    t = None

    name, ext = os.path.splitext(input_path)
    name = name.split('/')[-1]

    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    if os.path.isfile('tmp/' + name + '_etfpickle'):
        with open('tmp/' + name + '_etf.pickle', 'rb') as f:
            t = pickle.load(f)
    else:
        t = etf(image_gray, 3, iters=2)

        with open('tmp/' + name + '_etf.pickle', 'wb') as f:
            pickle.dump(t,f)

    texture = np.random.rand(size[0],size[1]).astype(np.float32)
    kernellen=31
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)

    for h in range(size[0]):
        for w in range(size[1]):
            if(t[h][w][0] == 0):
                t[h][w][0] = np.random.rand()*0.01
            if(t[h][w][1] == 0):
                t[h][w][1] = np.random.rand()*0.01

    tnorm = np.linalg.norm(t, axis=2)
    np.place(tnorm, tnorm == 0, [1])
    t = np.divide(t, np.stack([tnorm, tnorm], axis=2))

    etf_lic = lic_internal.line_integral_convolution(t.astype(np.float32), texture, kernel)

    dpi = 100
    plt.bone()
    plt.clf()
    plt.axis('off')
    plt.figimage(etf_lic)
    plt.gcf().set_size_inches((size[0]/float(dpi),size[1]/float(dpi)))
    plt.savefig("output/" + name + "_flow.png", dpi=dpi)

