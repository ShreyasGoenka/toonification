import numpy as np
import cv2
from etf import etf
from fdog import fdog
from fbl import fbl
import os
import time
import pickle
import sys

def luminisece_quantization(image):

    imagel = cv2.cvtColor(image.astype(np.uint8),cv2.COLOR_BGR2LAB)
    size = image.shape

    bins = 10
    for i in range(size[0]):
        for j in range(size[1]):
            imagel[i,j,:] = bins*(imagel[i,j,:]//bins)
        
    result = cv2.cvtColor(imagel, cv2.COLOR_LAB2BGR)
    return result


def NPR_image(input_path):

    image = cv2.imread(input_path, 1)
    print(image.shape)
    image = cv2.resize(image, (0, 0), fx = 0.1, fy = 0.15)
    image = cv2.GaussianBlur(image, (5,5), 0)
    # image = cv2.resize(image, (128, 128))
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    name, ext = os.path.splitext(input_path)
    print(name)
    name = name.split('/')[-1]

    if not os.path.isdir('tmp'):
        os.mkdir('tmp')

    if os.path.isfile('tmp/' + name + '_flow.pickle'):
        with open('tmp/' + name + '_flow.pickle', 'rb') as f:
            flow = pickle.load(f)
    else:
        flow_start = time.time()
        flow = etf(image_gray, 5, iters=1)
        flow_stop = time.time()
        print("etf time: ", flow_stop-flow_start)

        with open('tmp/' + name + '_flow.pickle', 'wb') as f:
            pickle.dump(flow,f)

    if os.path.isfile('tmp/' + name + '_edges.pickle'):
        with open('tmp/' + name + '_edges.pickle', 'rb') as f:
            edges = pickle.load(f)

    else:
        fdog_start = time.time()
        edges = fdog(image_gray, flow, iters=1)
        fdog_stop = time.time()

        print ('fdog time:', fdog_stop - fdog_start)

        with open('tmp/' + name + '_edges.pickle', 'wb') as file:
            pickle.dump(edges,file)

    if os.path.isfile('tmp/' + name + '_bl.pickle'):
        with open('tmp/' + name + '_bl.pickle', 'rb') as f:
            image_bl = pickle.load(f)
    else:
        fbl_start = time.time()
        image_bl = fbl(image, flow, iters=1)
        fbl_stop = time.time()

        print('fbl time: ', fbl_stop - fbl_start)

        with open('tmp/' + name + '_bl.pickle', 'wb') as f:
            pickle.dump(image_bl, f)

    edgesl = cv2.Laplacian(image_gray, cv2.CV_64F)
    edgesl = 255-edgesl

    image_lbl = luminisece_quantization(image_bl)
    
    cartoon_laplace = np.minimum(np.stack([edgesl,edgesl,edgesl], axis = 2), image_lbl)
    cartoon = np.minimum(np.stack([edges,edges,edges], axis = 2), image_lbl)

    cv2.imwrite("output/" + name + "_edges.png", edges)
    cv2.imwrite("output/" + name + "_edges_laplace.png", edgesl)
    cv2.imwrite("output/" + name + "_fbl.png", image_bl)
    cv2.imwrite("output/" + name + "_quantized_fbl.png", image_lbl)
    cv2.imwrite("output/" + name + "_cartoon_laplace.png", cartoon_laplace)
    cv2.imwrite("output/" + name + "_cartoon.png", cartoon)

if __name__ == "__main__":
    input_path = sys.argv[1]
    NPR_image(input_path)
