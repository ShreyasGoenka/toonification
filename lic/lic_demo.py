import numpy as np
import pylab as plt
import pickle

import lic_internal

dpi = 100
size = 700
video = False

vortex_spacing = 0.5
extra_factor = 2.

a = np.array([1,0])*vortex_spacing
b = np.array([np.cos(np.pi/3),np.sin(np.pi/3)])*vortex_spacing
rnv = int(2*extra_factor/vortex_spacing)
vortices = [n*a+m*b for n in range(-rnv,rnv) for m in range(-rnv,rnv)]
vortices = [(x,y) for (x,y) in vortices if -extra_factor<x<extra_factor and -extra_factor<y<extra_factor]


xs = np.linspace(-1,1,size).astype(np.float32)[None,:]
ys = np.linspace(-1,1,size).astype(np.float32)[:,None]

vectors = np.zeros((size,size,2),dtype=np.float32)
for (x,y) in vortices:
    rsq = (xs-x)**2+(ys-y)**2
    vectors[...,0] +=  (ys-y)/rsq
    vectors[...,1] += -(xs-x)/rsq

for i in range(vectors.shape[0]):
    for j in range(vectors.shape[1]):
        if(vectors[i][j][0] == 0 or vectors[i][j][1] == 0):
            print("HAHA")

with open('tmp/baboon_flow.pickle', 'rb') as f:
    vectors = pickle.load(f).astype(np.float32)
size = vectors.shape[0]
texture = np.random.rand(size,size).astype(np.float32)

plt.bone()
frame=0

if video:
    kernellen = 31
    for t in np.linspace(0,1,16*5):
        kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)*(1+np.sin(2*np.pi*5*(np.arange(kernellen)/float(kernellen)+t)))

        kernel = kernel.astype(np.float32)

        image = lic_internal.line_integral_convolution(vectors, texture, kernel)

        plt.clf()
        plt.axis('off')
        plt.figimage(image)
        plt.gcf().set_size_inches((size/float(dpi),size/float(dpi)))
        plt.savefig("flow-%04d.png"%frame,dpi=dpi)
        frame += 1
else:
    kernellen=31
    kernel = np.sin(np.arange(kernellen)*np.pi/kernellen)
    kernel = kernel.astype(np.float32)


    image = lic_internal.line_integral_convolution(vectors, texture, kernel)
 
    plt.clf()
    plt.axis('off')
    plt.figimage(image)
    plt.gcf().set_size_inches((size/float(dpi),size/float(dpi)))
    plt.savefig("flow-image.png",dpi=dpi)


