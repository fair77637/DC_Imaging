import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt


def reshape(array, ConstPixelSpacing):
    """ Reshape array to have cubic voxels of size 1mm^3 """
    '''mind the a, b, c value is adjusted to make each '''

    width = array.shape[0]
    height = array.shape[1]
    depth = array.shape[2]
    a = int((width)*ConstPixelSpacing[0])
    b = int((height)*ConstPixelSpacing[1])
    c = int((depth)*ConstPixelSpacing[2])
    reshapedArray1 = np.zeros((a,height,depth))
    xp = np.linspace(0, (width-1)*ConstPixelSpacing[0], width) #adjust xp as [0,1ConstPixelSpacing,2CPS..
    x  = np.linspace(0, a-1, a)

    for j in range(height):
        for k in range(depth):
            reshapedArray1[:,j,k] = np.interp(x, xp, array[:,j,k])
    reshapedArray2 = np.zeros((a,b,depth))
    yp = np.linspace(0,(height-1)*ConstPixelSpacing[1],height)
    y = np.linspace(0,b-1,b)
    for j in range(a):
        for k in range(depth):
            reshapedArray2[j,:,k] = np.interp(y, yp, reshapedArray1[j,:,k])
    reshapedArray3 = np.zeros((a,b,c))
    zp = np.linspace(0,(depth-1)*ConstPixelSpacing[2],depth)
    z = np.linspace(0,c-1,c)
    for j in range(a):
        for k in range(b):
            reshapedArray3[j,k,:] = np.interp(z, zp, reshapedArray2[j,k,:])
    return reshapedArray3



def thresholdnp(array, lo, hi):
    thresholded1 = np.multiply(array, (array>lo).astype(int))
    thresholded2 = np.multiply(thresholded1, (array<hi).astype(int))
    return thresholded2


def displayMask(array,ConstPixelSpacing,x, y, z):
    # Display the orthogonal slices
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(array[x,:,:])
    ax2.imshow(array[:,y,:])
    ax3.imshow(array[:,:,z])

    # Additionally display crosshairs
    ax1.axhline(y * ConstPixelSpacing[1], lw=1)
    ax1.axvline(x * ConstPixelSpacing[0], lw=1)

    ax2.axhline(z * ConstPixelSpacing[2], lw=1)
    ax2.axvline(x * ConstPixelSpacing[0], lw=1)

    ax3.axhline(z * ConstPixelSpacing[2], lw=1)
    ax3.axvline(y * ConstPixelSpacing[1], lw=1)

    plt.show()
    
    
def distribution(xr,num_bins):
    # Get the number of points
    num_pts = len(xr.flatten())

    # Extract the minimum and maximum intensity values and calculate the number of  bins for the histogram
    lim_low = np.min(xr)
    lim_high = np.max(xr)
    # num_bins = (lim_high - lim_low + 1)

    plt.figure(figsize=(10, 4), dpi=100)
    plt.hist(xr, bins=num_bins, normed=True, range=(lim_low, lim_high), color='lightgray');
    plt.xlim([lim_low,lim_high]); # we limit the x-axis to the range of interest
    plt.show()

    print('Number of points ' + str(num_pts))

