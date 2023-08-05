import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def convolve(array, kernel, s, p):
    # kernel, input, and output sizes
    k = kernel.__len__()
    n = array.__len__()
    m = (n - k + s + 2 * p) // s

    # padding with zeroes
    if p > 0:
        pad = []
        for i in range(n):
            pad.insert(i, 0)
        for i in range(0, p):
            array.insert(i, pad)
        for i in range(n, n - p):
            array.insert(i, pad)
        for i in range(n + p):
            for j in range(0, p):
                array[i].insert(j, 0)
            for j in range(n, n - p):
                array[i].insert(j, 0)

    # adding elements to the output matrix using the formula
    output = []
    for i in range(m):
        row = []
        for j in range(m):
            x = 0
            for iK in range(k):
                for jK in range(k):
                    x = x + array[s * i + iK][s * j + jK] * kernel[iK][jK]
            row.insert(j, x)
        output.insert(i, row)
    return output


def printMatrix(conv):
    matrix = '\n'.join([''.join(['{:8}'.format(item) for item in row]) for row in conv])
    print(matrix)


# this method only works with input, kernel ,output sets of size 5, 3, 3
def prettyPrint(input, kernel, output):
    pad = [' ', ' ', ' ']
    kernel.insert(3, pad)
    output.insert(3, pad)
    kernel.insert(0, pad)
    output.insert(0, pad)

    times = [' ', ' ', '*', ' ', ' ']
    equal = [' ', ' ', '=', ' ', ' ']
    all = []
    for i in range(5):
        row = list(map(str, input[i]))
        row.extend(list(map(str, times[i])))
        row.extend(list(map(str, kernel[i])))
        row.extend(list(map(str, equal[i])))
        row.extend(list(map(str, output[i])))
        all.insert(i, row)
    printMatrix(all)


def heatmap(conv):
    fig = plt.imshow(conv, cmap='PiYG_r', interpolation='nearest')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.show()


def image_convolution(img,kernel):
        # Assuming a rectangular image
        n=img.__len__()
        # To simplify things
        k = kernel.__len__()
        tgt_size=n-k+1
        # 2D array of zeros
        convolved_img = np.zeros(shape=(tgt_size, tgt_size))

        # Iterate over the rows
        for i in range(tgt_size):
            # Iterate over the columns
            for j in range(tgt_size):
                # img[i, j] = individual pixel value
                # Get the current matrix
                mat = img[i:i + k, j:j + k]

                # Apply the convolution - element-wise multiplication and summation of the result
                # Store the result to i-th row and j-th column of our convolved_img array
                convolved_img[i, j] = np.sum(np.multiply(mat, kernel))

        return convolved_img


# testing the convolve function
array = [[1, 2, 3, 4, 5],
         [1, 3, 2, 3, 10],
         [3, 2, 1, 4, 5],
         [6, 1, 1, 2, 2],
         [3, 2, 1, 5, 4]
         ]
array = [[1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         [1, 1, 1, 1, 1],
         ]
kernel = [[1, 0, -1],
          [1, 0, -1],
          [1, 0, -1]]
kernel = [[1, 0, 1],
          [0, 1, 0],
          [1, 0, 1]]
kernel1 = [[-1, -1, -1],
           [-1, 8, -1],
           [-1, -1, -1]]
kernel2 = [[0, -1, 0],
           [-1, 8, -1],
           [0, -1, 0]]

stride = 1
padding = 0
#conv = convolve(array, kernel2, stride, padding)
conv = image_convolution(np.asarray(array), kernel2)
# Image.fromarray(np.uint8(conv)).show()
# prettyPrint(array, kernel, conv)
printMatrix(conv)

# image convolution
im = Image.open('heart.jpg')
rgb = np.array(im.convert('RGB'))
r = rgb[:, :, 0]
# Image.fromarray(np.uint8(r)).show()


conv = convolve(r, kernel2,1,0)

# choose the desired kernel and stride
# conv = convolve(r, kernel1, 1, 0)

Image.fromarray(np.uint8(conv)).show()
# show the convolution heatmap
heatmap(conv)

# show the convolution matrix
# printMatrix(conv)
