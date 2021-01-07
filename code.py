from PIL import Image
import pywt
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imgPath = 'C:/Users/Hp/Desktop/input.png'

print('Path Imported')

I = Image.open(imgPath)

print('Image Imported')

b = I.convert('YCbCr')

print('Image Converted to YCbCr')

y, cb, cr = b.split()

[one, two, three] = pywt.swtn(y,'db1',3)

print('SWT applied Successfully')

[LL, LH, HL, HH] = three['aa'], three['ad'], three['da'], three['dd']

row, column = LL.shape

print('Got LL(approximation) Band')

C = []

#Block = np.zeros(8*8).reshape(8,8)
#for i in range (1,(row-7)*(column-7)+1):
#    C.append(Block)

Counter=0

for i in range (0,row-7):
    for j in range (0,column-7):
        Counter = Counter+1
        Copy = LL[i:i+8,j:j+8]
        C.append(Copy)

print('Overlapping block generated ,','Total Blocks = ',Counter)

D = []

Counter = 0
for i in range (0,row-7):   #(0,row-7)
    for j in range (0,column-7):   #(0,column-7)
        Temp = C[Counter]
        
        #Creating first feature
        f1 = 0
        padX = 0
        padY = 0
        for x in range(0,4):
            for y in range(padY,8-padY):
                f1 = f1 + Temp[x][y]
            padY = padY + 1
        #ended first feature
        
        #Creating second feature
        f2 = 0
        padX = 0
        padY = 0
        for x in range(7,3,-1):
            for y in range(padY,8-padY):
                f2 = f2 + Temp[x][y]
            padY = padY + 1
        #ended second feature
        
        #Creating third feature
        f3 = 0
        padX = 0
        padY = 0
        for y in range(0,4):
            for x in range(padX,8-padX):
                f3 = f3 + Temp[x][y]
            padX = padX + 1
        #ended third feature
        
        #Creating fourth feature
        f4 = 0
        padX = 0
        padY = 0
        for y in range(7,3,-1):
            for x in range(padX,8-padX):
                f4 = f4 + Temp[x][y]
            padX = padX + 1
        #ended fourth feature
        
        #normalizing features
        f1 = f1/20
        f2 = f2/20
        f3 = f3/20
        f4 = f4/20
        #ended normalization
        
        #finding co-ordinates of Block
        if((Counter+1)%(row-7)==0):
            x = (Counter+1)/(row-7)
            y = row-7
        else:
            x = ((Counter+1)//(row-7))+1
            y = (Counter+1)%(row-7)
        #ended finding co-ordinated
        
        #feature Vector creating
        featureVector = np.array([f1, f2, f3, f4, x, y])
        D.append(featureVector)
        #appended feature vector
        
        #print(featureVector)
        Counter = Counter+1
        
print('Feature Vector Generated')

#Bubble sort For lexicographic Sorting
def bubbleSort(C):
    n=len(C)
    k=len(C[0])
    for i in range(n):
        for j in range(0,n-1-i):
            for z in range(0,k):
                if(C[j][z]<C[j+1][z]):
                    break
                if(C[j][z]>C[j+1][z]):
                    C[j],C[j+1] = C[j+1],C[j]
                    break
                
#Quick sort For lexicographic Sorting
def check(a,b):
    k=len(a)

    for z in range(0,k):
        if (a[z] < b[z]):
            return True
        if (a[z] > b[z]):
            return False
    
    return False
        
def partition(a, L, H):
    i = (L-1)
    pivot = a[H]
    
    for j in range(L,H):
        if ( check(a[j], pivot) ):
            i = i + 1
            a[i], a[j] = a[j], a[i]
    
    a[i+1], a[H] = a[H], a[i+1]
    return ( i+1 )

def quickSort(a,L,H):
    if L < H:
        pi = partition(a,L,H)
        
        quickSort(a,L,pi-1)
        quickSort(a,pi+1,H)


#Applying lexicographic Sorting on feature vectors (Stored in D)
quickSort(D,0,len(D)-1)

print('Feature Vectors Sorted')

Counter = 0
Similar = []

for i in range(0,255015):
    P = D[i]
    for j in range(i+1,i+11):
        Q = D[j]
        
        x = P[4] - Q[4]
        y = P[5] - Q[5]
        diff1 = math.sqrt(x*x + y*y)
        
        d1 = P[0] - Q[0]
        d2 = P[1] - Q[1]
        d3 = P[2] - Q[2]
        d4 = P[3] - Q[3]
        diff2 = math.sqrt(d1*d1 + d2*d2 + d3*d3 + d4*d4)
        
        if ( diff2 < 0.0015 and diff1 > 40):
            #x1, y1, x2, y2
            Mathched_Block = np.array([P[4], P[5], Q[4], Q[5]])
            Similar.append(Mathched_Block)
            Counter = Counter + 1
        
print('Got Similar Blocks ,',' Total Blocks = ',Counter)

#convert to gray from rgb
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

ResImg = mpimg.imread(imgPath)
Mask = rgb2gray(ResImg)
#Mask = I.convert('LA')

for i in range(0,Counter):
    Block = Similar[i]
    x1 = int(Block[0])
    y1 = int(Block[1])
    x2 = int(Block[2])
    y2 = int(Block[3])
    
    for x in range(x1,x1+8):
        for y in range(y1,y1+8):
            Mask[x][y] = 255
    
    for x in range(x2,x2+8):
        for y in range(y2,y2+8):
            Mask[x][y] = 255

for i in range(0,row):
    for j in range(0,column):
        if Mask[i][j] < 255:
            Mask[i][j] = 0

#Showing the output result
titles=['Original','RDFD Result']

plt.subplot(1,2,1)
plt.imshow(I,cmap='gray')
plt.title(titles[0])
plt.xticks([])
plt.yticks([])
plt.show(block=True)

plt.subplot(1,2,2)
plt.imshow(Mask,cmap='gray')
plt.title(titles[1])
plt.xticks([])
plt.yticks([])
plt.show(block=True)

print('End')


