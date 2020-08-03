
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import os
import sys

sys.setrecursionlimit(2 * (10 ** 9))


def InMatrice(max_size1, max_size2, x, y):
    if x<0 or y<0 or x>=max_size1 or y>=max_size2:
        return False
    return True

def Fill(image, edges, x, y, pixeli_schimbati):
    #Algoritmul care umple spatiile din Canny Edge Detected Picture cu culori
    if pixeli_schimbati % 1000 == 0:
        print("pixeli schimbati = ", pixeli_schimbati)
    image[x][y] = (0,165,255)#un portocaliu
    edges[x][y] = 255


    if InMatrice(image.shape[0], image.shape[1], x, y-1) and edges[x][y-1] != 255:
        Fill(image, edges, x, y-1, pixeli_schimbati+1)
    if InMatrice(image.shape[0], image.shape[1], x, y+1) and edges[x][y+1] != 255:
        Fill(image, edges, x, y+1, pixeli_schimbati+1)
    if InMatrice(image.shape[0], image.shape[1], x-1, y) and edges[x-1][y] != 255:
        Fill(image, edges, x-1, y, pixeli_schimbati+1)
    if InMatrice(image.shape[0], image.shape[1], x+1, y) and edges[x+1][y] != 255:
        Fill(image, edges, x+1, y, pixeli_schimbati+1)


    if pixeli_schimbati==0:
        cv.imwrite("../data/Umplere.jpeg",image)
        cv.imwrite("../data/UmplereGri.jpeg",edges)

def Fill_smecher(image, edges, pixel_size, x, y, pixeli_schimbati):
    #Acest fill nu umple cate un pixel la un moment dat, ci cate un patrat de pixel_size*pixel_size pixeli
    #Numim acest patrat de pixeli "macropixel"
    #Consideram ca parametrii x si y sunt coltul din stanga-sus al patratului ce urmeaza a fi umplut la pasul curent

    #Umplem cu portocaliu macropixelul curent, despre care stim teoretic sigur ca nu contine obstacole
    suma_cul = (0,0,0)
    nr_px = 0
    for i in range(x, x+pixel_size):
        for j in range(y, y+pixel_size):
            suma_cul+=image[i][j]
            nr_px+=1
            image[i][j] = (0, 165, 255)  # un portocaliu
            edges[i][j] = 255

    dx = [ 0, 0,pixel_size,-1*(pixel_size)]
    dy = [pixel_size,-1*(pixel_size), 0, 0]

    for k in range(4):#parcurg toti cei 4 vecini ai macropixelului curent
        x2 = x + dx[k]
        y2 = y + dy[k]
        if InMatrice(image.shape[0], image.shape[1], x2, y2) and InMatrice(image.shape[0], image.shape[1], x2+pixel_size-1, y2+pixel_size-1):
            #Inseamna ca vecinul curent se afla complet in matrice
            #Cautam acum daca avem loc liber
            obstacol = False
            for i in range(x2, x2 + pixel_size):
                for j in range(y2, y2 + pixel_size):
                    if edges[i][j]==255:
                        obstacol = True
                        break
            if obstacol==False:
                rez = Fill_smecher(image, edges,pixel_size, x2, y2, pixeli_schimbati+(pixel_size**2))
                suma_cul+=rez[0]
                nr_px+=rez[1]

    if pixeli_schimbati==0:
        print("Heian nidan!")
        cv.imwrite("../data/Umplere_smechera.jpeg", image)
        cv.imwrite("../data/UmplereGri_smechera.jpeg", edges)
    return suma_cul, nr_px

def dist_euclidiana_3d(cul1, cul2):
    return ((cul1[0]-cul2[0])**2  +  (cul1[1]-cul2[1])**2  +  (cul1[2]-cul2[2])**2)**(1/2)
    #distanta Minkowski penntru p=60: return (  (cul1[0] - cul2[0]) ** 60 + (cul1[1] - cul2[1]) ** 60 + (cul1[2] - cul2[2]) ** 6  ) ** (1 / 60)

def IdentifyColor(cul_medie):
    print("********O NOUA APELARE PENTRU IDENTIFYCOLOR********")
    print("am primit ca parametru culoarea:", cul_medie)
    alb = [
        'alb',
        [255, 255, 255],  # culoarea "pura"
        [223.76612903225808, 232.7459677419355, 237.11290322580646],
        [184.4, 183.55, 188.0],#un gri
        [235.70833333333334, 241.56944444444446, 246.09722222222223],#un alb spre gri
        [244.875, 254.56944444444446, 252.58333333333334]
    ]
    galben = [
        'galben',
        [0, 255, 255],  # culoarea "pura"
        [240, 230, 140], #un kaki
        [110.125, 231.5, 248.625],#galben obtinut experimental
        [112.03703703703704, 254.4814814814815, 252.74074074074073],
        [131.97222222222223, 253.875, 253.26388888888889],
        [67,246, 240]
    ]
    verde = [
        'verde',
        [0, 255, 0],#verde pur
        [62.5855795148248, 162.13544474393532, 30.851078167115904],  # verde obtinut anterior in practica
        [115.25, 191.44444444444446, 99.05555555555556],
        [91.61309523809524, 182.625, 77.07142857142857],
        [160.6, 236.05, 122.95],
        [142.44444444444446, 251.83333333333334, 148.22222222222223],
        [160.66666666666666, 234.55555555555554, 131.0]
    ]
    albastru = [
        'albastru',
        [255, 0, 0],#albastru pur
        [235, 105, 18],
        [184.84426229508196, 147.14344262295083, 26.065573770491802],
        [255,191,0],
        [218.41666666666666, 176.25, 77.08333333333333],
        [206.61111111111111, 141.5, 57.0],
        [198.0, 142.5, 44.0],
        [206.11111111111111, 139.66666666666666, 47.77777777777778],
        [226.66666666666666, 181.0, 55.916666666666664]
    ]
    rosu = [
        'rosu',
        [0, 0, 255],
        [220, 20, 60],#crimson
        [14.681818181818182, 96.43181818181819, 241.77272727272728],
        [41.6875, 82.35416666666667, 234.24166666666667],
        [42.27205882352941, 72.76470588235294, 215.3014705882353],
        [13.42142857142857, 97.84285714285714, 247.36428571428573],
        [37.75, 29.75, 182.5],
        [21.333333333333332, 22.305555555555557, 203.69444444444446],
        [42.111111111111114, 3.4444444444444446, 221.0]
    ]
    portocaliu = [
        'portocaliu',
        [0, 165, 255],#orange
        [0,140,255],#dark orange
        [36.384415584415585, 160.93246753246754, 179.16688311688313],  # portocaliu obtinut experimental
        [25.055555555555557, 141.94444444444446, 234.25],
        [4.051724137931035, 165.7155172413793, 253.18965517241378],
        [41.5, 122.18125, 240.8375],#pana sa o pun pe asta detecta un portocaliu ca fiind rosu
        [45.81018518518518, 126.07870370370371, 238.75462962962962],
        [61.76923076923077, 184.69230769230768, 238.23076923076923],
        [44.111111111111114, 186.0, 244.44444444444446]
    ]
    culori = [alb, galben, verde, albastru, rosu, portocaliu]

    k_nearest_neighbours = []#initially, empty list
    K = 5
    ind = 0
    ind_cul = 0
    for ind_cul in range (len(culori)):
        culoare = culori[ind_cul]
        for ind_nuanta in range(1,len(culoare)):
            nuanta = culoare[ind_nuanta]
            dist = dist_euclidiana_3d(nuanta, cul_medie)
            #print("DEKI ind_cul = ", ind_cul, " iar ind_nuanta = ", ind_nuanta)
            #print("\nculoare = ", culoare)
            #print("culori[ind_cul]) = ", culori[ind_cul])
            #print("\nce naiba, dist = ", dist, " si cul_medie = ", cul_medie, ", iar nuanta = ", nuanta)
            #print("(pas intermediar, la inceput de for)nuanta = ", nuanta)
            #print("(pas intermediar)dist = ", dist)
            #print("(pas intermediar)knn este acum:")
            #for foo in k_nearest_neighbours:
            #    print(foo, " adica ", culori[foo[0]][foo[1]])
            #print("\n")
            if len(k_nearest_neighbours)==0:
                k_nearest_neighbours.append([ind_cul, ind_nuanta, dist])
            else:
                if len(k_nearest_neighbours) < K:
                    k_nearest_neighbours.append([0,0,1000000000])#un element fictiv
                if k_nearest_neighbours[-1][2] > dist:
                    k_nearest_neighbours[-1] = [ind_cul, ind_nuanta, dist]
                    indice = len(k_nearest_neighbours)-1
                    while indice>=1 and k_nearest_neighbours[indice-1][2] > k_nearest_neighbours[indice][2]:#e mai mare decat ce am gasit acum
                        k_nearest_neighbours[indice-1], k_nearest_neighbours[indice] = k_nearest_neighbours[indice], k_nearest_neighbours[indice-1]
                        indice-=1
            #print("(pas intermediar, la sfarsit de for)knn este acum:")
            #for foo in k_nearest_neighbours:
            #    print(foo, " adica ", culori[foo[0]][foo[1]])
            #print("\n\n")
    print("\n(LA SFARSITUL FUNCTIEI)Knn este acum:")
    for foo in k_nearest_neighbours:
        print(foo, " adica ", culori[foo[0]][foo[1]], ", adica ", culori[foo[0]][0])
    print("\n\n")


    nume_culori = ['verde', 'albastru', 'rosu', 'portocaliu', 'alb', 'galben']
    return nume_culori[0]
base_dir = '../data'
dir_pos_examples = os.path.join(base_dir, 'exemplePozitive')

'''_____________________________________________________________________________________________'''
culori_detectate = []
'''_____________________________________________________________________________________________'''


img = cv.imread('../data/exempleReale/pentruExtrasCuloriWhatsApp Image 2020-06-11 at 17.14.16.jpeg.jpeg',0)#alb-negru
img_color = cv.imread('../data/exempleReale/pentruExtrasCuloriWhatsApp Image 2020-06-11 at 17.14.16.jpeg.jpeg')#color

#img = cv.resize(img, (70, 70))
#img_color = cv.resize(img_color, (70, 70))

edges = cv.Canny(img,5,15)

print("type img[0][0] = ", type(img[0][0]))
print("img[0][0] = ", img[0][0])
print("shape img = ", img.shape)
cv.imwrite("../data/Muchii.jpeg",edges)
#Fill(img_color,edges,int(img.shape[0]/2), int(img.shape[1]/2), 0)
x = img.shape[0]
y = img.shape[1]
mijlx = int(img.shape[0]/2)
mijly = int(img.shape[1]/2)


culoare_medie, nr_px = Fill_smecher(img_color, edges, 3, mijlx, mijly,0)#centru
print('culoare_medie = ', culoare_medie)
print('nr_px = ', nr_px)

cl_md = [0.0, 0.0, 0.0]
cl_md[0] = float(culoare_medie[0]*1.0 / nr_px)
cl_md[1] = float(culoare_medie[1]*1.0 / nr_px)
cl_md[2] = float(culoare_medie[2]*1.0 / nr_px)
print("dupa impartire: ", cl_md)
print("culoare sticker centru: ", IdentifyColor(cl_md))

culoare_medie, nr_px = Fill_smecher(img_color, edges, 3, mijlx, mijly//2,0)#stanga
cl_md = [0.0, 0.0, 0.0]
cl_md[0] = float(culoare_medie[0]*1.0 / nr_px)
cl_md[1] = float(culoare_medie[1]*1.0 / nr_px)
cl_md[2] = float(culoare_medie[2]*1.0 / nr_px)
print("dupa impartire: ", cl_md)
print("culoare sticker stanga: ", IdentifyColor(cl_md))


culoare_medie, nr_px = Fill_smecher(img_color, edges, 3, mijlx, mijly + x//3,0)#dreapta
cl_md = [0.0, 0.0, 0.0]
cl_md[0] = float(culoare_medie[0]*1.0 / nr_px)
cl_md[1] = float(culoare_medie[1]*1.0 / nr_px)
cl_md[2] = float(culoare_medie[2]*1.0 / nr_px)
print("dupa impartire: ", cl_md)
print("culoare sticker dreapta: ", IdentifyColor(cl_md))



'''Fill_smecher(img_color, edges, 2, mijlx, mijly + x//4,0)#dreapta
Fill_smecher(img_color, edges, 2, mijlx//2, mijly//2,0)#stanga-sus
Fill_smecher(img_color, edges, 2, mijlx//2, mijly,0)#sus
Fill_smecher(img_color, edges, 2, mijlx//2, int(y * 0.75),0)#dreapta-sus
Fill_smecher(img_color, edges, 2, int(x*0.75), y//4,0)#stanga-jos
Fill_smecher(img_color, edges, 2, int(x*0.75), mijly,0)#jos
Fill_smecher(img_color, edges, 2, int(x*0.75), int(y*0.75),0)#stanga-jos
'''

#cv.imshow("edges",edges)


'''
plt.subplot(121),plt.imshow(img,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges,cmap = 'gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
'''