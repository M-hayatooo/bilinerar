import math
import numpy as np
from PIL.Image import open, fromarray
# PIL import Image, fromarray


#画像の読み込みはライブラリを用いた
#im = Image.open('data/orijinal.jpg')
#print(im.format, im.height, im.mode)

def Nh(gazo, kakudairitu, kaitenn): # NearestHokann
    if not isinstance(kakudairitu, tuple):
            kakudairitu = (kakudairitu, kakudairitu)
    kakudairitu = (float(kakudairitu[0]), float(kakudairitu[1]))

    #グレースケール or カラー画像 の場合分け
    if gazo.shape[2] == None:
        size_store = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]))
    else:
        size_store = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]), gazo.shape[2])

    # 変更前の画像の 縦、横 のサイズ    #hi, wi = gazo.shape[0], gazo.shape[1]
    # 目的の画像の 縦、横 のサイズ    #g_hi, g_wi = gazo.shape[0]*kakudairitu, gazo.shape[1]*kakudairitu
    #ghi = int(g_hi), gwi = int(g_wi)
    #shape = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]))
    g = np.zeros(size_store, dtype=np.uint8)
    
# g.shape[0] == g_hi

    for x in range(g.shape[0]):
        for y in range(0, g.shape[1]):
            xi, yi = int(round(x/kakudairitu)), int(round(y/kakudairitu))
            # 存在しない座標の処理
            if xi > g.shape[1] -1: xi = g.shape[1] -1
            if yi > g.shape[0] -1: yi = g.shape[0] -1

            g[y, x] = gazo[yi, xi]

    if kaitenn == 0:
        exit
    else :
        for i in range(g.shape[1]):
            y = g.shape[0]/2 - i
            for j in range(g.shape[1]):
                x = j - g.shape[1]/2
                xr =  x*math.cos(math.radians(kaitenn)) + y*math.sin(math.radians(kaitenn))
                yr = -x*math.sin(math.radians(kaitenn)) + y*math.cos(math.radians(kaitenn))
                #最近傍補間
                jx = int(xr + g.shape[1]/2 + 0.5)
                iy = int(g.shape[1]/2 - yr + 0.5)
                
                # 存在しない座標の処理
                if (0 <= iy & iy < g.shape[0]) & (0 <= jx & jx < g.shape[1]):
                    g[i, j] = gazo[iy, jx]

    return g

     
def bih(gazo, kakudairitu, kaitenn):  # Bi-linear hokannn
    # 変更前の画像の 縦、横 のサイズ
    hi, wi = gazo.shape[0], gazo.shape[1]
    # 目的の画像の 縦、横 のサイズ
    g_hi, g_wi = gazo.shape[0]*kakudairitu, gazo.shape[1]*kakudairitu
    g = np.empty((g_hi, g_wi))
    
    for yb in range(0, g_hi):
        for xb in range(0, g_wi):
            x, y = xb/kakudairitu, yb/kakudairitu
            ox, oy = int(x), int(y)
            # 存在しない座標の処理
            if ox > wi -2: ox = wi -2
            if oy > hi -2: oy = hi -2

            #重みの計算
            dx = x - ox
            dy = x - oy
            g[yb][xb] = (1-dx)*(1-dy)*gazo[oy][ox] + dx*(1-dy)*gazo[oy][ox+1] + (1-dx)*dy*gazo[oy][ox+1] + dx*dy*gazo[oy+1][ox+1]
    
            if kaitenn == 0:
                exit
            else :
                for i in range(0, g_hi):
                    y = g_hi/2 - i
                    for j in range(0, g_wi):
                        x = j - g_wi/2
                        xr =  x*math.cos(math.radians(kaitenn)) + y*math.sin(math.radians(kaitenn))
                        yr = -x*math.sin(math.radians(kaitenn)) + y*math.cos(math.radians(kaitenn))
                        #バイリニア補間

                        jx = int(xr + g_wi/2 + 0.5)
                        iy = int(g_wi/2 - yr + 0.5)
                        
                        dx = xr - jx
                        dy = yr - iy

                        # 存在しない座標の処理
                        if (0 <= iy & iy < g_hi) & (0 <= jx & jx < g_wi):
                            g[i,j] = gazo[iy, jx]
                            
    return g


def yomikomi(fp: str):
    # fp : ファイル名 (string) 
    # ここで画像を読み込み
    return np.array(open(fp))

def save(obj, fp: str):

    fromarray(obj).save(fp)