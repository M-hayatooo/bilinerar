import math
import numpy as np
from PIL.Image import open, fromarray

def Nh(gazo, kakudairitu, kaitenn): # NearestHokann
    if not isinstance(kakudairitu, tuple):
            kakudairitu = (kakudairitu, kakudairitu)
    kakudairitu = (float(kakudairitu[0]), float(kakudairitu[1]))

    if kaitenn == 0:
        #ここはグレースケールかカラー画像かの場合分け
        if gazo.shape[2] == None:
            size_store = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]))
        else:
            size_store = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]), gazo.shape[2])
        g = np.zeros(size_store, dtype=np.uint8)
    else: # 回転するなら回転後の部分が写る最大サイズを用意
        if gazo.shape[0]>gazo.shape[1]:            
            if gazo.shape[2] == None:
                size_store = (round(gazo.shape[0] * kakudairitu[0] ), round(gazo.shape[1] * kakudairitu[1] ))
                size_store_r = (round(gazo.shape[0] * kakudairitu[0] * 1.5), round(gazo.shape[0] * kakudairitu[1] * 1.5))
            else:
                size_store = (round(gazo.shape[0] * kakudairitu[0] ), round(gazo.shape[1] * kakudairitu[1] ), gazo.shape[2])
                size_store_r = (round(gazo.shape[0] * kakudairitu[0] * 1.5), round(gazo.shape[0] * kakudairitu[1] * 1.5), gazo.shape[2])
            g = np.zeros(size_store, dtype=np.uint8)
            gr = np.zeros(size_store_r, dtype=np.uint8)

        elif gazo.shape[0] <= gazo.shape[1]:            
            if gazo.shape[2] == None:
                size_store = (round(gazo.shape[0] * kakudairitu[0] ), round(gazo.shape[1] * kakudairitu[1] ))
                size_store_r = (round(gazo.shape[1] * kakudairitu[0] * 1.5), round(gazo.shape[1] * kakudairitu[1] * 1.5))
            else:
                size_store = (round(gazo.shape[0] * kakudairitu[0] ), round(gazo.shape[1] * kakudairitu[1] ), gazo.shape[2])
                size_store_r = (round(gazo.shape[1] * kakudairitu[0] * 1.5), round(gazo.shape[1] * kakudairitu[1] * 1.5), gazo.shape[2])
            g = np.zeros(size_store, dtype=np.uint8)
            gr = np.zeros(size_store_r, dtype=np.uint8)

    #ghi = int(g_hi), gwi = int(g_wi)
    #shape = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]))
    
    # g.shape[0] == g_hi
    for x in range(g.shape[1]):
        for y in range(g.shape[0]):
            xi, yi = int( round(x / kakudairitu[0])) ,int( round( y / kakudairitu[0] ) ) 
            # 存在しない座標の処理
            if xi > gazo.shape[1] -1: xi = gazo.shape[1] -1
            if yi > gazo.shape[0] -1: yi = gazo.shape[0] -1

            g[y, x] = gazo[yi, xi]

    if kaitenn == 0:
        return g
    else :# ここのgr.shape[0]とgr.shape[1]のループ数が逆
        if gr.shape[0] < gr.shape[1]:
            maxsize = gr.shape[1]
        else:
            maxsize = gr.shape[0]
        print(maxsize)
        for i in range(maxsize):
            y = (maxsize / 2)  - i
            for j in range(maxsize):
                x = j - (maxsize / 2)
                xr =  x * math.cos(math.radians(kaitenn)) + y * math.sin(math.radians(kaitenn))
                yr = -x * math.sin(math.radians(kaitenn)) + y * math.cos(math.radians(kaitenn))
                #最近傍補間
                jx = int(xr + maxsize / (2*1.5) + 0.5)
                iy = int(maxsize / (2*1.5) - yr + 0.5)
                
                # 存在しない座標の処理
                if (0 <= iy & iy < g.shape[0]) & (0 <= jx & jx < g.shape[1]):
                    gr[i, j] = g[iy, jx]
        return gr

     
def bih(gazo, kakudairitu, kaitenn):  # Bi-linear hokannn
    if not isinstance(kakudairitu, tuple):
            kakudairitu = (kakudairitu, kakudairitu)
    kakudairitu = (float(kakudairitu[0]), float(kakudairitu[1]))

    # 元の画像の 縦、横 のサイズ
    wi = int( gazo.shape[1]  )## x
    hi = int( gazo.shape[0]  )## y

    # 目的の画像の 縦、横 のサイズ
    g_wi = int( gazo.shape[1] * kakudairitu[1] )## x座標
    g_hi = int( gazo.shape[0] * kakudairitu[0] )## y座標
    print("x=",g_wi, "   y=",g_hi)
    if kaitenn == 0:
        #これはグレースケールかカラー画像かの場合分け
        if gazo.shape[2] == None:
            size_store = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]))
        else:
            size_store = (round(gazo.shape[0] * kakudairitu[0]), round(gazo.shape[1] * kakudairitu[1]), gazo.shape[2])
        g = np.zeros(size_store, dtype=np.uint8)
    else: # 回転するなら回転後の部分が写る最大サイズを用意
        if gazo.shape[2] == None:
            size_store = (round(gazo.shape[0] * kakudairitu[0] * 1.5), round(gazo.shape[1] * kakudairitu[1] * 1.5))
        else:
            size_store = (round(gazo.shape[0] * kakudairitu[0] * 1.5), round(gazo.shape[1] * kakudairitu[1] * 1.5), gazo.shape[2])
        g = np.zeros(size_store, dtype=np.uint8)
        
    for xb in range(g.shape[1]):
        for yb in range(g.shape[0]):
            x, y = ( xb / kakudairitu[0] ), ( yb / kakudairitu[1] )
            ox, oy = int(x), int(y)
            # 存在しない座標の処理
            if ox > gazo.shape[1] -2: ox = gazo.shape[1] -2
            if oy > gazo.shape[0] -2: oy = gazo.shape[0] -2

            #重みの計算
            dx = x - ox
            dy = y - oy
            g[yb, xb] = (1 - dx) * (1 - dy) * gazo[oy, ox] + dx * (1 - dy) * gazo[oy, ox+1] + (1 - dx) * dy * gazo[oy+1, ox] + dx * dy * gazo[oy+1, ox+1]
    
            if kaitenn == 0:
                exit
            else :
                for i in range(0, g_hi):
                    y = g_hi / 2 - i
                    for j in range(0, g_wi):
                        x = j - g_wi / 2
                        xr =  x * math.cos(math.radians(kaitenn)) + y * math.sin(math.radians(kaitenn))
                        yr = -x * math.sin(math.radians(kaitenn)) + y * math.cos(math.radians(kaitenn))
                        #バイリニア補間
                        jx = int(xr + g_wi / 2 + 0.5)
                        iy = int(g_wi / 2 - yr + 0.5)
                        
                        dx = xr - jx
                        dy = yr - iy

                        # 存在しない座標の処理
                        if (0 <= iy & iy < hi) & (0 <= jx & jx < wi):
                            g[i,j] = gazo[iy, jx]
                            
    return g


def yomikomi(fp: str):  # fp : ファイル名 (string) 
    # ここで画像を読み込み
    return np.array(open(fp))

def save(obj, fp: str): #これで画像を保存
    fromarray(obj).save(fp)
