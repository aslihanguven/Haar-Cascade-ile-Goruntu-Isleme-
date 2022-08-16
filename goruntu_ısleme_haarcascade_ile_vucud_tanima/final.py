# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 19:46:20 2022

@author: aslig
"""

#cv2 ve numpy kütüphaneleri eklendi
import cv2
import numpy as np

#video açma işlemi yapıldı
cam = cv2.VideoCapture("video.avi")

#tüm vücut için cascade dosyası okunuyor
tum_vucut = cv2.CascadeClassifier("haarcascade_fullbody.xml")
#alt vücut için cascade dosyası okunuyor
alt_vucut = cv2.CascadeClassifier("haarcascade_lowerbody.xml")
#üst vücut için cascade dosyası okunuyor
ust_vucut = cv2.CascadeClassifier("haarcascade_upperbody.xml")

#resim okuma işlemi yapıldı
ret, resim = cam.read()

# 0'lardan oluşan boş bir resim oluşturuldu
tespit = np.zeros((resim.shape[0],resim.shape[1],3), np.uint8)

#trackbar oluşturmak için kullanılmayacak bir fonksiyon oluşturuldu
def nothing(x):
    pass

#resim penceresi oluşturuldu
cv2.namedWindow("resim", cv2.WINDOW_NORMAL)

#parametreleri hazır olarak girmeyip, elle ayarlayabilmemiz için üç adet trackbar oluşturuldu.
#ilk parametre
cv2.createTrackbar("ilk_param", "resim", 0, 100, nothing)
#ikinci parametre
cv2.createTrackbar("ikinci_param", "resim", 0, 100, nothing)
#programı durdurup , devam ettirebilmemiz için bir switch oluşturuldu.
cv2.createTrackbar("switch", "resim", 0, 1, nothing)

#video açma işlemi yapıldı.
while cam.isOpened():
    #döngü her başladığında oluşturduğumuz boş resmin sıfırlanması yani silinmesi sağlandı.
    tespit[:] = 0
    
    #video üzerinden switche bakıldı. Switch 1 ise beklemeye alınıyor , 0 ise devam ediliyor.
    if cv2.getTrackbarPos("switch", "resim") == 1:
        cv2.waitKey(1)
        continue
    
    #video üzerinden resim okuma işlemi yapıldı   
    ret, resim = cam.read()
    
    #resim okunamazsa program sonlanıyor ve ekrana bitti yazdırılıyor
    if not ret:
        print("bitti")
        break
    
    #resim okuma işlemi tamamlandıktan sonra griye dönüştürülüyor
    resim_gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    
    #gri resim üzerinde trackbarlardan pozisyonlar alındı
    #İlk parametre 1.01den ikinci parametre ise 1'den başlatılıyor ve artarak devam ediyor
    ilk_param = cv2.getTrackbarPos("ilk_param", "resim")/100+1.01
    ikinci_param = cv2.getTrackbarPos("ikinci_param", "resim")+1
    
    #parametreler yazdırılıyor
    print("ilk parametre: {}, ikinci parametre: {} ".format(
        ilk_param,ikinci_param))
    
    
    #vücutların tespiti yaptırıldı. minSize ve maxSize ile oynayarak doğruluk arttırılabilir
    vucutlar = tum_vucut.detectMultiScale(resim_gri, ilk_param, ikinci_param,minSize=(40,40),maxSize=(110,110))
    #alt vücut tespiti yapıldı
    alt_vucutlar = alt_vucut.detectMultiScale(resim_gri,ilk_param, ikinci_param,minSize=(40,40),maxSize=(80,80))
    #üst vücut tespiti yapıldı.
    ust_vucutlar = ust_vucut.detectMultiScale(resim_gri, ilk_param, ikinci_param,minSize=(40,40),maxSize=(80,80))
    
    
    #vücutların çizdirme işleminin yaptırılması için for döngüsü açıldı
    for x, y, w, h in vucutlar:
        #tespit edilen tüm vücutlar , oluşturduğumuz boş resmin içine eklendi
        tespit[y:y+h, x:x+w] = resim[y:y+h, x:x+w]
        # ilk nokta ,(x,y) noktası olarak, ikinci nokta ise (x+w , y+h) olarak belirlendi.Çerçeve rengi pembe ve kalınlığı 3 olarak belirlendi
        cv2.rectangle(resim, (x,y), (x+w, y+h), (255,0,255), 3)
        #tespit edilen vücutların maviye boyanması sağlandı
        resim[y:y+h, x:x+w, 0] = 255
        
     #alt vücutların çizdirme işleminin yaptırılması için for döngüsü açıldı 
    for x, y, w, h in alt_vucutlar:
        #ilk nokta ,(x,y) noktası olarak, ikinci nokta ise (x+w , y+h) olarak belirlendi.Çerçeve rengi beyaz ve kalınlığı 2 olarak belirlendi
        cv2.rectangle(resim, (x,y), (x+w, y+h), (255,255,255), 2)
        #tespit edilen alt vücutların yeşile boyanması sağlandı.
        resim[y:y+h, x:x+w, 1] = 255
    
    #üst vücutların çizdirme işleminin yaptırılması için for döngüsü açıldı.
    for x, y, w, h in ust_vucutlar:
        #ilk nokta ,(x,y) noktası olarak, ikinci nokta ise (x+w , y+h) olarak belirlendi.Çerveve rengi mavi ve kalınlığı 1 olarak belirlendi
        cv2.rectangle(resim, (x,y), (x+w, y+h), (255,153,51), 1)
        #tespit edilen üst vücutların kırmızıya boyanması sağlandı
        resim[y:y+h, x:x+w, 2] = 255
        
    
    #görüntüleme işlemleri yapıldı
    cv2.imshow("resim",resim)
    cv2.imshow("tespit edilenler",tespit)
    #sonlandırma tuşu q olarak belirlendi.Eğer 5 ms'de q tuşuna basılırsa break devreye girecek ve döngüden çıkılacaktır.
    if cv2.waitKey(5) == ord("q"):
        print("by")
        break

#tüm penceler kapatıldı
cv2.destroyAllWindows()
