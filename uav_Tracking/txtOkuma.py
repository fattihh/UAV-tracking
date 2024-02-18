import os
import glob

def txtOku(klasorYolu):
    klasor_yolu=klasorYolu
    txt_dosyaları = glob.glob(os.path.join(klasor_yolu,'*.txt'))
    return txt_dosyaları



