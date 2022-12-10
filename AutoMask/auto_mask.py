import os
import cv2
import numpy as np
from PIL import Image

def auto_mask(filedir):
    # Remove Background
    last_slash = filedir.rindex('/')
    filename = filedir[last_slash+1:]
    nobg_filedir = filedir[:last_slash+1] + 'nobg_' + filename
    os.system('backgroundremover -i ' + filedir + ' -o ' + nobg_filedir)
    
    # Get mask
    input_image = cv2.imread(filedir) 
    nobg_image = cv2.imread(nobg_filedir)
    pre_mask = cv2.cvtColor(input_image - nobg_image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(pre_mask, 5, 255, cv2.THRESH_BINARY)
    cv2.imwrite(filedir[:last_slash+1] + 'masked_' + filename, mask)
    
    os.system('rm ' + nobg_filedir)
