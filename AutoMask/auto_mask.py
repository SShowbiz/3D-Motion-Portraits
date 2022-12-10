import cv2
from subprocess import Popen

def auto_mask(filedir):
    # Remove Background
    last_slash = filedir.rindex('/')
    filename = filedir[last_slash+1:]
    nobg_filedir = filedir[:last_slash+1] + 'nobg_' + filename

    Popen(['backgroundremover', '-i', filedir, '-o', nobg_filedir]).wait()
    
    # Get mask
    input_image = cv2.imread(filedir) 
    nobg_image = cv2.imread(nobg_filedir)

    pre_mask = cv2.cvtColor(input_image - nobg_image, cv2.COLOR_BGR2GRAY)

    _, mask = cv2.threshold(pre_mask, 5, 255, cv2.THRESH_BINARY)

    mask_file_name = filedir[:last_slash+1] + 'masked_' + filename
    cv2.imwrite(mask_file_name, mask)
    
    Popen(['rm', nobg_filedir]).wait()

    return mask_file_name
