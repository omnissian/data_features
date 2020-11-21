import os
from PIL import Image
import numpy as np
import io
from wand.image import Image as wi
import PIL
import pdb
def uniq_pix(path):
    names = os.listdir(path)
    for name in names:
        name="Doc5.pdf"
        path_current = path + name
        # input1 = PdfFileReader(open(path_current, "rb"))
        PDFfile = wi(filename=path_current,resolution=400)
        Images = PDFfile.convert('jpg')
        ImageSequence = 1
        for img in PDFfile.sequence:
            Image = wi(image=img)
            print("type(Image) ",type(Image))
            Image.save(filename=path_current+"Image"+str(name) + str(ImageSequence) + ".png")

            #----------- to Pillow image-----
            img_buffer = np.asarray(bytearray(Image.make_blob(format='png')), dtype='uint8')
            pdb.set_trace()
            bytesio = io.BytesIO(img_buffer)
            pil_img = PIL.Image.open(bytesio)
            #----------- to Pillow image-----

            ImageSequence += 1

path = "D:/PathName/PathName2/Documents1/"
uniq_pix(path)

name = "LABEL_5.png"
