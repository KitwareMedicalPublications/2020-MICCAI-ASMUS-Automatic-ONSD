import sys
from datetime import datetime
from ocularus import *
from data_manager import *
import json
import sys
sys.setrecursionlimit(30000)

# size_id = sys.argv[1]
size_ids = ['3mm', '4mm', '5mm', '6mm', '7mm']
# ids = ['3mm-capture_1', '4mm-capture_1', '5mm-capture_1', '6mm-capture_1', '7mm-capture_1']
capture_ids = ['-capture_' + str(x) for x in [5, 6, 7, 8, 9, 10]]
dm = DataManager('clarius phantom study-20200308')
eyeseg = EyeSegmentationRANSAC()
nerveseg = NerveSegmentationSkeleton()
for size_id in size_ids:
    for cid in capture_ids:
        myid = size_id + cid
        filepath = dm.get_by_absid('video', myid)
        outdir = myid + '-out'
        cor = ClariusOfflineReader(filepath)
        cropper = CropPreprocess(cor.get(0), 0.05, 0.05)
        results = []
        for i in range(cor.video.shape[0]):
            print(i)
            t1 = datetime.now()
            img = cor.get(i)
            img2 = cropper.crop(img)
            print('here')
            eye = eyeseg.process(img2)
            print('here2')
            if not eye.found():
                nerve = None
            else:
                nerve = nerveseg.process(img2, eye)
            print('here3')
            t2 = datetime.now()
            duration = t2-t1
            print(t2-t1)
            results.append((img2, eye, nerve, duration))
            write_result(outdir, i, img2, eye, nerve, duration)
        width = estimate_width(results)
        with open(outdir + '/' + 'width.p', 'wb') as f:
            pickle.dump(width, f)
