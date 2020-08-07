# 2020-MICCAI-ASMUS-Automatic-ONSD

This code requires python (tested with python 3.76) and [ffmpeg](https://ffmpeg.org/download.html) installed.  I strongly recommend installing ffmpeg to C:\src\ffmpeg-4.2.2-wind64-dev
as there are some hard-coded imports of skvideo.  Otherwise, update the top of ocularus.py to point to your install location.

```
cd REPO_dir
pip install -r requirements.txt
```

## Repo structure
The notebooks have been tested with Jupyter Lab.

```
./data/2020-MICCAI-ASMUS-output
+ analytical "raw" data from the phantom study

./data/clarius phantom study-202003308
+ the 50 video phantom dataset as .mp4

./python/cmd-run.py
+ executing python `cmd-run.py` will process the entire phantom dataset (this generates gigs of data)

./python/clarius.py
+ data for loading the .mp4 files from an clarius ultrasound

./python/data_manager.py
+ utility for managing paths in the dataset

./python/ocularus.py
+ main code file

./python/Phantom Study.ipynb
+ some examples and creates the main figures and analysis in the paper

./Eye Mold v3.stl
+ The 3D-printed mold referenced in the paper.  
```

Contact [brad.moore@kitware.com](mailto:brad.moore@kitware.com) or post an issue on github.


