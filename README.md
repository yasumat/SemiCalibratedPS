# Semi-calibrated Photometric Stereo in Python

written by Yasuyuki Matsushita (yasumat@ist.osaka-u.ac.jp) supported 
by Osaka University and JSPS KAKENHI Grant Number JP16H01732, Japan

based on a part of photometric stereo works in 2010-2018 conducted at Microsoft Research Asia and Osaka University
together with external collaborators listed in the publication list below.

### What is Photometric Stereo?

Photometric Stereo is an approach to determining surface normal of 
a scene from a set of images recorded from a fixed viewpoint but under
varying lighting conditions, originally proposed by Woodham [1].
<p align="center">
<img src="./lambert_noshadow.gif" width="256"> &rarr; <img src="./gt_normal_disp.png" width="256">
</p>

### What is *Semi-calibrated* Photoemtric Stereo?

Conventional Photometric Stereo assumes calibration of *both* light directions and intensities. 
Semi-calibrated photometric stereo only assumes the known light directions, 
but the light intensities (and camera exposures as well) are treated unknown. The unknown light intensities are 
simultaneously estimated together with surface normal. It eliminates the need for calibrating light intensities, and
allows the use of auto-exposure of the camera without remembering the camera parameter, such as shutter-speed.

### How to use?

Download this package and run the demo code:
```
python demo.py
```

To switch solution methods, look into `demo.py` and choose any of the following one when calling `SCPS.solve`

    LINEAR            # Linear solution method
    FACTORIZATION     # Factorization based method
    ALTERNATE         # Alternating minimization method
    
### Conditions of use

This package is distributed under the GNU General Public License. For
information on commercial licensing, please contact the authors at the
contact address below. If you use this code for a publication, please
consider citing the following papers:

    @inproceedings{SCPS2018,
        title   = {Semi-Calibrated Photometric Stereo},
        author  = {DongHyeon Cho, Yasuyuki Matsushita, Yu-Wing Tai, and In So Kweon},
        journal = {IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)},
        year    = {2018}
	}
	
	@inproceedings{SCPS2016,
	    title     = {Photometric Stereo Under Non-uniform Light Intensities and Exposures},
	    author    = {Donghyeon Cho, Yasuyuki Matsushita, Yu-Wing Tai, and In So Kweon},
        booktitle = {European Conference on Computer Vision (ECCV)},
        year      = {2016},
        volume    = {II},
        pages     = {170--186}
    }  

	
### Dependencies
The code is written in Python 3.6 but should be able to adapt it to Python 2.x if needed.
You might need the following Python packages installed:
* `cv2` (OpenCV, used for image I/O)
* `glob` (used for reading out a list of images)
* `numpy` (main computation depends on matrix operations)
* `sklearn` (scikit-learn, used for normalization of array)

### Acknowledgements

This code implementation work is supported by Osaka University and JSPS KAKENHI Grant
Number JP16H01732, Japan.

### Contact information

Questions / Comments? Bug reports? Please contact Yasuyuki Matsushita at yasumat@ist.osaka-u.ac.jp.

### References

[1] Woodham, R.J. Photometric method for determining surface orientation from multiple images. 
Optical Engineerings 19, I, 139-144, 1980

