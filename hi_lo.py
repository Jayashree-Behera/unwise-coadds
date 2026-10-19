import numpy as np

# this class is meant to store the min and max images, which will get
# updated
class HiLo():
    def __init__(self):
        self.sidelen = 2048 # square ; this is in coadd pixel space
        self.im_min = np.zeros((self.sidelen, self.sidelen), dtype=float) + 1.0e30
        self.im_max = np.zeros((self.sidelen, self.sidelen), dtype=float) - 1.0e30
        self.id_min = np.zeros((self.sidelen, self.sidelen), dtype='S9')
        self.id_max = np.zeros((self.sidelen, self.sidelen), dtype='S9')

    def update(self, rimg):

        id = rimg.scan_id + str(rimg.frame_num).zfill(3)

        x0_coadd = rimg.coextent[0]
        x1_coadd = rimg.coextent[1] + 1
        y0_coadd = rimg.coextent[2]
        y1_coadd = rimg.coextent[3] + 1

        slc = slice(y0_coadd, y1_coadd), slice(x0_coadd, x1_coadd)

        # Basic slicing returns views, so writing through sub_* already
        # updates self.*; the assignments back into self.*[slc] that used to
        # follow were copies of each sub-array onto itself. Boolean masks also
        # avoid materialising the two index arrays np.where built only to feed
        # straight back into fancy indexing.
        sub_im_min = self.im_min[slc]
        sub_im_max = self.im_max[slc]

        is_min = rimg.rimg < sub_im_min
        is_max = rimg.rimg > sub_im_max

        sub_im_min[is_min] = rimg.rimg[is_min]
        sub_im_max[is_max] = rimg.rimg[is_max]

        self.id_min[slc][is_min] = id
        self.id_max[slc][is_max] = id
