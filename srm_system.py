import numpy as np
from level_set import *
import copy

import constants

class SRM_Assembly():
    def __init__(self, sub_srms):
        self.srm_counts = copy.deepcopy(sub_srms)
        self.sub_srms = list(self.srm_counts.keys())
    def isStillBurning(self):
        return np.any([sub.isStillBurning() for sub in self.sub_srms])

    def getArea(self):
        tot_area = 0
        for sub in self.sub_srms:
            if sub.isStillBurning():
                tot_area += sub.getArea() * self.srm_counts[sub]
        return tot_area

    def advance(self, vel):
        for sub in self.sub_srms:
            if sub.isStillBurning():
                sub.advance(vel)

    def getLength(self):
        lens = []
        for sub in self.sub_srms:
            sub_len = sub.getLength()
            lens.append(sub_len)
        return np.average(lens)

    def make_plot(self):
        figs = []
        for sub in self.sub_srms:
            if type(sub) == LevelSetSRM:
                figs.append(sub.make_plot())
        return figs
                
        

class SRM_Base:
    def __init__(self):
        self.L_t = constants.grain_length
    
    def getLength(self):
        return self.L_t

    def advanceLengthwise(self, vel):
        self.L_t = self.L_t - 2 * vel * constants.delta_t

    def advance(self, vel):
        return NotImplementedError()

        

class BatesSRM(SRM_Base):
    def __init__(self):
        super(BatesSRM, self).__init__()
        self.r_t = constants.R_inner

    def advance(self, vel):
        # compute gradient
        self.advanceLengthwise(vel)
        self.r_t = self.r_t + vel * constants.delta_t

    def getArea(self):
        area_inner = self.L_t * 2. * np.pi * self.r_t

        # print("area inner is {}".format(area_inner))

        # area for one end
        cap_area = np.pi * (constants.R_outer ** 2 - self.r_t ** 2)

        # print("bates cap area is {}".format(cap_area))

        if cap_area <= .1:
            print("small cap area {}, perim is {}, len is {}".format(cap_area, area_inner / L_t,  L_t))

        return area_inner + cap_area + cap_area

    def getVolume(self):
        # area for one end
        cap_area = np.pi * (constants.R_outer ** 2 - self.r_t ** 2)

        return self.L_t * cap_area

    def isStillBurning(self):
        return self.r_t < constants.R_outer and self.getLength() > 0


class LevelSetSRM(SRM_Base):
    def __init__(self, init_conds):
        super(LevelSetSRM, self).__init__()
        Nx = Ny = 200

        Lx = Ly = (constants.R_outer) * 2.1

        self.level_set = LevelSet(Nx, Ny, Lx, Ly, init_conds)

        self.area = None

    def advance(self, vel):
        # advance level set
        self.level_set.advance(vel)
        # also advance my length
        self.advanceLengthwise(vel)
    
        # area is stale
        self.area = None

    def getArea(self):
        # lazy-compute it
        if self.area is None:
            perim = self.level_set.computePerim()
            # compute area of internal curve
            internal_area = self.level_set.computeArea()

            outer_circ_area = np.pi * constants.R_outer * constants.R_outer
            cap_area = outer_circ_area - internal_area

            if cap_area <= .1:
                print("small cap area {}, perim is {}, len is {}".format(cap_area, perim,  self.getLength()))
                # self.level_set.make_plot("test")

            if cap_area <= 0:
                print("cap area is {}, outer is {}, inner is {}".format(cap_area, outer_circ_area, internal_area))
                assert (False)
            self.area = perim * self.getLength() + 2 * cap_area
        return self.area

    def getVolume(self):
        return self.level_set.computeArea() * self.getLength()

    def isStillBurning(self):
        if self.getLength() <= 0:
            print("no length left!")
            return False
        SDF = np.copy(self.level_set.getSDF())
    
        nm = np.max(SDF[self.level_set.X**2 + self.level_set.Y**2 <= constants.R_outer**2])

        # if nm < 0.1:
        #     print("nm is:", nm)

        min_dist = min(self.level_set.delta_x, self.level_set.delta_y)

        # Is any part of the SDF still burning?
        return nm >= min_dist / 4.

    def make_plot(self):
        return self.level_set.make_plot()
