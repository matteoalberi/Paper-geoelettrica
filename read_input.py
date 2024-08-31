""" Import data library.
- ReadIris reads Syscal iris binary .bin and .pro files
- ReadStn reads Geometrics OhmMapper ASCII .stn files
- ReadGps reads gps .txt files and, if requested, performs data interpolation.
written by Silvio Pierini """

import os
import re
import warnings

import numpy as np
import datetime
from scipy import interpolate

A_ohm=1e-3
Ppc = .8
class SyscalData(object):
    def __init__(self, url=None, version=None, type_of=None, comment=None, cole_cole=None, common_file_path=None,
                 sz_name=np.empty(0), fname_iab_vmn=np.empty(0), nb_files=None, download_version=None, size=None,
                 measure=None):
        self.url = url
        self.version = version
        self.type_of = type_of
        self.comment = comment
        self.cole_cole = cole_cole
        self.common_file_path = common_file_path
        self.nb_files = nb_files
        self.size_file_name = sz_name
        self.file_name_iab_or_vmn = fname_iab_vmn
        self.download_version = download_version
        self.size = size
        if measure is None:
            self.measure = ErtMeasure()
        else:
            self.measure = measure

    def ReadBin(self):
        """function that read .bin iris output"""
        # open data file
        fid = open(self.url, "rb")
        # find eof
        fid.seek(0, 2)
        eof = fid.tell()
        fid.seek(0, 0)
        # version
        self.version = np.fromfile(fid, "I", 1)
        # read type of syscal
        self.type_of = np.fromfile(fid, np.uint8, 1)

        # read comment string
        if any(self.type_of == i for i in np.array([3, 4, 5, 8, 9, 10, 11])) or (
                self.version >= 2147483650 and any(self.type_of == i for i in np.array([1, 2, 6, 7]))):
            self.comment = np.fromfile(fid, "c", 1024)

        # todo: if not tested
        if (any(self.type_of == i for i in np.array([3, 4, 5, 8, 9, 11])) and self.version == 2147483651) or (
                self.version >= 2147483651 and any(self.type_of == i for i in np.array([1, 6, 10]))):
            self.cole_cole = np.fromefile(fid, float, 64000 * 3)

        # todo: if not tested
        if self.version >= 2147483652:
            self.common_file_path = np.fromfile(fid, "c", 260)
            self.nb_files = np.fromfile(fid, "H", 1)
            for i in range(self.nb_files):
                self.size_file_name = np.append(self.size_file_name, np.fromfile(fid, "H", 1))
                self.file_name_iab_or_vmn = np.append(self.file_name_iab_or_vmn, np.fromfile(fid, "c",
                                                                                             self.size_file_name[i]))

        # read data block
        if any(self.type_of == i for i in np.array([3, 4, 5, 8, 9, 11])):
            while fid.tell() < eof:
                self.measure.el_array = np.append(self.measure.el_array, np.fromfile(fid, "h", 1))
                self.measure.more_tm_measure = np.append(self.measure.more_tm_measure, np.fromfile(fid, "h", 1))
                self.measure.time = np.append(self.measure.time, np.fromfile(fid, "f", 1))
                self.measure.m_dly = np.append(self.measure.m_dly, np.fromfile(fid, "f", 1))
                self.measure.type_cp_xyz = np.append(self.measure.type_cp_xyz, np.fromfile(fid, "h", 1))
                self.measure.q = np.append(self.measure.q, np.fromfile(fid, "h", 1))
                self.measure.pos = np.vstack([self.measure.pos, np.fromfile(fid, "f", 12)])
                self.measure.ps = np.append(self.measure.ps, np.fromfile(fid, "f", 1))
                self.measure.vp = np.append(self.measure.vp, np.fromfile(fid, "f", 1))
                self.measure.in_ = np.append(self.measure.in_, np.fromfile(fid, "f", 1))
                self.measure.rho = np.append(self.measure.rho, np.fromfile(fid, "f", 1))
                self.measure.m = np.append(self.measure.m, np.fromfile(fid, "f", 1))
                self.measure.dev = np.append(self.measure.dev, np.fromfile(fid, "f", 1))
                self.measure.tm = np.vstack([self.measure.tm, np.fromfile(fid, "f", 20)])
                self.measure.mx = np.vstack([self.measure.mx, np.fromfile(fid, "f", 20)])

                tmp = fid.read(1)
                if len(str(tmp)) >= 5:
                    self.measure.channel = np.append(self.measure.channel,
                                                     int(bin(int(str(tmp)[-3:-1], base=16))[-4:], base=2))
                    self.measure.nb_channel = np.append(self.measure.nb_channel,
                                                        int(bin(int(str(tmp)[-3:-1], base=16))[2:6], base=2))
                else:
                    self.measure.channel = np.append(self.measure.channel, 0)
                    self.measure.nb_channel = np.append(self.measure.nb_channel, 0)

                tmp = fid.read(1)
                if str(tmp).isdigit():
                    self.measure.overload = np.append(self.measure.overload,
                                                      int(bin(int(str(tmp)[-3:-1], base=16))[-1:], base=2))
                    self.measure.channel_valide = np.append(self.measure.channel_valide,
                                                            int(bin(int(str(tmp)[-3:-1], base=16))[-2:-1], base=2))

                self.measure.quad_number = np.append(self.measure.quad_number, np.fromfile(fid, "H", 1))
                self.measure.name = np.append(self.measure.name, np.fromfile(fid, "c", 12))
                self.measure.lat = np.append(self.measure.lat, np.fromfile(fid, "f", 1))
                self.measure.lon = np.append(self.measure.lon, np.fromfile(fid, "f", 1))
                self.measure.nb_cren = np.append(self.measure.nb_cren, np.fromfile(fid, "f", 1))
                self.measure.rs_chk = np.append(self.measure.rs_chk, np.fromfile(fid, "f", 1))

                if self.measure.more_tm_measure[-1] == 2:
                    self.measure.tx_vab = np.append(self.measure.tx_vab, np.fromfile(fid, "f", 1))
                    self.measure.tx_bat = np.append(self.measure.tx_bat, np.fromfile(fid, "f", 1))
                    self.measure.rx_bat = np.append(self.measure.rx_bat, np.fromfile(fid, "f", 1))
                    self.measure.temperature = np.append(self.measure.temperature, np.fromfile(fid, "f", 1))
                elif self.measure.more_tm_measure[-1] == 3:
                    self.measure.tx_vab = np.append(self.measure.tx_vab, np.fromfile(fid, "f", 1))
                    self.measure.tx_bat = np.append(self.measure.tx_bat, np.fromfile(fid, "f", 1))
                    self.measure.rx_bat = np.append(self.measure.rx_bat, np.fromfile(fid, "f", 1))
                    self.measure.temperature = np.append(self.measure.temperature, np.fromfile(fid, "f", 1))
                    self.measure.date_time = np.append(self.measure.date_time, 693960 + np.fromfile(fid, "d", 1))

                if self.version >= 2147483652:
                    self.measure.iab_file = np.append(self.measure.iab_file, np.fromfile(fid, "h", 1))
                    self.measure.vmn_file = np.append(self.measure.vmn_file, np.fromfile(fid, "h", 1))

        else:
            print(self.version)
            raise Exception('Device not managed for the moment')
        return

    def ReadPro(self):
        """function that read .pro iris output"""
        # open data file
        fid = open(self.url, "rb")
        # find eof
        fid.seek(0, 2)
        eof = fid.tell()
        fid.seek(0, 0)

        self.download_version = np.fromfile(fid, "i", 1)
        self.size = np.fromfile(fid, "i", 1)
        while fid.tell() < eof:
            page = np.fromfile(fid, "i", 1)
            test = np.fromfile(fid, "I", 1)
            if test <= 0:  # todo: not tested
                break

            self.measure.date_time = np.append(self.measure.date_time, test)
            self.measure.name = np.append(self.measure.name, np.fromfile(fid, "c", 12))

            tmp = fid.read(1)
            self.measure.channel = np.append(self.measure.channel, int(bin(int(str(tmp)[-3:-1], base=16))[-4:], base=2))
            self.measure.nb_channel = np.append(self.measure.nb_channel,
                                                int(bin(int(str(tmp)[-3:-1], base=16))[2:6], base=2))
            tmp = fid.read(1)
            self.measure.overload = np.append(self.measure.overload,
                                              int(bin(int(str(tmp)[-3:-1], base=16))[-1:], base=2))
            self.measure.channel_valide = np.append(self.measure.channel_valide,
                                                    int(bin(int(str(tmp)[-3:-1], base=16))[-2:-1], base=2))

            self.measure.quad_number = np.append(self.measure.quad_number, np.fromfile(fid, "H", 1))

            tmp = fid.read(1)
            self.measure.vrunning = np.append(self.measure.vrunning,
                                              int(bin(int(str(tmp)[-3:-1], base=16))[-1:], base=2))
            self.measure.vsigned = np.append(self.measure.vsigned,
                                             int(bin(int(str(tmp)[-3:-1], base=16))[-2:-1], base=2))
            self.measure.normalized = np.append(self.measure.normalized,
                                                int(bin(int(str(tmp)[-3:-1], base=16))[-3:-2], base=2))
            self.measure.imperial = np.append(self.measure.imperial,
                                              int(bin(int(str(tmp)[-3:-1], base=16))[-4:-3], base=2))

            tmp = fid.read(1)
            self.measure.time_set = np.append(self.measure.time_set,
                                              int(bin(int(str(tmp)[-3:-1], base=16))[-4:], base=2))
            self.measure.time_mode = np.append(self.measure.time_mode,
                                               int(bin(int(str(tmp)[-3:-1], base=16))[2:6], base=2))

            tmp = fid.read(1)
            self.measure.type_ = np.append(self.measure.type_, int(bin(int(str(tmp)[-3:-1], base=16))[2:], base=2))

            self.measure.el_array = np.append(self.measure.el_array, np.fromfile(fid, "i", 1))
            self.measure.time = np.append(self.measure.time, np.fromfile(fid, "H", 1))
            self.measure.vdly = np.append(self.measure.vdly, np.fromfile(fid, "H", 1))
            self.measure.m_dly = np.append(self.measure.m_dly, np.fromfile(fid, "H", 1))
            self.measure.tm = np.vstack([self.measure.tm, np.fromfile(fid, "H", 20)])
            self.measure.unused = np.append(self.measure.unused, np.fromfile(fid, "H", 1))
            self.measure.lat = np.append(self.measure.lat, np.fromfile(fid, "f", 1))
            self.measure.lon = np.vstack([self.measure.lon, np.fromfile(fid, "f", 12)])
            self.measure.inrx = np.append(self.measure.inrx, np.fromfile(fid, "f", 1))
            self.measure.pos = np.append(self.measure.pos, np.fromfile(fid, "f", 1))
            self.measure.mov = np.append(self.measure.mov, np.fromfile(fid, "f", 1))
            self.measure.rho = np.append(self.measure.rho, np.fromfile(fid, "f", 1))
            self.measure.dev = np.vstack([self.measure.dev, np.fromfile(fid, "f", 12)])
            self.measure.nb_cren = np.append(self.measure.nb_cren, np.fromfile(fid, "f", 1))
            self.measure.rs_chk = np.append(self.measure.rs_chk, np.fromfile(fid, "f", 1))
            self.measure.tx_vab = np.append(self.measure.tx_vab, np.fromfile(fid, "f", 1))
            self.measure.tx_bat = np.append(self.measure.tx_bat, np.fromfile(fid, "f", 1))
            self.measure.rx_bat = np.vstack([self.measure.rx_bat, np.fromfile(fid, "f", 12)])
            self.measure.temperature = np.append(self.measure.temperature, np.fromfile(fid, "f", 1))
            self.measure.ps = np.append(self.measure.ps, np.fromfile(fid, "f", 1))
            self.measure.vp = np.append(self.measure.vp, np.fromfile(fid, "f", 1))
            self.measure.in_ = np.append(self.measure.in_, np.fromfile(fid, "f", 1))
            self.measure.m = np.append(self.measure.m, np.fromfile(fid, "f", 1))
            self.measure.mx = np.append(self.measure.mx, np.fromfile(fid, "f", 1))
            self.measure.unused = np.append(self.measure.unused, np.fromfile(fid, "I", 1))
            if page == -1:
                break
        return


class ErtMeasure(object):
    def __init__(self, channel=np.empty(0), channel_valide=np.empty(0), date_time=np.empty(0), dev=np.empty(0),
                 el_array=np.empty(0), iab_file=np.empty(0), imperial=np.empty(0), in_=np.empty(0), inrx=np.empty(0),
                 lat=np.empty(0), lon=np.empty(0), m=np.empty(0), m_dly=np.empty(0), more_tmes=np.empty(0),
                 more_tm_measure=np.empty(0), mov=np.empty(0), mx=np.empty([0, 20]), name=np.empty(0),
                 nb_channel=np.empty(0), nb_cren=np.empty(0), normalized=np.empty(0), overload=np.empty(0),
                 pos=np.empty([0, 12]), ps=np.empty(0), q=np.empty(0), quad_number=np.empty(0), rho=np.empty(0),
                 rs_chk=np.empty(0), rx_bat=np.empty(0), temperature=np.empty(0), time=np.empty(0),
                 time_mode=np.empty(0), time_set=np.empty(0), tm=np.empty([0, 20]), tx_bat=np.empty(0),
                 tx_vab=np.empty(0), type_=np.empty(0), type_cp_xyz=np.empty(0), unused=np.empty(0),
                 vmn_file=np.empty(0), vdly=np.empty(0), vp=np.empty(0), vrunning=np.empty(0), vsigned=np.empty(0)):
        self.channel = channel
        self.channel_valide = channel_valide
        self.date_time = date_time
        self.dev = dev  # Q
        self.el_array = el_array  # electrode array
        self.iab_file = iab_file
        self.imperial = imperial
        self.in_ = in_  # in
        self.inrx = inrx
        self.lat = lat
        self.lon = lon
        self.m = m  # chargeability
        self.m_dly = m_dly  # m_delay
        self.more_tmes = more_tmes  # ??
        self.more_tm_measure = more_tm_measure
        self.mov = mov
        self.mx = mx  # IP Windows value
        self.name = name
        self.nb_channel = nb_channel
        self.nb_cren = nb_cren
        self.normalized = normalized
        self.overload = overload
        self.pos = pos  # electrode position
        self.ps = ps  # ps
        self.q = q  # ignored parameters
        self.quad_number = quad_number
        self.rho = rho  # resistivity
        self.rs_chk = rs_chk
        self.rx_bat = rx_bat
        self.temperature = temperature
        self.time = time  # injection time
        self.time_mode = time_mode
        self.time_set = time_set
        self.tm = tm  # IP windows duration (Tm)
        self.tx_bat = tx_bat
        self.tx_vab = tx_vab
        self.type_ = type_
        self.type_cp_xyz = type_cp_xyz  # Kid or not
        self.unused = unused
        self.vmn_file = vmn_file
        self.vdly = vdly
        self.vp = vp  # vp
        self.vrunning = vrunning
        self.vsigned = vsigned


class ElectData(object):
    def __init__(self, typ=None, quad=None, elect=None):
        if typ == 'ERT':
            self.type = 'ERT'
        elif typ == 'OHM':
            self.type = 'OHM'
        elif typ is None:
            self.type = None
        else:
            raise Exception('Il type value è errato. deve essere ERT o OHM')

        self.type = typ
        if quad is None:
            self.quad = Quad()
        else:
            self.quad = quad

        if elect is None:
            self.elect = Elect()
        else:
            self.elect = elect

    def ReadErtConfig(self, url):

        if self.type != 'ERT':
            raise Exception('Il type value è errato. ReadErtConfig può essere invocato solo da dati ERT')

        with open(url, 'r') as fid:
            lines = fid.readlines()

        i = 1
        line = lines[i].split()

        while line[0] != '#' and i < len(lines):
            i = i + 1
            line = lines[i].split()
        i = i + 1

        line = lines[i].split()

        while line[0] != '#' and i < len(lines):
            i = i + 1

            self.quad.quad_number = np.append(self.quad.quad_number, int(line[0]))
            self.quad.a = np.append(self.quad.a, int(line[1]))
            self.quad.b = np.append(self.quad.b, int(line[2]))
            self.quad.m = np.append(self.quad.m, int(line[3]))
            self.quad.n = np.append(self.quad.n, int(line[4]))

            line = lines[i].split()

        self.elect.elect_number = np.unique(np.concatenate((self.quad.a, self.quad.b, self.quad.m, self.quad.n)))

        return

    def GetErtIrisRho(self, iris_data):

        if self.type != 'ERT':
            raise Exception('Il type value è errato. GetErtIrisRho può essere invocato solo da dati ERT')

        if type(iris_data) != SyscalData:
            raise Exception('iris_data non è un oggetto "SyscalData"')

        if len(iris_data.measure.quad_number) != len(self.quad.quad_number):
            raise Exception('Il numero di quadripoli misurato e quello progettato sono differenti')
        ix = np.argsort(iris_data.measure.quad_number[np.argsort(self.quad.quad_number)])
        self.quad.rho = iris_data.measure.rho[ix]
        self.quad.vp = iris_data.measure.vp[ix]
        self.quad.dev = iris_data.measure.dev[ix]

        return

    def GetErtGpsElect(self, gps_data):

        if self.type != 'ERT':
            raise Exception('Il type value è errato. GetErtGpsElect può essere invocato solo da dati ERT')

        if type(gps_data) != GpsData:
            raise Exception('gps_data non è un oggetto "GpsData"')

        if np.any(np.unique(self.elect.elect_number) != np.unique(gps_data.id)):
            raise Exception('gli elettrodi georeferenziati non coincidono con quelli progettati!')

        ix = np.argsort(gps_data.id[np.argsort(self.elect.elect_number)])
        self.elect.elect_lat = gps_data.lat[ix]
        self.elect.elect_lon = gps_data.lon[ix]
        self.elect.elect_elev = gps_data.elev[ix]

        return

    def GetOhmData(self, ohm_data, sd_perc):

        if self.type != 'OHM':
            raise Exception('Il type value è errato. GetOhmData può essere invocato solo da dati OHM')

        if type(ohm_data) != OhmData:
            raise Exception('ohm_data non è un oggetto "OhmData"')

        self.quad.rho = np.append(self.quad.rho, ohm_data.rho)
        self.quad.vp = np.append(self.quad.rho, ohm_data.rho*A_ohm)
        self.quad.dev = np.append(self.quad.rho, sd_perc * ohm_data.rho)
        if len(self.quad.quad_number) > 0:
            self.quad.quad_number = np.append(self.quad.quad_number,
                                              np.arange(len(ohm_data.pt_num)) + self.quad.quad_number[-1] + 1)
        else:
            self.quad.quad_number = np.append(self.quad.quad_number,
                                              np.arange(len(ohm_data.pt_num)) + 1)
        if len(self.quad.n) > 0:
            self.quad.a = np.append(self.quad.a, self.quad.n[-1] + (ohm_data.pt_num - 1) * 4 + 1)
            self.quad.b = np.append(self.quad.b, self.quad.n[-1] + (ohm_data.pt_num - 1) * 4 + 2)
            self.quad.m = np.append(self.quad.m, self.quad.n[-1] + (ohm_data.pt_num - 1) * 4 + 3)
            self.quad.n = np.append(self.quad.n, self.quad.n[-1] + ohm_data.pt_num * 4)
        else:
            self.quad.a = ohm_data.pt_num * 4 + 1
            self.quad.b = ohm_data.pt_num * 4 + 2
            self.quad.m = ohm_data.pt_num * 4 + 3
            self.quad.n = ohm_data.pt_num * 4 + 4
        self.elect.elect_number = np.append(self.elect.elect_number, np.arange(len(ohm_data.pt_num) * 4) + 1)

        # interp mean rope points
        pt_lat = ohm_data.lat
        pt_lon = ohm_data.lon
        pt_elev = ohm_data.elev

        # interp electrodes lat-lon-elev
        for i in range(len(pt_lat)):

            v1 = np.array([pt_lat[i], pt_lon[i], pt_elev[i]])
            p0 = np.array([pt_lat[i], pt_lon[i], pt_elev[i]])
            if i < len(pt_lat) - 2:
                if ohm_data.line[i] == ohm_data.line[i + 1]:
                    v1 = np.array([pt_lat[i + 1], pt_lon[i + 1], pt_elev[i + 1]])

            v2 = np.array([pt_lat[i], pt_lon[i], pt_elev[i]])
            if i > 0:
                if ohm_data.line[i - 1] == ohm_data.line[i]:
                    v2 = np.array([pt_lat[i - 1], pt_lon[i - 1], pt_elev[i - 1]])

            v_dir = (v1 - v2) / np.linalg.norm(v1 - v2, axis=-1, ord=2)
            elect1 = v_dir * (-(ohm_data.rop_len + ohm_data.tr_dip * (1 - Ppc)) / 2 - ohm_data.tr_dip * Ppc) + p0
            elect2 = v_dir * (-(ohm_data.rop_len + ohm_data.tr_dip * (1 - Ppc)) / 2) + p0
            elect3 = v_dir * (+(ohm_data.rop_len + ohm_data.tr_dip * (1 - Ppc)) / 2) + p0
            elect4 = v_dir * (+(ohm_data.rop_len + ohm_data.tr_dip * (1 - Ppc)) / 2 + ohm_data.tr_dip * Ppc) + p0

            self.elect.elect_lat = np.append(self.elect.elect_lat, elect1[0])
            self.elect.elect_lon = np.append(self.elect.elect_lon, elect1[1])
            self.elect.elect_elev = np.append(self.elect.elect_elev, elect1[2])
            self.elect.elect_lat = np.append(self.elect.elect_lat, elect2[0])
            self.elect.elect_lon = np.append(self.elect.elect_lon, elect2[1])
            self.elect.elect_elev = np.append(self.elect.elect_elev, elect2[2])
            self.elect.elect_lat = np.append(self.elect.elect_lat, elect3[0])
            self.elect.elect_lon = np.append(self.elect.elect_lon, elect3[1])
            self.elect.elect_elev = np.append(self.elect.elect_elev, elect3[2])
            self.elect.elect_lat = np.append(self.elect.elect_lat, elect4[0])
            self.elect.elect_lon = np.append(self.elect.elect_lon, elect4[1])
            self.elect.elect_elev = np.append(self.elect.elect_elev, elect4[2])

        return

    def FuseElectrode(self, inp):
        if inp.thresh < 0:
            raise Exception('il threshold value è < 0!')
        # check if two electrodes are closer than a threshold and fuse them
        dist = np.empty(0)
        for i in range(len(self.elect.elect_lat[::4])):
            pt = np.array([self.elect.elect_lat[4 * i], self.elect.elect_lon[4 * i], self.elect.elect_elev[4 * i]])
            pts = np.transpose(np.stack([self.elect.elect_lat[::4], self.elect.elect_lon[::4],
                                         self.elect.elect_elev[::4]]))
            pt_mtx = np.tile(pt, [len(pts), 1])
            dst = np.linalg.norm(pt_mtx - pts, axis=1, ord=2)
            dist = np.append(dist, np.min(dst[dst > 0]))
        d_mean = np.mean(dist)

        if inp.thresh > d_mean:
            inp.thresh = d_mean
            warnings.warn('Input threshold maggiore della distanza media fra i punti = ' + str(np.round(d_mean, 2)) +
                          'm. Parametro reimpostato a tale valore')

        i = 0
        while i < len(self.elect.elect_number) - 1:
            el_num = self.elect.elect_number[i]
            pt = np.array([self.elect.elect_lat[i], self.elect.elect_lon[i], self.elect.elect_elev[i]])
            pts = np.transpose(np.stack([self.elect.elect_lat, self.elect.elect_lon, self.elect.elect_elev]))
            pt_mtx = np.tile(pt, [len(pts), 1])
            dist = np.linalg.norm(pt_mtx - pts, axis=1, ord=2)
            ix = np.where(np.all(np.array([dist < inp.thresh, dist > 0]), 0))

            # delete closer electrodes
            self.elect.elect_lat[i] = np.mean(np.append(pt[0], self.elect.elect_lat[ix]))
            self.elect.elect_lon[i] = np.mean(np.append(pt[1], self.elect.elect_lon[ix]))
            self.elect.elect_elev[i] = np.mean(np.append(pt[2], self.elect.elect_elev[ix]))

            self.elect.elect_lat = np.delete(self.elect.elect_lat, ix)
            self.elect.elect_lon = np.delete(self.elect.elect_lon, ix)
            self.elect.elect_elev = np.delete(self.elect.elect_elev, ix)
            rem_num = self.elect.elect_number[ix]
            self.elect.elect_number = np.arange(len(self.elect.elect_number) - len(rem_num))

            for j in rem_num:
                self.quad.a[np.where(self.quad.a == j)] = el_num
                self.quad.a[np.where(self.quad.a > j)] = self.quad.a[np.where(self.quad.a > j)] - 1
                self.quad.b[np.where(self.quad.b == j)] = el_num
                self.quad.b[np.where(self.quad.b > j)] = self.quad.b[np.where(self.quad.b > j)] - 1
                self.quad.m[np.where(self.quad.m == j)] = el_num
                self.quad.m[np.where(self.quad.m > j)] = self.quad.m[np.where(self.quad.m > j)] - 1
                self.quad.n[np.where(self.quad.n == j)] = el_num
                self.quad.n[np.where(self.quad.n > j)] = self.quad.n[np.where(self.quad.n > j)] - 1

            # update index
            i = i + 1

        return

    def WriteData(self, url):
        # templ = '{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t'\
        #         '{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\n'
        templ = '{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t{:.8e}\t' \
                '{:.8e}\t{:.8e}\n'
        with open(url, 'w') as fid:
            # fid.write('XA\tYA\tZA\tXB\tYB\tZB\tXM\tYM\tZM\tXN\tYN\tZN\tV/A\tUNCERT\tLINEID\n')
            fid.write('XA\tYA\tZA\tXB\tYB\tZB\tXM\tYM\tZM\tXN\tYN\tZN\tV/A\tUNCERT\n')
            for ix in np.arange(len(self.quad.quad_number)):
                ixa = np.where(self.elect.elect_number == self.quad.a[ix])
                while hasattr(ixa, "__len__"):
                    ixa = ixa[0]
                ixb = np.where(self.elect.elect_number == self.quad.b[ix])
                while hasattr(ixb, "__len__"):
                    ixb = ixb[0]
                ixm = np.where(self.elect.elect_number == self.quad.m[ix])
                while hasattr(ixm, "__len__"):
                    ixm = ixm[0]
                ixn = np.where(self.elect.elect_number == self.quad.n[ix])
                while hasattr(ixn, "__len__"):
                    ixn = ixn[0]
                fid.write(templ.format(self.elect.elect_lat[ixa], self.elect.elect_lon[ixa], self.elect.elect_elev[ixa],
                                       self.elect.elect_lat[ixb], self.elect.elect_lon[ixb], self.elect.elect_elev[ixb],
                                       self.elect.elect_lat[ixm], self.elect.elect_lon[ixm], self.elect.elect_elev[ixm],
                                       self.elect.elect_lat[ixn], self.elect.elect_lon[ixn], self.elect.elect_elev[ixn],
                                       self.quad.rho[ix], self.quad.dev[ix]))


class Quad(object):
    def __init__(self, quad_number=np.empty(0), rho=np.empty(0), vp=np.empty(0), dev=np.empty(0), a=np.empty([0, 3]),
                 b=np.empty([0, 3]),
                 m=np.empty([0, 3]), n=np.empty([0, 3])):
        self.quad_number = quad_number
        self.rho = rho
        self.vp = vp
        self.dev = dev
        self.a = a
        self.b = b
        self.m = m
        self.n = n


class Elect(object):
    def __init__(self, elect_number=np.empty(0), elect_lat=np.empty(0), elect_lon=np.empty(0), elect_elev=np.empty(0)):
        self.elect_number = elect_number
        self.elect_lat = elect_lat
        self.elect_lon = elect_lon
        self.elect_elev = elect_elev


class OhmData(object):
    def __init__(self, rho=np.empty(0), time=np.empty(0), pt_num=np.empty(0), lat=np.empty(0), lon=np.empty(0),
                 elev=np.empty(0), line=np.empty(0), mark=np.empty(0), mark_num=np.empty(0), mark_time=np.empty(0),
                 mark_lat=np.empty(0), mark_lon=np.empty(0), mark_elev=np.empty(0), starting_time=0,
                 op_offset=np.empty(0), rec_dip=np.empty(0), rop_len=np.empty(0), tr_dip=np.empty(0)):
        self.rho = rho
        self.time = time
        self.pt_num = pt_num
        self.lat = lat
        self.lon = lon
        self.elev = elev
        self.line = line
        self.mark = mark
        self.mark_num = mark_num
        self.mark_time = mark_time
        self.mark_lat = mark_lat
        self.mark_lon = mark_lon
        self.mark_elev = mark_elev
        self.starting_time = starting_time
        self.op_offset = op_offset
        self.rec_dip = rec_dip
        self.rop_len = rop_len
        self.tr_dip = tr_dip

    def ReadFile(self, url):
        with open(url, 'r') as fid:
            lines = fid.readlines()

        i = len(lines) - 1
        row = np.array(lines[i][0:-1].split())
        if float(row[0]) != 3:
            raise Exception('Il file non finisce con un mark value!')

        self.starting_time = datetime.datetime(2000 + int(row[4][6:8]), int(row[4][0:2]), int(row[4][3:5]),
                                               int(row[3][0:2]), int(row[3][3:5]), int(row[3][6:8]),
                                               int(row[3][9:11]))
        mark = -1
        pt = -1
        row = np.array(lines[i][0:-1].split())
        pt_tmp = -1
        rho_tmp = np.empty(0)
        mark_tmp = np.empty(0)
        time_tmp = np.empty(0)
        pt_num_tmp = np.empty(0)
        line_tmp = np.empty(0)
        while i >= 0:
            if float(row[0]) == 3:

                mark = mark + 1
                self.mark_num = np.append(self.mark_num, mark)
                t = datetime.datetime(2000 + int(row[4][6:8]), int(row[4][0:2]), int(row[4][3:5]), int(row[3][0:2]),
                                      int(row[3][3:5]), int(row[3][6:8]), int(row[3][9:11]))
                self.mark_time = np.append(self.mark_time, (t - self.starting_time).total_seconds())
                line = float(row[6])

                if np.all(line_tmp == line):
                    self.rho = np.append(self.rho, rho_tmp)
                    self.mark = np.append(self.mark, mark_tmp)
                    self.time = np.append(self.time, time_tmp)
                    self.pt_num = np.append(self.pt_num, pt_num_tmp)
                    self.line = np.append(self.line, line_tmp)
                    pt = pt_tmp
                    rho_tmp = np.empty(0)
                    mark_tmp = np.empty(0)
                    time_tmp = np.empty(0)
                    pt_num_tmp = np.empty(0)
                    line_tmp = np.empty(0)
                pt_tmp = pt
            elif float(row[0]) == 0:
                pt_tmp = pt_tmp + 1
                rho_tmp = np.append(rho_tmp, float(row[1]))
                mark_tmp = np.append(mark_tmp, mark)
                t = datetime.datetime(2000 + int(row[4][6:8]), int(row[4][0:2]), int(row[4][3:5]), int(row[3][0:2]),
                                      int(row[3][3:5]), int(row[3][6:8]), int(row[3][9:11]))
                time_tmp = np.append(time_tmp, (t - self.starting_time).total_seconds())
                pt_num_tmp = np.append(pt_num_tmp, pt_tmp)
                line_tmp = np.append(line_tmp, line)
            elif float(row[0]) == 33 and row[1] == 'GOHM':
                self.op_offset = np.append(self.op_offset, float(row[2]))
                self.rec_dip = np.append(self.rec_dip, float(row[3]))
                self.rop_len = np.append(self.rop_len, float(row[4]))
                self.tr_dip = np.append(self.tr_dip, float(row[5]))

            i = i - 1
            row = np.array(lines[i][0:-1].split())
        if np.any(self.op_offset - self.op_offset[0] != 0):
            raise Exception('Operator offset cambia fra le linee!')
        else:
            self.op_offset = self.op_offset[0]
        if np.any(self.rec_dip - self.rec_dip[0] != 0):
            raise Exception('Rec dipole cambia fra le linee!')
        else:
            self.rec_dip = self.rec_dip[0]
        if np.any(self.rop_len - self.rop_len != 0):
            raise Exception('Rope_len offset cambia fra le linee!')
        else:
            self.rop_len = self.rop_len[0]
        if np.any(self.tr_dip - self.tr_dip[0] != 0):
            raise Exception('transmitter dipole cambia fra le linee!')
        else:
            self.tr_dip = self.tr_dip[0]
        return

    def LocalizeOhm(self, mark_gps):

        # allocate mark XyZ coord
        if len(mark_gps.id) == len(self.mark_num):
            self_ix = np.argsort(self.mark_num)
            mark_ix = np.argsort(mark_gps.id)
            self.mark_lat = mark_gps.lat[mark_ix[self_ix]]
            self.mark_lon = mark_gps.lon[mark_ix[self_ix]]
            self.mark_elev = mark_gps.elev[mark_ix[self_ix]]
        else:
            raise Exception('Numero di mark gps maggiore di quelli acquisiti!')

        flat = interpolate.interp1d(self.mark_time, self.mark_lat)
        flon = interpolate.interp1d(self.mark_time, self.mark_lon)
        felev = interpolate.interp1d(self.mark_time, self.mark_elev)
        self.lat = flat(self.time)
        self.lon = flon(self.time)
        self.elev = felev(self.time)

        return


class GpsData(object):
    def __init__(self, id_=np.empty(0), lon=np.empty(0), lat=np.empty(0), elev=np.empty(0)):
        self.id = id_
        self.lon = lon
        self.lat = lat
        self.elev = elev

    def ReadFile(self, url):
        with open(url, 'r') as fid:
            lines = fid.readlines()

        id_ = np.empty(0)
        lon = np.empty(0)
        lat = np.empty(0)
        elev = np.empty(0)
        for i in range(len(lines)):
            row = np.array((lines[i].replace(',', '.')).split())
            id_ = np.append(id_, float(re.findall(r'\d+', row[0])[0]))
            lon = np.append(lon, float(row[1]))
            lat = np.append(lat, float(row[2]))
            elev = np.append(elev, float(row[3]))
        x = np.arange(id_.min(), id_.max() + 1)
        flon = interpolate.interp1d(id_, lon)
        flat = interpolate.interp1d(id_, lat)
        felev = interpolate.interp1d(id_, elev)

        self.id = x
        self.lon = flon(x)
        self.lat = flat(x)
        self.elev = felev(x)

        return


def ReadIris(url):
    # check url consistency
    if not (os.path.isfile(url)):
        raise Exception("Filepath inserito errato!")

    # initialize data class
    data = SyscalData()
    data.measure = ErtMeasure()
    data.url = url
    # invoke reading function
    if url[-3:] == "bin":
        data.ReadBin()
    elif url[-3:] == "pro":
        data.ReadPro()
    else:
        raise Exception("Formato del file inserito non corretto. Inserire un file iris .bin o .pro ")

    return data


def ReadStn(url):
    # check url consistency
    if not (os.path.isfile(url)):
        raise Exception("Filepath inserito errato!")

    data = OhmData()
    data.ReadFile(url)

    return data


def ReadGps(url):
    # check url consistency
    if not (os.path.isfile(url)):
        raise Exception("Filepath inserito errato!")

    data = GpsData()
    data.ReadFile(url)

    return data


def ReadQuadConfig(url):
    # check url consistency
    if not (os.path.isfile(url)):
        raise Exception("Filepath inserito errato!")

    data = ElectData(typ='ERT')
    data.ReadErtConfig(url)

    return data


def WriteElevMesh(lat, lon, elev, url):
    # find minimum x and y step
    dist = ((lat - lat[:, None]) ** 2 + (lon - lon[:, None]) ** 2).ravel()
    dist = dist[dist > 0]
    dx = np.sqrt(np.min(dist))
    x = np.arange(min(lon), max(lon) + .01, dx)
    y = np.arange(min(lat), max(lat) + .01, dx)
    Xm, Ym = np.meshgrid(x, y)

    X = Xm.ravel()
    Y = Ym.ravel()

    # interp = interpolate.LinearNDInterpolator(list(zip(lon, lat)), elev)
    interp = interpolate.Rbf(lon, lat, elev, function="multiquadric", smooth=5)
    Zm = interp(Xm, Ym)
    Z = np.matrix.flatten(Zm)
    tmpl = '{:.4e} {:.4e} {:.4e}\n'
    with open(url, 'w') as fid:
        for i in np.arange(len(X)):
            fid.write(tmpl.format(X[i], Y[i], Z[i]))
