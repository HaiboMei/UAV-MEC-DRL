#-- coding:UTF-8 --
"""
Reinforcement learning UAV-enabled MEC example.
Basical setting:
AoI:                100m*100m;
Height_levels, maximal height, minimula height:
                    20, 100m, 200m;

This script is the environment part of the UAV-enabled MEC.
The RL is in RL_brain.py.

View more on my information see paper: "Deep Reinforcement Learning based 3D-Trajectory Design and Task Offloading in UAV-enabled MEC System"
by Haibo Mei, Kun Yang, Qiang Liu;
"""
import numpy as np
import random as rd
import time
import math as mt
import sys
import copy
from scipy.interpolate import spline
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk
import matplotlib.pyplot as plt
from TSP import tsp
tsp=tsp()

UNIT =  1              # pixels
IOT_H = 100           # grid height
IOT_W = 100           # grid width
Max_Hight = 100        # maximum level of height
Min_Hight = 50         # minimum level of height


#weight variables for the reward function
beta = 20

#gradent of the horizontal and vertical locations of the UAV
D_k = 200
F_k = 2000
 # 2M(2000), 2000 cycles

# Initialize the wireless environement and some other verables.
N_0 = mt.pow(10, ((-169 / 3) / 10))  # Noise power spectrum density is -169dBm/Hz;
a = 9.61  # referenced from paper [Efficient 3-D Placement of an Aerial Base Station in Next Generation Cellular Networks, 2016, ICC]
b = 0.16  # and paper [Optimal LAP Altitude for Maximum Coverage, IEEE WIRELESS COMMUNICATIONS LETTERS, VOL. 3, NO. 6, DECEMBER 2014]
eta_los = 1  # Loss corresponding to the LoS connections defined in (2) of the paper;
eta_nlos = 20  # Loss corresponding to the NLoS connections defined in (2)of the paper;
A = eta_los - eta_nlos  # A varable defined in (2) of the paper;
C = 20 * np.log10(
    4 * np.pi * 9 / 3) + eta_nlos  # C varable defined in (2)of the paper, where carrier frequncy is 900Mhz=900*10^6, and light speed is c=3*10^8; then one has f/c=9/3;
B = 2000  # overall Bandwith is 2Gb;
Power = 5 * mt.pow(10, 5)  # maximum uplink transimission power of one GT is 5mW;

class UAV_MEC(object):
    def __init__(self):
        super(UAV_MEC, self).__init__()
        self.N_slot = 400  # number of time slots in one episode
        self.x_s = 10
        self.y_s = 10
        self.h_s = 2
        self.GTs = 6
        self.l_o_v = 50*self.h_s  # initial vertical location
        self.l_f_v = 50*self.h_s  # final vertical location
        self.l_o_h = [0, 0]  # initial horizontal location
        self.l_f_h = [0, 0]  # final horizontal location
        self.eps = 60   #number of episode

        self.UAV_trajectory_tsp = np.zeros((self.N_slot, 3), dtype=np.float)

        #cycles / s,
        #cycles / s;
        self.f_u = 100
        self.f_g = 5
        self.D_max = np.sqrt(mt.pow(self.l_o_h[0]*self.x_s - self.l_f_h[0]*self.x_s, 2) + mt.pow(self.l_o_h[1]*self.y_s - self.l_f_h[1]*self.y_s,2))  # the distance from initial point to final point

        # north, south, east, west, hover
        self.action_space_uav_horizontal = ['n', 's', 'e','w','h']
        # ascend, descend, slf
        self.action_space_uav_vertical = ['a', 'd', 's']
        # offloading, local exection
        self.action_space_task_offloading = np.zeros((self.GTs, 2), dtype=np.int)
        #overall_action_space
        self.n_actions  = len(self.action_space_uav_horizontal)*len(self.action_space_uav_vertical)*mt.pow(2,self.GTs)
        self.n_features = 3   #horizontal:x, y, vertical trajectory of the UAV

        #generate action table;
        self.actions = np.zeros((np.int(self.n_actions),1+2+self.GTs), dtype=np.int)
        index = 0
        for h in range(len(self.action_space_uav_horizontal)):
            for v in range(len(self.action_space_uav_vertical)):
                LL= self.brgd(self.GTs)  #list all the possible combination of 0-1 offloading options among the GTs
                for l in range (len(LL)):
                    o_string = LL[l]
                    of = []
                    for ch in range (len(o_string)):
                        if o_string[ch] == '0':
                            of.append(0)
                        else:
                            of.append(1)
                    self.actions[index,:]=[index, h, v]+ of[:]
                    index = index + 1

        self._build_uav_mec()

    def _build_uav_mec(self):
        # initilize the GT coordinates and tasks
        # model of GTs' location and task
        self.location = np.zeros((5, 2), dtype=np.float)
        self.location[0, :] = [rd.randint(0, IOT_H/3), rd.randint(0, IOT_W/3)]
        self.location[1, :] = [rd.randint(IOT_H/3, 2*IOT_H/3), rd.randint(IOT_W/3, 2*IOT_W/3)]
        self.location[2, :] = [rd.randint(2*IOT_H/3, 3*IOT_H/3), rd.randint(2*IOT_W/3, 3*IOT_W/3)]
        self.location[0, :] = [rd.randint(0, 2*IOT_H/3), rd.randint(0, 2*IOT_W/3)]
        self.location[1, :] = [rd.randint(0, 2*IOT_H/3), rd.randint(0, 2*IOT_W/3)]
        self.location[2, :] = [rd.randint(0, 2*IOT_H/3), rd.randint(0, 2*IOT_W/3)]

        self.w_k = np.zeros((self.GTs, 2), dtype=np.float)
        self.u_k = np.zeros((self.GTs, 2), dtype=np.float)

        for count in range(3):  #3*2=6;
            for cou in range(2):
                g = count * 2 + cou
                # horizontal coordinate of the GT
                self.w_k[g, 0] = self.location[count, 0] + rd.randint(20, 40)
                self.w_k[g, 1] = self.location[count, 1] + rd.randint(20, 40)

                self.w_k[g, 0] = self.w_k[g, 0] * self.x_s + self.x_s * rd.random()
                self.w_k[g, 1] = self.w_k[g, 1] * self.y_s + self.y_s * rd.random()
                # D_k of the GT task
                self.u_k[g, 0] = D_k / 2 + (D_k / 2) * rd.random()
                # F_k of the GT task
                self.u_k[g, 1] = F_k / 2 + (F_k / 2) * rd.random()

        #initial UAV trajectory using TSP
        w_k_tmp = np.zeros((self.GTs+1, 2), dtype=np.float)
        for g in range(self.GTs):
            w_k_tmp[g,0]=self.w_k[g, 0]
            w_k_tmp[g,1]=self.w_k[g, 1]
        w_k_tmp[self.GTs, 0] = 0
        w_k_tmp[self.GTs, 1] = 0

        [tsp_result,sum_tsp_path]=tsp.solve(w_k_tmp)
        Delta_distance = sum_tsp_path /self.N_slot
        UAV_trajectory_tsp_tmp = np.zeros((self.N_slot, 3), dtype=np.float)

        UAV_trajectory_tsp_tmp[0,0] = w_k_tmp[tsp_result[0], 0]
        UAV_trajectory_tsp_tmp[0,1] = w_k_tmp[tsp_result[0], 1]
        UAV_trajectory_tsp_tmp[0,2] = 200
        count = 0
        text_dist=0
        for n in range(0, self.GTs):
            current_note = tsp_result[n]
            next_note = tsp_result[n + 1]
            dis = np.sqrt((w_k_tmp[current_note, 0] - w_k_tmp[next_note, 0])**2 + (w_k_tmp[current_note, 1] - w_k_tmp[next_note, 1])**2)
            text_dist = text_dist + dis
            line_count =np.int(np.floor_divide(dis,Delta_distance))
            if (np.remainder(dis,Delta_distance)>0):
                line_count+=1
            x_dis = w_k_tmp[next_note, 0] - w_k_tmp[current_note, 0]
            y_dis = w_k_tmp[next_note, 1] - w_k_tmp[current_note, 1]
            delta_x_dis = x_dis / line_count
            delta_y_dis = y_dis / line_count
            for index in range(1,line_count):
                UAV_trajectory_tsp_tmp[count + 1,0]= UAV_trajectory_tsp_tmp[count,0] + delta_x_dis
                UAV_trajectory_tsp_tmp[count + 1,1]= UAV_trajectory_tsp_tmp[count,1] + delta_y_dis
                if (UAV_trajectory_tsp_tmp[count + 1,0]<0):
                    UAV_trajectory_tsp_tmp[count + 1,0]=0
                if (UAV_trajectory_tsp_tmp[count + 1,1]<0):
                    UAV_trajectory_tsp_tmp[count + 1,1]=0
                UAV_trajectory_tsp_tmp[count + 1, 2] =200
                #if (count<=(self.N_slot/2)):
                    #UAV_trajectory_tsp_tmp[count + 1,2] = UAV_trajectory_tsp_tmp[count,2]-(200.00/np.float(self.N_slot))
                #else:
                    #UAV_trajectory_tsp_tmp[count + 1, 2] =  UAV_trajectory_tsp_tmp[count,2]+(200.00/np.float(self.N_slot))
                count = count + 1
        while (count<(self.N_slot-1)):
            UAV_trajectory_tsp_tmp[count+1, 0]=0
            UAV_trajectory_tsp_tmp[count+1, 1]=0
            UAV_trajectory_tsp_tmp[count+1, 2]=  200
            count=count+1

        for i in range(self.N_slot):
            self.UAV_trajectory_tsp[i, 0] = UAV_trajectory_tsp_tmp[self.N_slot-1-i, 0]
            self.UAV_trajectory_tsp[i, 1] = UAV_trajectory_tsp_tmp[self.N_slot-1-i, 1]
            self.UAV_trajectory_tsp[i, 2] = UAV_trajectory_tsp_tmp[self.N_slot-1-i, 2]
        self.plot_UAV_TSP(self.UAV_trajectory_tsp)
        return

    def reset(self):
        #reset the UAV trajectory
        self.h_n = 100
        self.l_n = [0, 0]
        return np.array([self.l_n[0], self.l_n[1], self.h_n])

    def link_rate (self, gt):
        h = self.h_n * self.h_s
        x = self.l_n[0]*self.x_s+0.5*self.x_s
        y = self.l_n[1]*self.y_s+0.5*self.y_s
        d = np.sqrt(mt.pow(h, 2) + mt.pow(x- self.w_k[gt,0],2) + mt.pow(y- self.w_k[gt,1],2))

        if (np.sqrt(mt.pow(x- self.w_k[gt,0], 2) + mt.pow(y- self.w_k[gt,1], 2))>0):
            ratio = h / np.sqrt(mt.pow(x - self.w_k[gt, 0], 2) + mt.pow(y - self.w_k[gt, 1], 2))
        else:
            ratio = np.Inf

        p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
        p_los = 1 / p_los
        L_km = 20 * np.log10(d) + A * p_los + C
        r = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
        return r

    def link_rate_single (self, h, x, y, w_k):
        d = np.sqrt(mt.pow(h, 2) + mt.pow(x- w_k[0],2) + mt.pow(y- w_k[1],2))
        if (np.sqrt(mt.pow(x- w_k[0], 2) + mt.pow(y- w_k[1], 2))>0):
            ratio = h / np.sqrt(mt.pow(x- w_k[0], 2) + mt.pow(y- w_k[1], 2))
        else:
            ratio = np.Inf
        p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
        p_los = 1 / p_los
        L_km = 20 * np.log10(d) + A * p_los + C
        r = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
        return r

    def step (self, action,t_n,slot):
        h = action[1]
        v = action[2]

        pre_l_n = self.l_n
        pre_h_n = self.h_n

        #update height of the UAV
        self.OtPoI = 0
        if v == 0:   # ascending
            self.h_n= self.h_n + 1
            if self.h_n>Max_Hight:
                self.h_n = self.h_n - 1
                self.OtPoI = 1
        elif v == 1:   # descending
            self.h_n= self.h_n - 1
            if self.h_n<Min_Hight:
                self.h_n = self.h_n + 1
                self.OtPoI = 1
        elif v == 2:   # SLF
            self.h_n = self.h_n

        #update horizontal location of the UAV
        if h == 0:  # north
            self.l_n[1] = self.l_n[1] + 1
            if self.l_n[1]> IOT_H: #if out of PoI
                self.l_n[1]= self.l_n[1] - 1
                self.OtPoI = 1
        elif h == 1:  # south
            self.l_n[1] = self.l_n[ 1] - 1
            if self.l_n[1]< 0:  #if out of PoI
                self.l_n[1]= self.l_n[1] + 1
        elif h == 2:  # east
            self.l_n[0] = self.l_n[0] + 1
            if self.l_n[0] > IOT_W:  # if out of PoI
                self.l_n[0] = self.l_n[0] - 1
                self.OtPoI = 1
        elif h == 3:   # west
           self.l_n[0] = self.l_n[0] - 1
           if self.l_n[0] < 0:  # if out of PoI
                self.l_n[0] = self.l_n[0] + 1
                self.OtPoI = 1
        elif h == 4:   #hover
           self.l_n[0] = self.l_n[0]
           self.l_n[1] = self.l_n[1]

        a_kn = np.zeros((1, self.GTs), dtype=np.int)
        r_kn = np.zeros((1, self.GTs), dtype=np.float)  #data of the uplink of the UAV-GT links
        d_s  = np.zeros((1, self.GTs), dtype=np.float)  #data process speed given the offloading strategy, UAV and GTs' locations
        engy = self.flight_energy_slot(pre_l_n,self.l_n,pre_h_n,self.h_n,t_n)

        for g in range(self.GTs):
            a_kn[0,g] = action[1+2+g]
            r_kn[0,g] = self.link_rate(g)
            d_s[0,g] = (t_n/self.u_k[g,0])*(a_kn[0,g]*((self.f_u*self.u_k[g,0]*r_kn[0,g])/(r_kn[0,g]*self.u_k[g,1]+self.f_u*self.u_k[g,0]))+(1-a_kn[0,g])*self.f_g)

        sum_d_s= np.sum(d_s)
        reward = sum_d_s/engy
        if self.OtPoI == 1:
            reward = reward - 0.001 #give an additional penality if out of PoI: P=0.3
        _state = np.array([self.l_n[0],self.l_n[1], self.h_n])
        return _state, reward

    def find_action(self, index):
        return self.actions[index,:]

    def brgd(self, n):
        if n == 1:
            return ["0", "1"]
        L1 = self.brgd (n - 1)
        L2 = copy.deepcopy(L1)
        L2.reverse()
        L1 = ["0" + l for l in L1]
        L2 = ["1" + l for l in L2]
        L = L1 + L2
        return L

    def flight_energy(self,UAV_trajectory,UAV_flight_time,EP):
        d_o = 0.6  # fuselage equivalent flat plate area;
        rho = 1.225  # air density in kg/m3;
        s = 0.05  # rotor solidity;
        G = 0.503  # Rotor disc area in m2;
        U_tip = 120  # tip seep of the rotor blade(m/s);
        v_o = 4.3  # mean rotor induced velocity in hover;
        omega = 300  # blade angular velocity in radians/second;
        R = 0.4  # rotor radius in meter;
        delta = 0.012  # profile drage coefficient;
        k = 0.1  # incremental correction factor to induced power;
        W = 20  # aircraft weight in newton;
        P0 = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
        P1 = (1 + k) * (pow(W, (3 / 2)) / np.sqrt(2 * rho * G))
        Energy_uav = np.zeros((EP, self.N_slot), dtype=np.float)
        P2 =11.46
        count =0
        for ep in range(self.eps-EP,self.eps):
            horizontal = UAV_trajectory[ep,:, [0, 1]]
            vertical = UAV_trajectory[ep,:, -1]
            t_n=UAV_flight_time[ep,:]

            for i in range(self.N_slot):
                if (i==0):
                    d = np.sqrt((horizontal[0,i] - self.l_o_h[0])**2 + (horizontal[1,i] - self.l_o_h[1])**2)
                    h = np.abs(vertical[i]-vertical[0])
                else:
                    d = np.sqrt((horizontal[0,i] - horizontal[0,i-1])**2 + (horizontal[1,i] - horizontal[1,i-1])**2)
                    h = np.abs(vertical[i] - vertical[i - 1])

                v_h = d/t_n[i]
                v_v = h/t_n[i]
                Energy_uav[count, i] = t_n[i] * P0 * (1 + 3 * np.power(v_h, 2) / np.power(U_tip,2)) + t_n[i] * (1 / 2) * d_o * rho * s * G * np.power(v_h,3) +\
                               t_n[i] * P1 * np.sqrt(np.sqrt(1 + np.power(v_h, 4) / (4 * np.power(v_o,4))) - np.power(v_h, 2) / (2 * np.power(v_o,2))) + P2*v_v * t_n[i]
            count=count+1
        return Energy_uav

    def flight_energy_slot(self,pre_l_n,l_n,pre_h,h,t_n):
        d_o = 0.6  # fuselage equivalent flat plate area;
        rho = 1.225  # air density in kg/m3;
        s = 0.05  # rotor solidity;
        G = 0.503  # Rotor disc area in m2;
        U_tip = 120  # tip seep of the rotor blade(m/s);
        v_o = 4.3  # mean rotor induced velocity in hover;
        omega = 300  # blade angular velocity in radians/second;
        R = 0.4  # rotor radius in meter;
        delta = 0.012  # profile drage coefficient;
        k = 0.1  # incremental correction factor to induced power;
        W = 20  # aircraft weight in newton;
        P0 = (delta / 8) * rho * s * G * (pow(omega, 3)) * (pow(R, 3))
        P1 = (1 + k) * (pow(W, (3 / 2)) / np.sqrt(2 * rho * G))
        P2 =11.46

        x_pre = pre_l_n[0] * self.x_s + 0.5 * self.x_s
        y_pre = pre_l_n[1] * self.y_s + 0.5 * self.y_s
        z_pre = pre_h * self.h_s
        x = l_n[0] * self.x_s + 0.5 * self.x_s
        y = l_n[1] * self.y_s + 0.5 * self.y_s
        z = h * self.h_s

        d = np.sqrt((x_pre - x)**2 +(y_pre - y)**2)
        h = np.abs(z_pre-z)
        v_h = d/t_n
        v_v = h/t_n
        Energy_uav = t_n* P0 * (1 + 3 * np.power(v_h, 2) / np.power(U_tip,2)) + t_n * (1 / 2) * d_o * rho * s * G * np.power(v_h,3) +\
                               t_n * P1 * np.sqrt(np.sqrt(1 + np.power(v_h, 4) / (4 * np.power(v_o,4))) - np.power(v_h, 2) / (2 * np.power(v_o,2))) + P2*v_v * t_n
        return Energy_uav

    def UAV_FLY(self, UAV_trajectory):
        for slot in range(self.N_slot):
            UAV_trajectory[slot, 0] = UAV_trajectory[slot,0] * self.x_s + 0.5 * self.x_s
            UAV_trajectory[slot, 1] = UAV_trajectory[slot,1] * self.y_s + 0.5 * self.y_s
            UAV_trajectory[slot, 2] = UAV_trajectory[slot,2] * self.h_s

        for slot in range(2,self.N_slot):
            diff = np.abs( UAV_trajectory[slot,0]- UAV_trajectory[slot-2,0])+np.abs( UAV_trajectory[slot,1]- UAV_trajectory[slot-2,1])
            if (diff>self.x_s):
                UAV_trajectory[slot - 1, 0]= (UAV_trajectory[slot-2,0]+ UAV_trajectory[slot,0])/2
                UAV_trajectory[slot - 1, 1] = (UAV_trajectory[slot - 2, 1] + UAV_trajectory[slot, 1]) / 2
        return UAV_trajectory

    def throughput(self, UAV_trajectorys, UAV_flight_time, Task_offloadings,EP):
        through = np.zeros((EP,self.N_slot),dtype=np.float)
        rate = np.zeros((EP, self.N_slot), dtype=np.float)
        count=0
        for ep in range(self.eps - EP, self.eps):
            r_kn = np.zeros((self.N_slot, self.GTs), dtype=np.float)  # data of the uplink of the UAV-GT links
            t_n=UAV_flight_time[ep,:]
            UAV_trajectory=UAV_trajectorys[ep,:]
            Task_offloading=Task_offloadings[ep,:]
            for i in range(self.N_slot):
                for g in range(self.GTs):
                    h = UAV_trajectory[i,2]
                    x = UAV_trajectory[i,0]
                    y = UAV_trajectory[i,1]
                    a_kn = Task_offloading[i,g]

                    d = np.sqrt(mt.pow(h, 2) + mt.pow(x - self.w_k[g,0], 2) + mt.pow(y - self.w_k[g,1], 2))
                    if (np.sqrt(mt.pow(x - self.w_k[g,0], 2) + mt.pow(y - self.w_k[g,1], 2))>0):
                        ratio = h / np.sqrt(mt.pow(x - self.w_k[g, 0], 2) + mt.pow(y - self.w_k[g, 1], 2))
                    else:
                        ratio = np.Inf
                    p_los = 1 + a * mt.pow(np.exp(1), (a * b - b * np.arctan(ratio) * (180 / np.pi)))
                    p_los = 1 / p_los
                    L_km = 20 * np.log10(d) + A * p_los + C
                    r_kn[i,g] = B * np.log2(1 + Power * mt.pow(10, (-L_km / 10)) / (B * N_0))
                    rate[count,i] = rate[count,i]+r_kn[i,g]
                    through[count,i]=through[count,i]+t_n[i]*(a_kn*(self.f_u*self.u_k[g,0]*r_kn[i,g])/(r_kn[i,g]*self.u_k[g,1]+self.f_u*self.u_k[g,0])+self.f_g*(1-a_kn))
            count=count+1
        return through, rate

    def plot_UAV_TSP(self, UAV_trajectory):
        myfont = matplotlib.font_manager.FontProperties(
        fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
        x = []
        y = []
        z = []
        for slot in range(self.N_slot):
            #if (slot%100 == 0):
            x.append(UAV_trajectory[slot,0])
            y.append(UAV_trajectory[slot,1])
            z.append(UAV_trajectory[slot,2])

        plt.plot(x[:], y[:], c='b', label=u"TSP")
        plt.scatter(self.w_k[:, 0], self.w_k[:, 1], c='g', marker='x',label=u"GT Locations")
        plt.ylabel(u'x(m)', fontProperties=myfont)
        plt.xlabel(u'y(m)', fontProperties=myfont)
        plt.legend(prop=myfont)
        plt.grid()
        plt.show()

