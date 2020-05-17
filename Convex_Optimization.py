import cvxpy as cv
import numpy as np

class Convex(object):
    def __init__(self):
        super(Convex, self).__init__()
        # Rotary-UAV propulsion energy model, defined in (1) of the paper;
        self.d_o = 0.6  # fuselage equivalent flat plate area;
        self.rho = 1.225  # air density in kg/m3;
        self.s = 0.05  # rotor solidity;
        self.G = 0.503  # Rotor disc area in m2;
        self.U_tip = 120  # tip seep of the rotor blade(m/s);
        self.v_o = 4.3  # mean rotor induced velocity in hover;
        self.omega = 300  # blade angular velocity in radians/second;
        self.R = 0.4  # rotor radius in meter;
        self.delta = 0.012  # profile drage coefficient;
        self.k = 0.1  # incremental correction factor to induced power;
        self.W = 20  # aircraft weight in newton;
        self.P0 = (self.delta / 8) * self.rho * self.s * self.G * (pow(self.omega, 3)) * (pow(self.R, 3))
        self.P1 = (1 + self.k) * (pow(self.W, (3 / 2)) / np.sqrt(2 * self.rho * self.G))

        self.Slot = 400
        self.V_h_max = 10
        self.V_h_min = 0
        self.V_v_max = 10
        self.t_max = 5
        self.t_min = 1
        self.E_max = 2000000
        self.theta = 0.3
        self._build_convex_optimization()

    def _build_convex_optimization(self):
        self.v_n_h = cv.Variable((1, self.Slot), nonneg=True)
        self.phi = cv.Variable((1, self.Slot),nonneg=True)

    def initial_op(self, UAV_trajectory, Task_offloading, t_kn,GTs,l_o_h,l_o_v,x_s,y_s,h_s,v_horizontal):
        self.horizontal = UAV_trajectory[:,[0,1]]
        self.vertical = UAV_trajectory[:,-1]
        self.offloading_l = Task_offloading
        self.t_kn = t_kn
        self.GTs = GTs
        self.l_o_h = np.zeros((1,2), dtype=np.float)
        self.l_o_h[0,0] = l_o_h[0] * x_s + 0.5 * x_s
        self.l_o_h[0,1] = l_o_h[1] * x_s + 0.5 * x_s
        self.l_o_v = l_o_v*y_s+0.5*y_s

        self.v_horizontal = v_horizontal

    def solve(self):
        v_horizontal_1 = np.zeros((1, self.Slot), dtype=np.float)
        v_horizontal_2 =  np.zeros((1, self.Slot), dtype=np.float)
        v_horizontal_3 = np.zeros((1, self.Slot), dtype=np.float)
        v_horizontal_min = np.zeros((1, self.Slot), dtype=np.float)
        v_horizontal_max = np.zeros((1, self.Slot), dtype=np.float)

        distance_h = np.zeros((1, self.Slot), dtype=np.float)
        distance_v = np.zeros((1, self.Slot), dtype=np.float)

        t_optimal = np.zeros((1, self.Slot), dtype=np.float)

        for i in range(self.Slot):
            if (i==0):
                distance_h[0,i] = np.sqrt(np.power(self.horizontal[i, 0] - self.l_o_h[0,0], 2) + np.power(self.horizontal[i, 1] - self.l_o_h[0,1], 2))
                distance_v[0,i] = np.abs(self.vertical[i]-self.vertical[0])
                if (distance_v[0, i] == 0):
                    distance_v[0, i] = 0.003
            else:
                distance_h[0,i] = np.sqrt(np.power(self.horizontal[i, 0] - self.horizontal[i-1, 0], 2) + np.power(self.horizontal[i, 1] - self.horizontal[i-1, 1], 2))
                distance_v[0,i] = np.abs(self.vertical[i] - self.vertical[i - 1])
                if (distance_v[0, i] == 0):
                    distance_v[0, i] = 0.003

            v_horizontal_min[0, i] = distance_h[0,i]/ self.t_max
            d_f = distance_h[0,i]
            if (d_f==0):
                #d_f = self.t_max*self.V_h_max
                d_f = np.inf
            v_horizontal_2[0, i]= self.V_v_max*d_f/distance_v[0,i]
            v_horizontal_3[0, i] =  d_f / self.t_min


            #maxt =np.sum(self.t_kn[:, i])/self.GTs
            maxt = 0
            for g in range(self.GTs):
                if (maxt < self.t_kn[g, i]):
                    maxt = self.t_kn[g, i]
            if (maxt>0):
                v_horizontal_1[0, i] = d_f / maxt
            else:
                v_horizontal_1[0, i] = np.inf

        min_max=np.inf
        max_min=0
        for i in range(self.Slot):
            v_horizontal_max[0, i]=np.min([v_horizontal_1[0,i],v_horizontal_2[0,i],v_horizontal_3[0,i],self.V_h_max])
            v_horizontal_min[0, i] = np.max([v_horizontal_min[0, i], self.V_h_min])
            if (v_horizontal_max[0, i]<min_max):
                min_max=v_horizontal_max[0, i]
            if (v_horizontal_min[0, i] > max_min):
                max_min = v_horizontal_min[0, i]

        phi_l = np.sqrt(np.sqrt(1 + np.power(self.v_horizontal, 4) / (4 * np.power(self.v_o, 4))) - np.power(self.v_horizontal, 2)/(2 * np.power(self.v_o, 2)))
        temp = self.P0 * (1 + 3 * cv.power(self.v_n_h, 2) / cv.power(self.U_tip, 2)) + (
                1 / 2) * self.d_o * self.rho * self.s * self.G * cv.power(self.v_n_h, 3) + self.P1 * self.phi
        e_m_phi = temp
        X_bl = (4 * cv.power(phi_l, 3) + 2 * phi_l * (cv.power(self.v_horizontal, 2)).T / cv.power(self.v_o, 2)) * self.phi.T \
               - 3 * cv.power(phi_l, 4) - cv.power(phi_l, 2) * (cv.power(self.v_horizontal, 2)).T / cv.power(self.v_o, 2)

        obj = cv.Minimize(cv.max(e_m_phi))
        constraints = [X_bl >= 1, self.v_n_h <= v_horizontal_max, self.v_n_h >= v_horizontal_min]
        prob = cv.Problem(obj, constraints)
        prob.solve(verbose=False)
        v_optimal = self.v_n_h.value
        #print(v_optimal)

        for i in range(self.Slot):
            t_optimal[0,i] =  distance_h[0,i]/v_optimal[0,i]
            if (t_optimal[0,i]==0):
                t_optimal[0, i]= self.t_max

        #print(t_optimal)
        print("status:", prob.status)
        return [v_optimal,t_optimal]
