import cvxpy as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import UAV_MEC_env as env
class Res_plot(object):
    def __init__(self,env):
        super(Res_plot, self).__init__()
        self.env = env
        self._build_result()

    def _build_result(self):
        return

    def plot_UAV_GT (self,w_k, UAV_trajectory_natural, UAV_trajectory_dqn, UAV_trajectory_tsp):
        myfont = matplotlib.font_manager.FontProperties(
        fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
        for e in range((self.env.eps-5),self.env.eps):
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            x_natural = []
            y_natural = []
            z_natural = []

            x_dqn = []
            y_dqn = []
            z_dqn = []

            x_tsp = []
            y_tsp = []
            z_tsp = []
            for slot in range(self.env.N_slot):
                #if (slot%100 == 0):
                x_natural.append(UAV_trajectory_natural[e,slot,0])
                y_natural.append(UAV_trajectory_natural[e,slot,1])
                z_natural.append(UAV_trajectory_natural[e,slot,2])

                x_dqn.append(UAV_trajectory_dqn[e, slot, 0])
                y_dqn.append(UAV_trajectory_dqn[e, slot, 1])
                z_dqn.append(UAV_trajectory_dqn[e, slot, 2])

                x_tsp.append(UAV_trajectory_tsp[e,slot, 0])
                y_tsp.append(UAV_trajectory_tsp[e,slot, 1])
                z_tsp.append(UAV_trajectory_tsp[e,slot, 2])

            ax.scatter(w_k[:, 0], w_k[:, 1], c='r', marker='x',label=u"GT locations")
            ax.plot(x_dqn[:], y_dqn[:], z_dqn[:], c='g',linestyle='-', marker='', label=u"Double DQN")
            ax.plot(x_natural[:], y_natural[:], z_natural[:], c='b', linestyle='--', marker='',label=u"Natural DQN")
            ax.plot(x_tsp[:], y_tsp[:], z_tsp[:], c='r',linestyle='-', marker='', label=u"TSP")
            ax.set_zlim(0, 250)
            ax.set_xlim(0, 1000)
            ax.set_ylim(0, 1000)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            plt.legend(prop=myfont)
            plt.show()

            plt.plot(x_dqn[:], y_dqn[:],c='g',linestyle='-', marker='', label=u"Double DQN")
            plt.plot(x_natural[:], y_natural[:],c='b',linestyle='--', marker='',label=u"Natural DQN")
            plt.plot(x_tsp[:], y_tsp[:],c='r',linestyle='-', marker='',label=u"TSP")
            plt.scatter(w_k[:, 0], w_k[:, 1], c='k', marker='x',label=u"GT Locations")
            plt.ylabel(u'x(m)', fontProperties=myfont)
            plt.xlabel(u'y(m)', fontProperties=myfont)
            plt.legend(prop=myfont)
            plt.grid()
            plt.show()
        return

    def plot_propulsion_energy(self,UAV_trajectory_tsp,UAV_trajectory_natural,UAV_trajectory_dqn,UAV_flight_time_tsp,UAV_flight_time_natural,UAV_flight_time_dqn,eps):
        PEnergy_tsp=self.env.flight_energy(UAV_trajectory_tsp,UAV_flight_time_tsp,eps)
        PEnergy_dqn = self.env.flight_energy(UAV_trajectory_natural, UAV_flight_time_natural,eps)
        PEnergy_ddqn = self.env.flight_energy(UAV_trajectory_dqn, UAV_flight_time_dqn,eps)

        plot_energy = np.zeros((3,eps),dtype=np.float)
        for i in range(eps):
            plot_energy[0,i] = plot_energy[0,i]+np.sum(PEnergy_tsp[i,:])
            plot_energy[1, i] = plot_energy[1, i] + np.sum(PEnergy_dqn[i, :])
            plot_energy[2, i] = plot_energy[2, i] + np.sum(PEnergy_ddqn[i, :])

        myfont = matplotlib.font_manager.FontProperties(
            fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
        plt.plot(np.arange(eps), plot_energy[0,:].T, c='r', linestyle='-', marker='<',label=u"TSP")
        plt.plot(np.arange(eps), plot_energy[1,:].T, c='b', linestyle='-', marker='>',label=u"Natural DQN")
        plt.plot(np.arange(eps), plot_energy[2,:].T, c='g', linestyle='-', marker='o',label=u"Double DQN")
        plt.xlabel(u'Episode', fontProperties=myfont)
        plt.ylabel(u'Propulsion Energy(J)', fontProperties=myfont)
        plt.legend(prop=myfont)
        plt.grid()
        plt.show()

        plt.plot(np.arange(eps), plot_energy[1, :].T, c='b', linestyle='-', marker='>',label=u"Natural DQN")
        plt.plot(np.arange(eps), plot_energy[2, :].T, c='g', linestyle='-', marker='o',label=u"Double DQN")
        plt.xlabel(u'Episode', fontProperties=myfont)
        plt.ylabel(u'Propulsion Energy(J)', fontProperties=myfont)
        plt.legend(prop=myfont)
        plt.grid()
        plt.show()

        sum_tsp = np.sum(plot_energy[0,:])/eps
        sum_dqn = np.sum(plot_energy[1,:])/eps
        sum_ddqn = np.sum(plot_energy[2,:])/eps
        print("Propulsion Energy: TSP:%f;Natural DQN:%f;D-DQN:%f" %(sum_tsp, sum_dqn,sum_ddqn))
        return

    def plot_data_throughput(self, UAV_trajectory_tsp, UAV_trajectory_natural, UAV_trajectory_dqn, UAV_flight_time_tsp, UAV_flight_time_natural,UAV_flight_time_dqn,Task_offloading_tsp,Task_offloading_natural,Task_offloading_dqn,eps):
        [Th_tsp,rate_tsp] = self.env.throughput(UAV_trajectory_tsp, UAV_flight_time_tsp,Task_offloading_tsp,eps)
        [Th_dqn,rate_dqn] = self.env.throughput(UAV_trajectory_natural, UAV_flight_time_natural,Task_offloading_natural,eps)
        [Th_ddqn,rate_ddqn] = self.env.throughput(UAV_trajectory_dqn, UAV_flight_time_dqn,Task_offloading_dqn,eps)

        plot_Th = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_Th[0, i] = plot_Th[0, i] + np.sum(Th_tsp[i, :])/1000
            plot_Th[1, i] = plot_Th[1, i] + np.sum(Th_dqn[i, :])/1000
            plot_Th[2, i] = plot_Th[2, i] + np.sum(Th_ddqn[i, :])/1000

        plot_Dr = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_Dr[0, i] = plot_Dr[0, i] + np.sum(rate_tsp[i, :])/1000
            plot_Dr[1, i] = plot_Dr[1, i] + np.sum(rate_dqn[i,  :])/1000
            plot_Dr[2, i] = plot_Dr[2, i] + np.sum(rate_ddqn[i, :])/1000

        plot_Dr=plot_Dr/eps

        myfont = matplotlib.font_manager.FontProperties(
            fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
        plt.plot(range(eps), plot_Th[0,:].T, c='r', linestyle='-', marker='<',label=u"TSP")
        plt.plot(range(eps), plot_Th[1,:].T, c='b', linestyle='-', marker='>',label=u"Natural DQN")
        plt.plot(range(eps), plot_Th[2,:].T, c='g', linestyle='-', marker='o',label=u"Double DQN")
        plt.xlabel(u'Episode', fontProperties=myfont)
        plt.ylabel(u'Throughput(Kbs)', fontProperties=myfont)
        plt.legend(prop=myfont)
        plt.grid()
        plt.show()

        plt.plot(range(eps), plot_Dr[0, :].T, c='r',linestyle='-', marker='<', label=u"TSP")
        plt.plot(range(eps), plot_Dr[1, :].T, c='b', linestyle='-', marker='>',label=u"Natural DQN")
        plt.plot(range(eps), plot_Dr[2, :].T, c='g', linestyle='-', marker='o',label=u"Double DQN")
        plt.xlabel(u'Episode', fontProperties=myfont)
        plt.ylabel(u'Data Rate(kbps)', fontProperties=myfont)
        plt.legend(prop=myfont)
        plt.grid()
        plt.show()


        sum_tsp = np.sum(plot_Th[0,:])/eps
        sum_dqn = np.sum(plot_Th[1,:])/eps
        sum_ddqn =np.sum(plot_Th[2,:])/eps
        print("Average Throughput: TSP:%f;Natural DQN:%f;D-DQN:%f" %(sum_tsp,sum_dqn,sum_ddqn))

        ave_tsp = np.sum(plot_Dr[0,:])/eps
        ave_dqn = np.sum(plot_Dr[1,:])/eps
        ave_ddqn = np.sum(plot_Dr[2,:])/eps
        print("Average Data Rate: TSP:%f;Natural DQN:%f;D-DQN:%f" %(ave_tsp,ave_dqn,ave_ddqn))
        return

    def plot_energy_efficiency(self, UAV_trajectory_tsp, UAV_trajectory_natural, UAV_trajectory_dqn, UAV_flight_time_tsp, UAV_flight_time_natural,UAV_flight_time_dqn,Task_offloading_tsp,Task_offloading_natural,Task_offloading_dqn,eps):
        [Th_tsp,rate_tsp] = self.env.throughput(UAV_trajectory_tsp, UAV_flight_time_tsp,Task_offloading_tsp,eps)
        [Th_dqn,rate_dqn] = self.env.throughput(UAV_trajectory_natural, UAV_flight_time_natural,Task_offloading_natural,eps)
        [Th_ddqn,rate_ddqn] = self.env.throughput(UAV_trajectory_dqn, UAV_flight_time_dqn,Task_offloading_dqn,eps)
        PEnergy_tsp=self.env.flight_energy(UAV_trajectory_tsp,UAV_flight_time_tsp,eps)
        PEnergy_dqn = self.env.flight_energy(UAV_trajectory_natural, UAV_flight_time_natural,eps)
        PEnergy_ddqn = self.env.flight_energy(UAV_trajectory_dqn, UAV_flight_time_dqn,eps)


        plot_ee = np.zeros((3, eps), dtype=np.float)
        for i in range(eps):
            plot_ee[0, i] = np.sum(Th_tsp[i, :])/np.sum(PEnergy_tsp[i, :])
            plot_ee[1, i] = np.sum(Th_dqn[i,  :])/np.sum(PEnergy_dqn[i, :])
            plot_ee[2, i] = np.sum(Th_ddqn[i,  :])/np.sum(PEnergy_ddqn[i, :])

        myfont = matplotlib.font_manager.FontProperties(
            fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")
        plt.plot(range(eps), plot_ee[0,:].T, c='r',linestyle='-', marker='<', label=u"TSP")
        plt.plot(range(eps), plot_ee[1,:].T, c='b', linestyle='-', marker='>',label=u"Natural DQN")
        plt.plot(range(eps), plot_ee[2,:].T, c='g',linestyle='-', marker='o', label=u"Double DQN")
        plt.xlabel(u'Episode', fontProperties=myfont)
        plt.ylabel(u'Energy-Efficiency(bits/J)', fontProperties=myfont)
        plt.legend(prop=myfont)
        plt.grid()
        plt.show()

        ave_tsp = np.sum(plot_ee[0, :]) / eps
        ave_dqn = np.sum(plot_ee[1, :]) / eps
        ave_ddqn = np.sum(plot_ee[2, :]) / eps
        print("Energy efficieny: TSP:%f;Natural DQN:%f;D-DQN:%f" % (ave_tsp, ave_dqn, ave_ddqn))
        return

    def plot_Q_value(self,q_natural,q_double):
        myfont = matplotlib.font_manager.FontProperties(
            fname=r"/usr/local/lib/python2.7/site-packages/matplotlib/mpl-data/fonts/ttf/SimHei.ttf")

        plt.plot(np.array(q_natural), c='b', linestyle='--', marker='',label=u"Natural DQN")
        plt.plot(np.array(q_double), c='g',linestyle='-', marker='', label=u"Double DQN")
        plt.legend(loc='best')
        plt.legend(prop=myfont)
        plt.ylabel(u'Q value', fontProperties=myfont)
        plt.xlabel(u'Traning step', fontProperties=myfont)
        plt.grid()
        plt.show()
        return


