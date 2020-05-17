#-- coding:UTF-8 --
"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.8.0
"""
from UAV_MEC_env import UAV_MEC
from Convex_Optimization import Convex
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib
from Res_plot import Res_plot

env = UAV_MEC()
res = Res_plot(env)
con_op = Convex()
MEMORY_SIZE = 3200
Episodes = env.eps

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):
    natural_DQN = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):
    double_DQN = DoubleDQN(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer())

# record the results

UAV_trajectory_tsp = np.zeros((Episodes, env.N_slot, 3), dtype=np.float)
Task_offloading_tsp = np.zeros((Episodes, env.N_slot, env.GTs), dtype=np.float)
UAV_flight_time_tsp =  np.zeros((Episodes, env.N_slot), dtype=np.float)

UAV_trajectory_natural = np.zeros((Episodes, env.N_slot, 3), dtype=np.float)
Task_offloading_natural = np.zeros((Episodes, env.N_slot, env.GTs), dtype=np.float)
UAV_flight_time_natural =  np.zeros((Episodes, env.N_slot), dtype=np.float)

UAV_trajectory_dqn = np.zeros((Episodes, env.N_slot, 3), dtype=np.float)
Task_offloading_dqn = np.ones((Episodes, env.N_slot, env.GTs), dtype=np.float)
UAV_flight_time_dqn = np.ones((Episodes, env.N_slot), dtype=np.float)

def train(RL):
    for ep in range(Episodes):
        # Initial the duration of each time slot using convex optimization
        if (ep>0):
            r_kn = np.zeros((env.GTs, env.N_slot), dtype=np.float)
            t_kn = np.zeros((env.GTs, env.N_slot), dtype=np.float)
            for k in range(env.GTs):
                for n in range(env.N_slot):
                    if (RL.double_q == True):
                        h = UAV_trajectory_dqn[ep-1,n,2]
                        x = UAV_trajectory_dqn[ep-1,n,0]
                        y = UAV_trajectory_dqn[ep-1,n,1]
                        a_kn = Task_offloading_dqn[ep-1, n, k]
                    else:
                        h = UAV_trajectory_natural[ep - 1, n, 2]
                        x = UAV_trajectory_natural[ep - 1, n, 0]
                        y = UAV_trajectory_natural[ep - 1, n, 1]
                        a_kn = Task_offloading_natural[ep - 1, n, k]

                    r_kn[k, n] = env.link_rate_single(h, x, y, env.w_k[k, :])
                    t_kn[k,n]=a_kn*(env.u_k[k,0]/(r_kn[k,n]*env.N_slot)+(env.u_k[k,1]/env.N_slot)/env.f_u)
            if (RL.double_q == True):
                con_op.initial_op(UAV_trajectory_dqn[ep-1, :, :], Task_offloading_dqn[ep-1,:,:], t_kn, env.GTs, env.l_o_h,env.l_o_v, env.x_s, env.y_s, env.h_s,v_n_l)
                [v_n_l, t_n] = con_op.solve()
                UAV_flight_time_dqn[ep, :] = t_n[0, :]
            else:
                con_op.initial_op(UAV_trajectory_natural[ep-1, :, :], Task_offloading_natural[ep-1, :, :], t_kn, env.GTs,env.l_o_h,env.l_o_v, env.x_s, env.y_s, env.h_s,v_n_l)
                [v_n_l,t_n] = con_op.solve()
                UAV_flight_time_natural[ep,:]=t_n[0,:]
        else:
            t_n = 1 * np.ones((1, env.N_slot), dtype=np.float)
            v_n_l = 1*np.ones((1, env.N_slot), dtype=np.float)

        observation = env.reset()

        for slot in range(env.N_slot):
            t_n_c= t_n[0, slot]
            action_index = RL.choose_action(observation)
            action = env.find_action(action_index)
            observation_, reward = env.step(action,t_n_c,slot)
            RL.store_transition(observation, action, reward, observation_)
            if (RL.double_q==True):
                UAV_trajectory_dqn[ep,slot,:] = observation_[:]
                Task_offloading_dqn[ep,slot,:] = action[-env.GTs:]
            else:
                UAV_trajectory_natural[ep, slot, :] = observation_[:]
                Task_offloading_natural[ep, slot, :] = action[-env.GTs:]

            if env.N_slot*ep+slot >= MEMORY_SIZE:
                RL.learn()
            observation = observation_
        print("Finish episode %d" %ep)
        if (RL.double_q == True):
            UAV_trajectory_dqn[ep,:]=env.UAV_FLY(UAV_trajectory_dqn[ep,:])
        else:
            UAV_trajectory_natural[ep,:] = env.UAV_FLY(UAV_trajectory_natural[ep,:])
        #for TSP
        UAV_trajectory_tsp[ep,:]=env.UAV_trajectory_tsp[:]
        Task_offloading_tsp[ep,:]=np.ones((env.N_slot, env.GTs), dtype=np.float)
        UAV_flight_time_tsp[ep,:]= 3*np.ones((1,env.N_slot), dtype=np.float)
    return RL.q

q_natural = train(natural_DQN)
print("Train double DQN")
q_double = train(double_DQN)

EPS=env.eps-1
res.plot_UAV_GT(env.w_k,UAV_trajectory_natural,UAV_trajectory_dqn,UAV_trajectory_tsp)
res.plot_Q_value(q_natural,q_double)
res.plot_propulsion_energy(UAV_trajectory_tsp,UAV_trajectory_natural,UAV_trajectory_dqn,UAV_flight_time_tsp,UAV_flight_time_natural,UAV_flight_time_dqn,EPS)
res.plot_data_throughput(UAV_trajectory_tsp,UAV_trajectory_natural,UAV_trajectory_dqn,UAV_flight_time_tsp,UAV_flight_time_natural,UAV_flight_time_dqn,Task_offloading_tsp,Task_offloading_natural,Task_offloading_dqn,EPS)
res.plot_energy_efficiency(UAV_trajectory_tsp,UAV_trajectory_natural,UAV_trajectory_dqn,UAV_flight_time_tsp,UAV_flight_time_natural,UAV_flight_time_dqn,Task_offloading_tsp,Task_offloading_natural,Task_offloading_dqn,EPS)