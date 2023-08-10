
import numpy as np
from environment.simulation import *

class UPMSP:
    def __init__(self, num_jt=10, num_j=1000, num_m=8, log_dir=None, K = 1, action_number = 10, min = 0.1, max = 4, action_mode = 'WCOVERT'):
        self.num_jt = num_jt
        self.num_machine = num_m
        self.jobtypes = [i for i in range(num_jt)]  # 1~10
        self.p_ij, self.p_j, self.weight = self._generating_data()

        # print(self.p_ij)
        # print(self.weight)

        self.num_job = num_j
        self.log_dir = log_dir
        self.jobtype_assigned = list()  # 어느 job이 어느 jobtype에 할당되는 지
        self.job_list = list()  # 모델링된 Job class를 저장할 리스트
        self.K = K
        self.done = False
        self.tardiness = 0.0
        self.e = 0
        self.time = 0
        self.action_number = action_number
        self.action_mode = action_mode
        if action_mode == 'heuristic':
            # self.mapping = {0: "WSPT", 1: "WMDD", 2: "ATC", 3: "WCOVERT"}
            self.mapping = {0: "WSPT", 1: "ATC", 2: "WCOVERT"}
        elif action_mode == 'WCOVERT':  # WCOVERT PPO
            self.mapping = {}
            for i in range(0, action_number-1):
                self.mapping[i] = min + i * (max - min) / (action_number-2)
            self.mapping[action_number-1] = 999999999999
        elif action_mode == 'ATC': # ATC PPO
            self.mapping = {}
            for i in range(0, action_number):
                self.mapping[i] = min + (i + 1) * (max - min) / action_number
        self.sim_env, self.process_dict, self.source_dict, self.sink, self.routing, self.monitor = self._modeling()
        # self.one_hot_enc = np.eye(self.num_job)
        # self.tardiness_jt = [0]*self.num_jt
        # self.fully_connected_machine_edge_index = [[self.num_machine],[m for m in range(self.num_machine)]]

    def step(self, action: int):
        done = False
        self.previous_time_step = self.sim_env.now
        routing_rule = self.mapping[action]

        self.routing.decision.succeed(routing_rule)
        self.routing.indicator = False

        while True:
            if self.routing.indicator:
                if self.sim_env.now != self.time:
                    self.time = self.sim_env.now
                break

            if self.sink.finished_job == self.num_job:
                done = True
                self.sim_env.run()
                if self.e % 50 == 0:
                    self.monitor.save_tracer()
                # self.monitor.save_tracer()
                break

            if len(self.sim_env._queue) == 0:
                self.monitor.save_tracer()
                print(self.monitor.filepath)
            self.sim_env.step()

        reward = self._calculate_reward()
        next_state = self._get_state()

        return next_state, reward, done

    def reset(self):
        self.e = self.e + 1 if self.e > 0 else 1  # episode
        self.p_ij, self.p_j, self.weight = self._generating_data()

        # print(self.p_ij)
        # print(self.weight)

        self.sim_env, self.process_dict, self.source_dict, self.sink, self.routing, self.monitor = self._modeling()
        self.done = False
        self.monitor.reset()

        self.tardiness = 0

        while True:
            # Check whether there is any decision time step
            if self.routing.indicator:
                break

            self.sim_env.step()

        return self._get_state()

    def _modeling(self):
        env = simpy.Environment()

        monitor = Monitor(self.log_dir + '/log_%d.csv '% self.e)
        # print(self.log_dir + '/log_%d.csv '% self.e)
        # monitor = Monitor("C:/Users/sohyon/PycharmProjects/UPJSP_SH/environment/result/log_{0}.csv".format(self.e))
        process_dict = dict()
        source_dict = dict()
        jt_dict = dict()  # {"JobType 0" : [Job class(), ... ], ... }
        time_dict = dict()  # {"JobType 0" : [pij,...], ... }
        routing = Routing(env, process_dict, source_dict, monitor, self.weight)
        #############################################
        routing.mapping = self.mapping
        routing.action_mode = self.action_mode
        #############################################
        # 0에서 9까지 랜덤으로 배정
        self.jobtype_assigned = np.random.randint(low=0, high=self.num_jt, size=self.num_job)
        for i in range(self.num_job):
            jt = self.jobtype_assigned[i]
            if "JobType {0}".format(jt) not in jt_dict.keys():
                jt_dict["JobType {0}".format(jt)] = list()
                time_dict["JobType {0}".format(jt)] = self.p_ij[jt]
            jt_dict["JobType {0}".format(jt)].append(
                Job("Job {0}-{1}".format(jt, i), self.p_ij[jt], job_type=jt))

        sink = Sink(env, monitor, jt_dict, self.num_job, source_dict, self.weight)

        for jt_name in jt_dict.keys():
            source_dict["Source {0}".format(int(jt_name[-1]))] = Source("Source {0}".format(int(jt_name[-1])), env,
                                                                        routing, monitor, jt_dict, self.p_j, self.K,
                                                                      self.num_machine)
        for i in range(self.num_machine):
            process_dict["Machine {0}".format(i)] = Process(env, "Machine {0}".format(i), sink, routing, monitor)

        return env, process_dict, source_dict, sink, routing, monitor

    def _get_state(self) -> np.ndarray:
        # define 8 features
        f_1 = np.zeros(self.num_jt)         # placeholder for feature 1
        f_2 = np.zeros(self.num_machine)    # placeholder for feature 2
        f_3 = np.zeros(self.num_machine)    # placeholder for feature 3
        f_4 = np.zeros(self.num_machine)    # placeholder for feature 4
        f_5 = np.zeros(self.num_jt)         # placeholder for feature 5
        f_6 = np.zeros(self.num_jt)         # placeholder for feature 6
        f_7 = np.zeros(self.num_jt)         # placeholder for feature 7
        f_8 = np.zeros([self.num_jt, 4])    # placeholder for feature 8
        f_9 = 0                           # feature 9

        for jt_name in self.source_dict.keys():  # for loop by job type
            j = int(jt_name[-1])  # j : job type index
            job_list = [job for job in self.routing.queue.items if job.job_type == j]  # job objects whose job type is j
            if not job_list:
                pass    # f_1[j] = 0
            else:
                f_1[j] = 2 ** (-1 / len(job_list))  # due date of jobs whose job type is j
            """
            To do 1 : count the number of job type j and fill f_1 by the paper's instruction
            - f_1 (job-related feature) : the number of non-processes job in job type j
            - Given 1 : Utilize Job object(class Job)'s storage object(self.routing.queue.items)(i.e. job objects in self.routing.queue.items) 
            - Given 2 : Utilize Job object(class Job)'s job type attribute(Job.job_type)
            """

        for i in range(self.num_machine):  # for loop by machine
            machine = self.process_dict["Machine {0}".format(i)]  # machine object

            if machine.idle:
                pass
                '''
                f_2[i] = 0
                f_3[i] = 0
                f_4[i] = 0
                '''
            else:
                jt_no = machine.job.job_type
                f_2[i] = jt_no / (self.num_jt)
                f_3[i] = (machine.planned_finish_time - self.sim_env.now) / self.p_j[jt_no]
                f_4[i] = (machine.job.due_date - self.sim_env.now) / self.p_j[jt_no]

            """
            To do 2 : check if machine is idle and fill f_2 by the paper's instruction
            - f_2 (machine-related feature) : what job type is processed in machine
            - Given 1 : Utilize machine(class Process)'s idle attribute(machine.idle)
            - Given 2 : Utilize machine's current processing job attribute(machine.job) (if machine is idle, machine.job is None)
            """

            """
            To do 3 : calculate remaining processing time and fill f_3 by the paper's insturction
            - f_3 (machine-related feature) : how much the time is remaining
            - Given 1 : Utilize machine object(class Process)'s planned finish time attribute(machine.planned_finish_time)
            - Given 2 : Utilize simulation's current time(self.sim_env.now)
            - Given 3 : Utilize job type j's average(or nominal) processing time(self.p_j)
            - Given 4 : Utilize machine's current processing job attribute(machine.job) (if machine is idle, machine.job is None)
            """

            """
            To do 4 : calculate due remaining and fill f_4 by the paper's insturction
            - f_4 (machine-related feature) : how much the time is remaining to go to the due date
            - Given 1 : Utilize due date of machine object(class Process)'s current working job(job.due_date)
            - Given 2 : Utilize simulation's current time(self.sim_env.now)
            - Given 3 : Utilize job type j's average(or nominal) processing time(self.p_j)
            - Given 4 : Utilize machine's current processing job attribute(machine.job) (if machine is idle, machine.job is None)
            """

        for jt_name in self.source_dict.keys():  # for loop by job type
            j = int(jt_name[-1])
            job_list = [job for job in self.routing.queue.items if job.job_type == j]  # job objects whose job type is j
            jt_duedates = [job.due_date for job in job_list]  # due date of jobs whose job type is j

            """
            To do 5 : calculate waiting job's minimal tightness of the due date allowance and fill feature 5 
            """
            if not jt_duedates:
                pass
                '''
                f_5[j] = 0
                f_6[j] = 0
                f_7[j] = 0
                for g in range(4):
                    f_8[j,g] = 0
                '''

            else:
                f_5[j] = (min(jt_duedates) - self.sim_env.now) / self.p_j[j]
                f_6[j] = (max(jt_duedates) - self.sim_env.now) / self.p_j[j]
                f_7[j] = (np.sum(jt_duedates) / len(job_list) - self.sim_env.now) / self.p_j[j]
                max_pij = max(self.p_ij[j])
                min_pij = min(self.p_ij[j])
                inf_no = np.inf
                # ng_inf_no=-np.inf
                NI_jg = np.zeros(4)

                for jt_duedates_jk in jt_duedates:
                    d_jk_m_t = jt_duedates_jk - self.sim_env.now
                    if d_jk_m_t > max_pij and d_jk_m_t < inf_no:
                        NI_jg[0] += 1
                    elif d_jk_m_t > min_pij and d_jk_m_t <= max_pij:
                        NI_jg[1] += 1
                    elif d_jk_m_t > 0 and d_jk_m_t <= min_pij:
                        NI_jg[2] += 1
                    else:
                        NI_jg[3] += 1
                for g in range(4):
                    if NI_jg[g] == 0:
                        pass
                    else:
                        f_8[j, g] = 2 ** (-1 / NI_jg[g])

            """
            To do 6 : calculate waiting job's maximal tightness of the due date allowance and fill feature 6
            """

            """
            To do 7 : calculate waiting job's average tightness of the due date allowance and fill feature 7 
            """

            """
            To do 8 : calculate each job type's tightness of the due date allowance and fill f_8 by the paper's insturction(utilize each object's already mentioned attribute) 
            - Given 1 : Utilize job type j's processing time on machine i(self.p_ji)
            - Given 2 : Utilize job_list(defined in line 173) and jt_duedates(defined in line 174)
            """
        f_8 = f_8.flatten()
        """
        feature 9 : 선택한 action에 대한 (job, machine) 조합이 선택하지 않은 action에 대한 결과와 얼마나 다른지 chk
        set에 저장하고 길이 사용
        f_9 = action(routing rule)에 대한 (job, machine) 종류 개수 / action 개수
        """
        f_9 = self.routing.action_result_info    # 선택한 action

        state = np.concatenate((f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9), axis=None)
        return state

    # def get_job_graph(self):
    #     waiting_job_node_feature = self._get_job_feature()
    #     waiting_job_edge_index = self._get_job_edge_index()
    #     return waiting_job_node_feature, waiting_job_edge_index
    #
    # def _get_job_feature(self):
    #     waiting_job_node_feature = list()
    #     scaling_factor = 1 / 10
    #     now = copy.deepcopy(self.sim_env.now)
    #     for job in self.routing.queue.items:
    #         jt = job.job_type
    #         feature1 = self.one_hot_enc[jt]
    #         feature2 = np.array([(now-job.due_date)/scaling_factor])
    #         feature3 = np.array([0])
    #         feature4 = np.array([now-job.past]/scaling_factor)
    #         feature = np.concatenate([feature1, feature2, feature3, feature4])
    #         waiting_job_node_feature.append(feature)
    #     for i in range(self.num_machine):
    #         machine = self.process_dict["Machine {0}".format(i)]
    #         if machine.idle != True:
    #             job = machine.job
    #             jt = job.job_type
    #             feature1 = self.one_hot_enc[jt]
    #             feature2 = np.array([(now - job.due_date)/scaling_factor])
    #             feature3 = np.array([(machine.planned_finish_time - self.sim_env.now) / self.p_j[machine.job.job_type]])
    #             feature4 = np.array([now - job.past] / scaling_factor)
    #             feature = np.concatenate([feature1, feature2, feature3, feature4])
    #             waiting_job_node_feature.append(feature)
    #         else:
    #             feature1 = np.zeros(self.num_jt)
    #             feature2 = np.array([0])
    #             feature3 = np.array([0])
    #             feature4 = np.array([0])
    #             feature = np.concatenate([feature1, feature2, feature3, feature4])
    #             waiting_job_node_feature.append(feature)
    #     return waiting_job_node_feature
    #
    # def _get_job_edge_index(self):
    #     waiting_job_edge_index = [[],[]]
    #     num_waiting_job = len(self.routing.queue.items)
    #     idle_machine_name = [self.process_dict["Machine {0}".format(i)].name+num_waiting_job for i in range(self.num_machine) if self.process_dict["Machine {0}".format(i)].idle == True]
    #     machine_name = [self.process_dict["Machine {0}".format(i)].name + num_waiting_job
    #                     for i in range(self.num_machine)]
    #     for i in idle_machine_name:
    #         for j in range(num_waiting_job):
    #             waiting_job_edge_index[0].append(i)
    #             waiting_job_edge_index[1].append(j)
    #     for k in machine_name:
    #         waiting_job_edge_index[0].append(k)
    #         waiting_job_edge_index[1].append(k)
    #     return waiting_job_edge_index
    #
    # def _get_machine_feature(self, machine_feature):
    #     dummy = np.zeros([1, machine_feature.shape[1]])
    #     return np.concatenate([machine_feature, dummy], axis = 0)
    #
    # def get_machine_edge_index(self):
    #     return self.fully_connected_machine_edge_index
    #
    # def get_tardiness_jt(self):
    #     return np.array(self.tardiness)/self.sim_env.now









    def _calculate_reward(self, mode= 'graph'):
        if mode == 'graph':
            reward = 0
            scaling_factor = 1/ 250
            now = copy.deepcopy(self.sim_env.now)
            ### job before process ###
            for job in self.routing.queue.items:
                jt = job.job_type
                w_j = self.weight[jt]
                if now < job.due_date:
                    reward += 0
                else:
                    reward += -(now - job.past) * w_j * scaling_factor
                job.past = now
            ### job processing ###
            for i in range(self.num_machine):
                machine = self.process_dict["Machine {0}".format(i)]
                if not machine.idle:
                    job = machine.job
                    jt = job.job_type
                    w_j = self.weight[jt]
                    if now < job.due_date:
                        reward += 0
                    else:
                        reward += -(now - job.past) * w_j * scaling_factor
                    job.past = now
                else:
                    pass
            ### job after process ###
            for job in self.sink.job_list:
                if job.sink_just:
                    jt = job.job_type
                    w_j = self.weight[jt]
                    if now < job.due_date:
                        reward += 0
                    else:
                        reward += -(now - job.past) * w_j * scaling_factor
                    job.past = now
                    job.sink_just = False
                else:
                    pass
            return reward

        # else: # 예전것. 안 씀.
        #     reward = 0
        #     finished_jobs = copy.deepcopy(self.sink.job_list)
        #     for job in finished_jobs:
        #         jt = job.job_type
        #         w_j = self.weight[jt]
        #         tardiness = min(job.due_date - job.completion_time, 0)
        #         reward += w_j * tardiness * (1/1000)
        #     self.sink.job_list = list()
        # return reward

    def _generating_data(self):
        processing_time = [[np.random.uniform(low=1, high=20) for _ in range(self.num_machine)] for _ in
                           range(self.num_jt)]
        p_j = [np.mean(jt_pt) for jt_pt in processing_time]
        weight = list(np.random.uniform(low=0, high=5, size=self.num_jt))

        return processing_time, p_j, weight
