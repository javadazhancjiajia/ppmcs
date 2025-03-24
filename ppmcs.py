import numpy as np
import os
import math

_xi = []


def least_sq(x: np.ndarray, y: np.ndarray):
    xi = np.array(_xi[-x.shape[0]:])
    sum_xi = np.sum(xi)
    sum_x_xi = np.sum(x * xi)
    sum_y_xi = np.sum(y * xi)
    sum_x_y_xi = np.sum(x * y * xi)
    sum_x_x_xi = np.sum(x * x * xi)
    k = (sum_x_xi * sum_y_xi - sum_x_y_xi * sum_xi) / (sum_x_xi ** 2 - sum_x_x_xi * sum_xi)
    b = (sum_y_xi - k * sum_x_xi) / sum_xi
    return k, b


def server_truth_discovery(data: np.ndarray, weights: np.ndarray, is_first: bool):
    if is_first:
        truths = crh.first_init(data)
    else:
        sum_weight = np.sum(weights)
        weighted_answers = data * (weights[:, np.newaxis])
        truths = np.sum(weighted_answers, axis=0) / sum_weight
    return truths


def sw(data, budget, boundary: tuple):
    ee = np.exp(budget)
    b = ((budget * ee) - ee + 1) / (2 * ee * (ee - 1 - budget))
    p = ee / (2 * b * ee + 1)
    q = 1 / (2 * b * ee + 1)
    r = np.random.uniform(0, 1, 1)[0]

    d_low = boundary[0]
    d_up = boundary[1]
    rg = d_up - d_low
    d_prime = (data - d_low) / rg

    if r <= (q * d_prime):
        d_prime_wide = r / q - b
    elif r <= (d_prime * q + 2 * b * p):
        d_prime_wide = (r - q * d_prime) / p + d_prime - b
    else:
        d_prime_wide = (r - q * d_prime - 2 * b * p) / q + d_prime + b

    return data - rg * (d_prime - d_prime_wide)


class Worker:
    def __init__(self, name, n, epsilon, omega, beta_pre, theta_pre, input_path):
        self.epsilon = epsilon
        self.omega = omega
        self.beta_pre = beta_pre
        self.theta_pre = theta_pre
        self.name = name

        self.d_cache = []

        for _ in range(n):
            self.d_cache.append({
                "remaining_budget": epsilon,
                "used_budgets": [],
                "his_data": [],
                "last_pre": [],
                "last_pd": 0,
                "_l": 0,
                "_k": 0,
                "_k_prime": 0,
                "error_list": []
            })
        self.epoch_cache = []

        if os.path.exists(input_path):
            self.data = open(input_path)
        else:
            print(f"invalid file path: {input_path}, abort")
            exit(-1)

    def read_data(self, epoch, period, boundary: list, d_boundary: list, mean: bool, is_last: bool):
        self.epoch_cache.append(epoch)

        line = self.data.readline()
        upload_data = [float(ele) for ele in line.strip().split(";")]
        split_data1 = []
        split_data2 = []

        for d_idx, d in enumerate(upload_data):
            d_info = self.d_cache[d_idx]
            theta = self.theta_pre * d_boundary[d_idx]
            beta = (boundary[d_idx][1] - boundary[d_idx][0]) * self.beta_pre

            d_info["his_data"].append(d)
            used_budgets = d_info["used_budgets"]
            remaining_budget = d_info["remaining_budget"]
            if used_budgets.__len__() >= self.omega:
                remaining_budget += used_budgets[-self.omega]

            if mean:
                c_budget = self.epsilon / self.omega / 2
                p_d = sw(d, c_budget, boundary[d_idx])
            elif d_info["_k"] <= 1 and is_last:
                c_budget = remaining_budget
                p_d = sw(d, c_budget, boundary[d_idx])
            else:
                _k_prime = d_info["_k_prime"]
                _l = d_info["_l"]
                _k = d_info["_k"]
                p_d = d_info["last_pd"]
                c_budget = 0

                is_new_start = False
                _k_prime += 1
                d_info["_k_prime"] = _k_prime

                if _k_prime <= _k:
                    d_info["error_list"].append(math.fabs(d - d_info["last_pre"][_k_prime]))
                    if math.fabs(d_info["last_pre"][0] - d) > beta:
                        is_new_start = True

                if _k_prime > _k or is_new_start:
                    # 1.1 Error Computation
                    p = 0.5
                    if d_info["error_list"].__len__() > 0:
                        error_list = d_info["error_list"]
                        # Kp=0.8, Ki=0.1, Kd=0.1
                        tau = 0.8 * error_list[-1]
                        if len(error_list) > 1:
                            for ele in error_list[:-1]:
                                tau += 0.1 * ele
                            tau += 0.1 * (error_list[-1] - error_list[-2])
                        tau = (_k / _k_prime) * tau
                        p = min(0.8, max(0.2, 1 - math.exp(-tau)))

                    # update l
                    if _k == 0:
                        _l = 1
                    elif is_new_start:
                        _l = max(1, _k // 2)
                    else:
                        _l = min(_l + 1, self.omega - 1)

                    # 1.2 data Prediction
                    next_epoch = epoch + period
                    temp_data = d_info["his_data"].copy()
                    temp_epoch = self.epoch_cache.copy()
                    pre = [d]

                    for i in range(_l):
                        a, b = least_sq(np.array(temp_epoch), np.array(temp_data))
                        next_data = a * next_epoch + b
                        pre.append(next_data)
                        temp_data.append(next_data)
                        temp_epoch.append(next_epoch)
                        next_epoch += period

                    # 1.3 Budget Computation
                    e_unit = remaining_budget / (_l + 1)
                    e_unit_2 = e_unit / 2

                    temp_k = 0
                    recycle_budget = 0
                    reserve_budget = 0
                    for idx, _p in enumerate(pre):
                        if math.fabs(_p - d) <= beta:
                            temp_k += 1
                        else:
                            break

                        if idx > 0:
                            if used_budgets.__len__() >= self.omega - idx:
                                recycle_budget += used_budgets[-self.omega + idx]

                            if recycle_budget < e_unit_2:
                                reserve_budget += e_unit_2 - recycle_budget
                                recycle_budget = 0
                            else:
                                recycle_budget -= e_unit_2

                    if temp_k == 1:
                        c_budget = temp_k * e_unit
                    else:
                        c_budget = temp_k * e_unit - p * reserve_budget

                    d_info["last_pre"] = pre
                    d_info["_k"] = temp_k - 1
                    d_info["_k_prime"] = 0
                    d_info["_l"] = _l
                    d_info["error_list"] = []

                    # 2 Data Perturbation
                    p_d = sw(d, c_budget, boundary[d_idx])

            used_budgets.append(c_budget)
            d_info["last_pd"] = p_d
            d_info["remaining_budget"] = remaining_budget - c_budget

            # 3 Data Split
            if "raw_pla" not in d_info:
                d_info["raw_pla"] = {
                    "start_x": epoch,
                    "start_y": d,
                    "is_first": True
                }

                d1 = (boundary[d_idx][0] + boundary[d_idx][1]) // 2
                split_data1.append(d1)
                split_data2.append(p_d - d1)
                d_info["per_pla"] = {
                    "start_x": epoch,
                    "start_y": d1
                }
            else:
                raw_pla = d_info["raw_pla"]
                perturb_pla = d_info["per_pla"]

                r_st_x = raw_pla["start_x"]
                r_st_y = raw_pla["start_y"]
                r_slope = (d - r_st_y) / (epoch - r_st_x)

                p_st_x = perturb_pla["start_x"]
                p_st_y = perturb_pla["start_y"]

                if raw_pla["is_first"]:
                    low = (d - theta - r_st_y) / (epoch - r_st_x)
                    up = (d + theta - r_st_y) / (epoch - r_st_x)

                    raw_pla["low"] = low
                    raw_pla["up"] = up
                    raw_pla["is_first"] = False
                    raw_pla["last_slope"] = r_slope
                    d1 = p_st_y + r_slope

                    perturb_pla["last_slope"] = (d1 - p_st_y) / (epoch - p_st_x)
                    perturb_pla["low"] = (d1 - theta - p_st_y) / (epoch - p_st_x)
                    perturb_pla["up"] = (d1 + theta - p_st_y) / (epoch - p_st_x)
                else:
                    if r_slope > raw_pla["up"] or r_slope < raw_pla["low"]:
                        if r_slope > raw_pla["up"]:
                            range_low = p_st_y + (epoch - p_st_x) * perturb_pla["up"]
                            range_up = range_low + theta
                            d1 = np.random.uniform(range_low, range_up, 1)[0]
                        else:
                            range_up = p_st_y + (epoch - p_st_x) * perturb_pla["low"]
                            range_low = range_up - theta
                            d1 = np.random.uniform(range_low, range_up, 1)[0]

                        raw_pla["start_x"] = epoch
                        raw_pla["start_y"] = d
                        raw_pla["is_first"] = True

                        perturb_pla["start_x"] = epoch
                        perturb_pla["start_y"] = d1
                    else:
                        if r_slope >= raw_pla["last_slope"]:
                            range_low = p_st_y + (epoch - p_st_x) * perturb_pla["low"]
                            range_up = p_st_y + (epoch - p_st_x) * perturb_pla["last_slope"]
                        else:
                            range_up = p_st_y + (epoch - p_st_x) * perturb_pla["up"]
                            range_low = p_st_y + (epoch - p_st_x) * perturb_pla["last_slope"]

                        d1 = np.random.uniform(range_low, range_up, 1)[0]
                        perturb_pla["last_slope"] = (d1 - p_st_y) / (epoch - p_st_x)
                        perturb_pla["low"] = max(perturb_pla["low"], (d1 - theta - p_st_y) / (epoch - p_st_x))
                        perturb_pla["up"] = min(perturb_pla["up"], (d1 + theta - p_st_y) / (epoch - p_st_x))

                        low = (d - theta - r_st_y) / (epoch - r_st_x)
                        up = (d + theta - r_st_y) / (epoch - r_st_x)
                        raw_pla["low"] = max(raw_pla["low"], low)
                        raw_pla["up"] = min(raw_pla["up"], up)
                        raw_pla["last_slope"] = r_slope

                split_data1.append(d1)
                split_data2.append(p_d - d1)

        return split_data1, split_data2

def start_with_dp(db_name):
    np.set_printoptions(formatter={'bracket': False})

    ome = 50
    bet_pre = 0.2
    the_per = 0.01
    eps = 1

    boundary = []
    global _xi
    if db_name == "lab":
        d_boundary = [14.758800000000004, 17.5501]
        boundary.append((14, 35))
        boundary.append((21, 55))
        t = 851
        m = 27
        n = 2
        bet_pre = 0.65

        alpha = 0.5
        _xi = [(alpha * (1 - alpha) ** (t - e - 1)) for e in range(t + ome)]
    elif db_name == "weather":
        d_boundary = [40.0, 27.0, 39.0, 42.0, 29.0, 22.0, 49.0, 36.0, 35.0, 31.0, 43.0, 18.0, 29.0, 46.0, 40.0, 38.0,
                      41.0, 40.0, 39.0, 19.0, 13.0, 36.0, 25.0]
        boundary = [[22, 64], [9, 44], [-8, 51], [15, 57], [-5, 42], [20, 65], [6, 58], [23, 67], [-6, 49], [12, 70],
                    [5, 64], [41, 70], [13, 50], [-13, 41], [6, 57], [-3, 61], [11, 74], [6, 54], [25, 65], [45, 66],
                    [43, 60], [11, 65], [18, 61]]
        t = 104
        m = 17
        n = 23
        bet_pre = 0.7

        alpha = 0.5
        _xi = [(alpha * (1 - alpha) ** (t - e - 1)) for e in range(t + ome)]
    elif db_name == "pems08":
        d_boundary = [673.0, 0.4497, 67.69999999999999]
        boundary.append((0.0, 865.0))
        boundary.append((0.0, 0.5264))
        boundary.append((3.9, 79.9))
        t = 1000
        m = 170
        n = 3
        bet_pre = 0.8

        alpha = 0.2
        _xi = [(alpha * (1 - alpha) ** (t - e - 1)) for e in range(t + ome)]
    elif db_name == "stock":
        d_boundary = [1350.0]
        boundary.append((2, 2044))
        t = 1259
        m = 468
        n = 1
        bet_pre = 0.6

        alpha = 0.3
        _xi = [(alpha * (1 - alpha) ** (t - e - 1)) for e in range(t + ome)]
    else:
        print("Unidentified dataset, abort.")
        return

    if not os.path.exists(f".log/ppmcs/{db_name}"):
        os.makedirs(f".log/ppmcs/{db_name}")
        os.makedirs(f".log/ppmcs/{db_name}/data1")
        os.makedirs(f".log/ppmcs/{db_name}/data2")

    for ep in range(1):
        worker_list = []
        for file_name in os.listdir(f".dataset/{db_name}"):
            worker_list.append(Worker(name=file_name, n=n, epsilon=eps, omega=ome, beta_pre=bet_pre, theta_pre=the_per, input_path=f".dataset/{db_name}/{file_name}"))

        d1_f = open(f".log/ppmcs/{db_name}/data1/{ep}", "w", encoding="utf-8")
        d2_f = open(f".log/ppmcs/{db_name}/data2/{ep}", "w", encoding="utf-8")
        for epoch in range(1, t + 1):
            data1 = []
            data2 = []

            for idx, worker in enumerate(worker_list):
                if epoch <= 10:
                    s1, s2 = worker.read_data(epoch=epoch, period=1, boundary=boundary, d_boundary=d_boundary, mean=True, is_last=False)
                elif epoch == t:
                    s1, s2 = worker.read_data(epoch=epoch, period=1, boundary=boundary, d_boundary=d_boundary, mean=False, is_last=True)
                else:
                    s1, s2 = worker.read_data(epoch=epoch, period=1, boundary=boundary, d_boundary=d_boundary, mean=False, is_last=False)
            
                data1.append(s1)
                data2.append(s2)

            np_data1 = np.array(data1)
            np_data2 = np.array(data2)

            d1_f.write(';'.join(map(str, np_data1.flatten())) + "\n")
            d2_f.write(';'.join(map(str, np_data2.flatten())) + "\n")
        d1_f.close()
        d2_f.close()


if __name__ == '__main__':
    np.set_printoptions(linewidth=1000)
    
    start_with_dp("lab")
