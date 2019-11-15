import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import math
import sys
import os
import random
from multiprocessing import Process, Queue
import shutil


def backtest_slice(q, ticker, plot=False):
    obj = Strategy(ticker, plot=plot)
    daily_return_slice = obj.backtest()
    q.put(daily_return_slice)


class Strategy():
    def __init__(self, ticker, plot=False) -> None:
        if 'slice' not in ticker:
            self.ticker = ticker
            self.all_data = pd.read_csv(ticker + ".csv")
        else:
            self.all_data = pd.read_csv(ticker)
        self.multiplier_df = pd.read_csv("multiplier.csv")
        self.date_list = sorted(list(set(self.all_data["date"])))
        self.all_data["quantity"] = self.all_data["Volume_(BTC)"].apply(lambda x: round(x, 2))
        self.x = list(range(1440))
        self.xtick = list(range(0, 1441, 120))
        self.xticklabel = list(range(0, 25, 2))
        self.plot = plot

    def multi_backtest(self, plot=False):
        try:
            shutil.rmtree("backtest_selected")
        except FileNotFoundError:
            pass
        os.makedirs("backtest_selected")
        if os.path.exists("daily_return.csv"):
            os.remove("daily_return.csv")
        if not os.path.exists(self.ticker + "/data_slice_9.csv"):
            self.slice()
        q = Queue()
        jobs = list()
        re_df = pd.DataFrame()
        for i in range(0, 10):
            ticker = self.ticker + "/data_slice_" + str(i) + ".csv"
            p = Process(target=backtest_slice, args=(q, ticker, plot))
            jobs.append(p)
            p.start()
            print("Start process" + str(i))
        for i in range(0, 10):
            re_df = re_df.append(q.get())
        for job in jobs:
            job.join()
        self.re_df = re_df

    def slice(self, process_num=10):
        n = process_num
        N = len(self.date_list)
        try:
            os.removedirs("btc/" + self.ticker)
            print("Re-writing the data slice of", self.ticker)
        except:
            print("Writing new data slice of", self.ticker)
        os.makedirs("btc/" + self.ticker)
        for i in range(0, n):
            date_scope = self.date_list[math.floor(i * (N / n)): math.floor((i + 1) * (N / n))]
            data_slice = self.all_data[self.all_data["date"].apply(lambda s: True if s in date_scope else False)]
            data_slice.to_csv("btc/" + "\\data_slice_" + str(i) + ".csv", index=False)
        print("Slice data into " + str(n) + " part.\n Save data slice to: " + "btc/" + self.ticker)

    def backtest(self) -> None:
        daily_return_slice = pd.DataFrame()
        for i in range(len(self.date_list)):
            daily_return_slice = daily_return_slice.append(self.backtest_oneday(i))
        return daily_return_slice

    def backtest_oneday(self, i: int) -> None:
        date = self.date_list[i]
        self.initDailyParam(pos_type="all", date=date, i=i)
        self.ax = None
        if self.plot:
            self.initPlot()

        for n in range(1440):
            self.Signal(n, 'B')
            self.Signal(n, 'S')

        re = round(self.pnl / self.y[0] * 100, 2)
        re_df = pd.DataFrame([[date, re], ], columns=["date", "return"])
        # re_df.to_csv("daily_return.csv", mode='a', index=None, header=None)

        if self.plot:
            self.ax.set_xticks(self.xtick, self.xticklabel)
            self.ax.set_title(date + " Daily Return: " + str(re) + '%')
            self.fig.savefig("backtest/" + self.date + ".png")
            if re != 0:
                self.fig.savefig("backtest_selected/" + self.date + ".png")
            plt.close()
        print(date)
        if re < - 1:
            print("------------", date, re, "-------------------")
        return re_df

    def initDailyParam(self, pos_type="all", date=None, i=None) -> None:
        if pos_type == "all":
            self.date = date
            df = self.all_data.iloc[1440 * i: 1440 * (i + 1)].copy()
            self.y = df["price"].tolist()
            self.q = df["quantity"].tolist()
            raw_slope_list = [0, ] + list(np.diff(self.y))
            self.multiplier = self.multiplier_df[self.multiplier_df["date"] == date]
            self.multiplier = self.multiplier["multiplier"].tolist()[0]
            self.multiplier = self.multiplier / 4 + self.y[0] / 1000 / 4
            self.slope_list_real = [int(round(t / self.multiplier)) for t in raw_slope_list]
            self.y_min = df["price"].min()
            self.y_max = df["price"].max()
            self.y_mid = 0.5 * (self.y_min + self.y_max)
            self.pnl = 0
            self.TREB_close_x = - 1000
        if pos_type == 'B' or pos_type == "all":
            self.RAPB_num = 0
            self.RAPB_sig_type = None
            self.RAPB_start_pos = None
            self.RAPB_start_price = None
            self.RAPB_peak_pos = None
            self.RAPB_peak_price = None
        if pos_type == 'S' or pos_type == "all":
            self.RAPS_num = 0
            self.RAPS_sig_type = None
            self.RAPS_start_pos = None
            self.RAPS_start_price = None
            self.RAPS_nadir_pos = None
            self.RAPS_nadir_price = None


    def smooth(self):
        p_x_ls1 = [self.x[0], ]
        p_y_ls1 = [self.y[0], ]
        for i in range(1, 1439):
            if self.y[i] >= self.y[i - 1] and self.y[i] >= self.y[i + 1]:
                p_x_ls1.append(self.x[i])
                p_y_ls1.append(self.y[i])
        p_x_ls1.append(self.x[-1])
        p_y_ls1.append(self.y[-1])
        p_x_ls2 = [p_x_ls1[0], ]
        p_y_ls2 = [p_y_ls1[0], ]
        for i in range(1, len(p_x_ls1) - 1):
            if (p_y_ls1[i] - p_y_ls1[i - 1]) * (p_y_ls1[i + 1] - p_y_ls1[i]) > 0:
                continue
            else:
                p_x_ls2.append(p_x_ls1[i])
                p_y_ls2.append(p_y_ls1[i])
        p_x_ls2.append(p_x_ls1[-1])
        p_y_ls2.append(p_y_ls1[-1])
        p_x_ls3 = [p_x_ls2[0], ]
        p_y_ls3 = [p_y_ls2[0], ]
        for i in range(1, len(p_x_ls2) - 1):
            if p_y_ls2[i] >= p_y_ls2[i - 1] and p_y_ls2[i] >= p_y_ls2[i + 1]:
                p_x_ls3.append(p_x_ls2[i])
                p_y_ls3.append(p_y_ls2[i])
        p_x_ls3.append(p_x_ls2[-1])
        p_y_ls3.append(p_y_ls2[-1])
        p_x_ls4 = [p_x_ls3[0], ]
        p_y_ls4 = [p_y_ls3[0], ]
        for i in range(1, len(p_x_ls3) - 1):
            if (p_y_ls3[i] - p_y_ls3[i - 1]) * (p_y_ls3[i + 1] - p_y_ls3[i]) > 0:
                continue
            else:
                p_x_ls4.append(p_x_ls3[i])
                p_y_ls4.append(p_y_ls3[i])
        p_x_ls4.append(p_x_ls3[-1])
        p_y_ls4.append(p_y_ls3[-1])
        p_x_ls5 = [p_x_ls4[0], ]
        p_y_ls5 = [p_y_ls4[0], ]
        for i in range(1, len(p_x_ls4) - 1):
            if p_y_ls4[i] >= p_y_ls4[i - 1] and p_y_ls4[i] >= p_y_ls4[i + 1]:
                p_x_ls5.append(p_x_ls4[i])
                p_y_ls5.append(p_y_ls4[i])
        p_x_ls5.append(p_x_ls4[-1])
        p_y_ls5.append(p_y_ls4[-1])




        n_x_ls1 = [self.x[0], ]
        n_y_ls1 = [self.y[0], ]
        for i in range(1, 1439):
            if self.y[i] <= self.y[i - 1] and self.y[i] <= self.y[i + 1]:
                n_x_ls1.append(self.x[i])
                n_y_ls1.append(self.y[i])
        n_x_ls1.append(self.x[-1])
        n_y_ls1.append(self.y[-1])
        n_x_ls2 = [n_x_ls1[0], ]
        n_y_ls2 = [n_y_ls1[0], ]
        for i in range(1, len(n_x_ls1) - 1):
            if (n_y_ls1[i] - n_y_ls1[i - 1]) * (n_y_ls1[i + 1] - n_y_ls1[i]) > 0:
                continue
            else:
                n_x_ls2.append(n_x_ls1[i])
                n_y_ls2.append(n_y_ls1[i])
        n_x_ls2.append(n_x_ls1[-1])
        n_y_ls2.append(n_y_ls1[-1])
        n_x_ls3 = [n_x_ls2[0], ]
        n_y_ls3 = [n_y_ls2[0], ]
        for i in range(1, len(n_x_ls2) - 1):
            if n_y_ls2[i] <= n_y_ls2[i - 1] and n_y_ls2[i] <= n_y_ls2[i + 1]:
                n_x_ls3.append(n_x_ls2[i])
                n_y_ls3.append(n_y_ls2[i])
        n_x_ls3.append(n_x_ls2[-1])
        n_y_ls3.append(n_y_ls2[-1])
        n_x_ls4 = [n_x_ls3[0], ]
        n_y_ls4 = [n_y_ls3[0], ]
        for i in range(1, len(n_x_ls3) - 1):
            if (n_y_ls3[i] - n_y_ls3[i - 1]) * (n_y_ls3[i + 1] - n_y_ls3[i]) > 0:
                continue
            else:
                n_x_ls4.append(n_x_ls3[i])
                n_y_ls4.append(n_y_ls3[i])
        n_x_ls4.append(n_x_ls3[-1])
        n_y_ls4.append(n_y_ls3[-1])
        n_x_ls5 = [n_x_ls4[0], ]
        n_y_ls5 = [n_y_ls4[0], ]
        for i in range(1, len(n_x_ls4) - 1):
            if n_y_ls4[i] <= n_y_ls4[i - 1] and n_y_ls4[i] <= n_y_ls4[i + 1]:
                n_x_ls5.append(n_x_ls4[i])
                n_y_ls5.append(n_y_ls4[i])
        n_x_ls5.append(n_x_ls4[-1])
        n_y_ls5.append(n_y_ls4[-1])
        self.support_x_list = n_x_ls3
        self.support_y_list = n_y_ls3
        self.press_x_list = p_x_ls3
        self.press_y_list = p_y_ls3


    def initPlot(self) -> (plt, plt):
        y_offset = self.y[0] / 1000 * 0.5
        self.fig, self.ax = plt.subplots(figsize=(30, 15))
        self.ax.plot(self.x, self.y, color="black", linewidth=0.5)
        self.ax.plot(self.x, self.y, ".", color="black", markersize=1)
        for i in range(1440):
            slope = self.slope_list_real[i]
            if abs(slope) > 3:
                # self.ax.text(self.x[i] - 1, self.y[i] + y_offset, str(abs(slope)), fontsize=10, color=color)
                self.ax.plot(self.x[i], self.y[i], ".", color="black", markersize=3)

        # support_x_list, support_y_list = self.support_lines(1439)
        support_x_list = self.support_xs(0, 1439) + [1439, ]
        support_y_list = [self.y[k] for k in support_x_list]
        self.ax.plot(support_x_list, support_y_list, "-", color="blue", linewidth=0.5)
        self.ax.plot(support_x_list, support_y_list, "*", color="red", markersize=2)
        # self.ax.plot(self.press_x_list, self.press_y_list, "*-", color="violet", linewidth=0.5, markersize=2)

        plt.title(self.date, size=15)

    def count(self, n: int, threshold: int, *args):
        k = 0
        for h in args:
            if h >= threshold:
                k += 1
        if k >= n:
            return True
        else:
            return False

    def previous_trend(self, n: int):
        if n < 8:
            var1 = 0
        else:
            var1 = round((self.yplus[n] - self.yplus[n - 8]) / self.multiplier / 8, 3)
        if n < 30:
            var2 = 0
        else:
            var2 = round((self.yplus[n] - self.yplus[n - 30]) / self.multiplier / 30, 3)
        if n < 60:
            var3 = 0
        else:
            var3 = round((self.yplus[n] - self.yplus[n - 60]) / self.multiplier / 60, 3)
        if n < 120:
            var4 = 10
        else:
            var4 = round((self.yplus[n] - self.yplus[n - 120]) / self.multiplier / 120, 3)
        if n < 240:
            var5 = 10
        else:
            var5 = round((self.yplus[n] - self.yplus[n - 240]) / self.multiplier / 240, 3)
        return var1, var2, var3, var4, var5

    def same_trend(self, var1, var2, var3, var4, var5):
        if not (var1 > var2 and var2 > var3 and var3 > var4 and var4 > var5):
            return False
        elif var1 < 0 and var1 >= 0.5 * var2 and var2 >= 0.5 * var3 and var3 >= 0.5 * var4 and var4 >= 0.5 * var5:
            return True
        elif var1 >= 0 and var2 < 0 and var2 >= 0.5 * var3 and var3 >= 0.5 * var4 and var4 >= 0.5 * var5:
            return True
        elif var2 >= 0 and var3 < 0 and var1 >= 2 * var2 and var3 >= 0.5 * var4 and var4 >= 0.5 * var5:
            return True
        elif var3 >= 0 and var4 < 0 and var1 >= 2 * var2 and var2 >= 2 * var3 and var4 >= 0.5 * var5:
            return True
        elif var4 >= 0 and var5 < 0 and var1 >= 2 * var2 and var2 >= 2 * var3 and var3 >= 2 * var4:
            return True
        elif var5 >= 0 and var1 >= 2 * var2 and var2 >= 2 * var3 and var3 >= 2 * var4 and var4 >= 2 * var5:
            return True
        else:
            return False

    def previous_range(self, n: int):
        if n < 60:
            range1 = None
        else:
            ls = self.y[n - 60: n]
            ls.pop(ls.index(max(ls)))
            ls.pop(ls.index(max(ls)))
            ls.pop(ls.index(min(ls)))
            ls.pop(ls.index(min(ls)))
            range1 = max(ls) - min(ls)
        return range1

    def calTriggerPrice(self, n: int, direction: str):
        price = self.y[n]
        if direction == 'B':
            H = self.delta2h(self.RAPB_peak_price - self.RAPB_start_price)
            if price - self.y[n - 1] >= 0:
                if price > self.RAPB_peak_price:
                    self.RAPB_peak_price = self.y[n]
                if self.RAPB_strike is False and self.delta2h(price - self.RAPB_start_price) > 6:
                    self.RAPB_strike = True
                return price - 4 * self.multiplier
            elif price - self.y[n - 1] < 0:
                if self.RAPB_strike is False:
                    return self.RAPB_start_price - 4 * self.multiplier
                elif H < 10:
                    return self.RAPB_start_price + 2 / 3 * H * self.multiplier
                elif H < 20:
                    return self.RAPB_start_price + 4 / 5 * (H + 1) * self.multiplier
                elif H < 30:
                    return self.RAPB_start_price + 4 / 5 * (H + 5) * self.multiplier
                else:
                    return self.RAPB_peak_price - 7 * self.multiplier
        if direction == 'S':
            H = self.delta2h(self.RAPS_start_price - self.RAPS_nadir_price)
            if price - self.y[n - 1] <= 0:
                if price < self.RAPS_nadir_price:
                    self.RAPS_nadir_price = self.y[n]
                if self.RAPS_strike is False and self.delta2h(self.RAPS_start_price - price) > 6:
                    self.RAPS_strike = True
                return price + 4 * self.multiplier
            elif price - self.y[n - 1] > 0:
                if self.RAPS_strike is False:
                    return self.RAPS_start_price + 4 * self.multiplier
                elif H < 10:
                    return self.RAPS_start_price - 2 / 3 * H * self.multiplier
                elif H < 20:
                    return self.RAPS_start_price - 4 / 5 * (H + 1) * self.multiplier
                elif H < 30:
                    return self.RAPS_start_price - 4 / 5 * (H + 5) * self.multiplier
                else:
                    return self.RAPS_nadir_price + 7 * self.multiplier

    def delta2h(self, delta):
        return round(delta / self.multiplier)

    def support_lines(self, n):
        interval = 20
        last_x = 0
        n_x_ls1 = [self.x[0], ]
        n_y_ls1 = [self.y[0], ]
        for i in range(1, n - 1):
            if (self.y[i] <= self.y[i - 1] and self.y[i] <= self.y[i + 1]) or i - last_x >= interval:
                n_x_ls1.append(self.x[i])
                n_y_ls1.append(self.y[i])
                last_x = i
        n_x_ls1.append(self.x[n])
        n_y_ls1.append(self.y[n])
        last_x = 0
        n_x_ls2 = [n_x_ls1[0], ]
        n_y_ls2 = [n_y_ls1[0], ]
        for i in range(1, len(n_x_ls1) - 1):
            if (n_y_ls1[i] - n_y_ls1[i - 1]) * (n_y_ls1[i + 1] - n_y_ls1[i]) and i - last_x < interval > 0:
                continue
            else:
                n_x_ls2.append(n_x_ls1[i])
                n_y_ls2.append(n_y_ls1[i])
                last_x = i
        n_x_ls2.append(n_x_ls1[-1])
        n_y_ls2.append(n_y_ls1[-1])
        last_x = 0
        n_x_ls3 = [n_x_ls2[0], ]
        n_y_ls3 = [n_y_ls2[0], ]
        for i in range(1, len(n_x_ls2) - 1):
            if (n_y_ls2[i] <= n_y_ls2[i - 1] and n_y_ls2[i] <= n_y_ls2[i + 1]) or i - last_x >= interval:
                n_x_ls3.append(n_x_ls2[i])
                n_y_ls3.append(n_y_ls2[i])
                last_x = i
        n_x_ls3.append(n_x_ls2[-1])
        n_y_ls3.append(n_y_ls2[-1])
        last_x = - 2 * interval
        n_x_ls4 = list()
        n_y_ls4 = list()
        for x, y in zip(n_x_ls3, n_y_ls3):
            if x - last_x >= interval:
                n_x_ls4.append(x)
                n_y_ls4.append(y)
                last_x = x
        return n_x_ls4, n_y_ls4

    def press_lines(self, n):
        p_x_ls1 = [self.x[0], ]
        p_y_ls1 = [self.y[0], ]
        for i in range(1, n - 1):
            if self.y[i] >= self.y[i - 1] and self.y[i] >= self.y[i + 1]:
                p_x_ls1.append(self.x[i])
                p_y_ls1.append(self.y[i])
        p_x_ls1.append(self.x[n])
        p_y_ls1.append(self.y[n])
        p_x_ls2 = [p_x_ls1[0], ]
        p_y_ls2 = [p_y_ls1[0], ]
        for i in range(1, len(p_x_ls1) - 1):
            if (p_y_ls1[i] - p_y_ls1[i - 1]) * (p_y_ls1[i + 1] - p_y_ls1[i]) > 0:
                continue
            else:
                p_x_ls2.append(p_x_ls1[i])
                p_y_ls2.append(p_y_ls1[i])
        p_x_ls2.append(p_x_ls1[-1])
        p_y_ls2.append(p_y_ls1[-1])
        p_x_ls3 = [p_x_ls2[0], ]
        p_y_ls3 = [p_y_ls2[0], ]
        for i in range(1, len(p_x_ls2) - 1):
            if p_y_ls2[i] >= p_y_ls2[i - 1] and p_y_ls2[i] >= p_y_ls2[i + 1]:
                p_x_ls3.append(p_x_ls2[i])
                p_y_ls3.append(p_y_ls2[i])
        p_x_ls3.append(p_x_ls2[-1])
        p_y_ls3.append(p_y_ls2[-1])
        p_x_ls4 = [p_x_ls3[0], ]
        p_y_ls4 = [p_y_ls3[0], ]
        for i in range(1, len(p_x_ls3) - 1):
            if (p_y_ls3[i] - p_y_ls3[i - 1]) * (p_y_ls3[i + 1] - p_y_ls3[i]) > 0:
                continue
            else:
                p_x_ls4.append(p_x_ls3[i])
                p_y_ls4.append(p_y_ls3[i])
        p_x_ls4.append(p_x_ls3[-1])
        p_y_ls4.append(p_y_ls3[-1])
        p_x_ls5 = [p_x_ls4[0], ]
        p_y_ls5 = [p_y_ls4[0], ]
        for i in range(1, len(p_x_ls4) - 1):
            if p_y_ls4[i] >= p_y_ls4[i - 1] and p_y_ls4[i] >= p_y_ls4[i + 1]:
                p_x_ls5.append(p_x_ls4[i])
                p_y_ls5.append(p_y_ls4[i])
        p_x_ls5.append(p_x_ls4[-1])
        p_y_ls5.append(p_y_ls4[-1])
        return p_x_ls5, p_y_ls5

    def support_xs(self, p, q):
        if q - p >= 8:
            ls = list(range(p, q + 1))
            if q - p in {11, 12, 13}:
                ls.pop(7)
                ls.pop(-7)
            elif q - p == 14:
                ls.pop(7)
            elif q - p >= 15:
                ls.pop(7)
                ls.pop(-8)
            ls = ls[4: -4]
            simplified_y_ls = [self.y[k] for k in ls]
            min_position = simplified_y_ls.index(min(simplified_y_ls))
            pos = ls[min_position]
            return self.support_xs(p, pos) + [pos, ] + self.support_xs(pos, q)
        elif q - p in {4, 5, 6}:
            return [p, ]
        else:
            raise ValueError("Wrong interval: " + str(q - p))



    def Signal(self, n: int, direction: str):

        if direction == 'B':
            self.slope_list = self.slope_list_real
            self.yplus = self.y
            color = "gold"
            sign = 1
        elif direction == 'S':
            self.slope_list = [- t for t in self.slope_list_real]
            self.yplus = [- t for t in self.y]
            color = "deeppink"
            sign = - 1
        else:
            raise ValueError("Wrong direction: " + direction)

        # Close position part
        if direction == 'B' and n > 2 and self.RAPB_num > 0:
            b_trigger_price = self.calTriggerPrice(n, 'B')
            if self.y[n] < b_trigger_price:
                self.pnl += self.y[n] - self.RAPB_start_price
                if self.plot:
                    cl = "red" if self.y[n] > self.RAPB_start_price else "green"
                    self.ax.plot([self.x[n], ], [self.y[n], ], marker='x', markersize=8, color=color)
                    self.ax.plot([self.RAPB_start_pos, self.RAPB_start_pos + 0.001],
                                 [self.RAPB_start_price, self.y[n]], color=cl, linewidth=2)
                    self.ax.text(self.RAPB_start_pos, self.y[n], str(round(self.y[n] / self.RAPB_start_price * 100 - 100, 2)))
                if self.RAPB_sig_type.startswith("TRE"):
                    self.TREB_close_x = n
                self.initDailyParam(pos_type='B')
        if direction == 'S' and n > 2 and self.RAPS_num > 0:
            s_trigger_price = self.calTriggerPrice(n, 'S')
            if self.y[n] > s_trigger_price:
                self.pnl += self.RAPS_start_price - self.y[n]
                if self.plot:
                    cl = "red" if self.y[n] < self.RAPS_start_price else "blue"
                    self.ax.plot([self.x[n], ], [self.y[n], ], marker='x', markersize=8, color=color)
                    self.ax.plot([self.RAPS_start_pos, self.RAPS_start_pos + 0.001],
                                 [self.RAPS_start_price, 2 * self.RAPS_start_price - self.y[n]], color=cl)
                    self.ax.text(self.RAPS_start_pos, self.y[n] - 2 * self.RAPS_start_price,
                                 str(round(self.y[n] / self.RAPS_start_price * 100 - 100, 2)))
                self.initDailyParam(pos_type='S')




        # Open position part
        if self.RAPB_num > 0 and direction == 'B':
            return
        if self.RAPS_num > 0 and direction == 'S':
            return
        if n < 8:
            return
        h8, h7, h6, h5, h4, h3, h2, h1 = self.slope_list[n - 7: n + 1]
        sig_type = None
        check_buy_signal = True
        check_sell_signal = True

        # if check_buy_signal and direction == 'B' and self.count(2, 5, h1, h2):  # Check rapid condition
        #     sig_type, diff = "RAPB1", 2
        #     check_buy_signal = False
        #
        # if check_buy_signal and direction == 'B' and min([h1, h2, h3, h4, h5]) >= - 2:  # Check rapid condition
        #     r1 = self.previous_range(n - 5)
        #     if r1 is not None and sign * (self.y[n] - self.y[n - 5]) > 2 * r1:  # Check stable condition
        #         sig_type, diff = "RAPB2", 6
        #         check_buy_signal = False




        ############################### Cutting Line #############################################

        # if check_sell_signal and direction == 'S' and self.count(2, 5, h1, h2):  # Check rapid condition
        #     var1, var2, var3, var4, var5 = self.previous_trend(n - 2)
        #     if var1 > var2 and var2 > var3 and var3 > var5:  # Check stable condition
        #         sig_type, diff = "RAPS1", 2
        #         check_sell_signal = False
        #         # if self.plot:
        #         #     self.ax.text(self.x[n], self.y[n], '(' + str(var1) +',' + str(var2) + ',' + str(var3) +',' + str(var5) +')')
        #         #     self.ax.plot([self.x[n - 2 - 8], self.x[n - 2]], [self.y[n - 2 - 8], self.y[n - 2]], color="cyan")
        #         #     self.ax.plot([self.x[n - 2 - 30], self.x[n - 2]], [self.y[n - 2 - 30], self.y[n - 2]], color="blue")
        #         #     if n > 62:
        #         #         self.ax.plot([self.x[n - 2 - 60], self.x[n - 2]], [self.y[n - 2 - 60], self.y[n - 2]], "--")
        #         #     if n > 240:
        #         #         self.ax.plot([self.x[n - 2 - 240], self.x[n - 2]], [self.y[n - 2 - 240], self.y[n - 2]], "--")
        #
        # if check_sell_signal and direction == 'S':  # Check rapid condition
        #     var1, var2, var3, var4, var5 = self.previous_trend(n)
        #     if var1 <= 0.4 and var4 <= -0.02 and self.same_trend(var1, var2, var3, var4, var5):
        #         sig_type, diff = "RAPS2", 0
        #         check_sell_signal = False
        #         # if self.plot:
        #         #     self.ax.text(self.x[n], self.y[n], '(' + str(var1) +',' + str(var2) + ',' + str(var3) + ',' + str(var4) +',' + str(var5) +')')
        #         #     self.ax.plot([self.x[n - 8], self.x[n]], [self.y[n - 8], self.y[n]], color="cyan", linestyle="dashed")
        #         #     self.ax.plot([self.x[n - 30], self.x[n]], [self.y[n - 30], self.y[n]], color="blue", linestyle="dashed")
        #         #     if n > 60:
        #         #         self.ax.plot([self.x[n - 60], self.x[n]], [self.y[n - 60], self.y[n]], color="olive", linestyle="dashed")
        #         #     if n > 120:
        #         #         self.ax.plot([self.x[n - 120], self.x[n]], [self.y[n - 120], self.y[n]], color="violet", linestyle="dashed")
        #         #     if n > 240:
        #         #         self.ax.plot([self.x[n - 240], self.x[n]], [self.y[n - 240], self.y[n]], color="darkorange", linestyle="dashed")
        #





        if not check_buy_signal or not check_sell_signal:
            if self.plot:
                self.plotSignal(n, diff, color=color)
            if direction == 'B':
                self.RAPB_num = 1
                self.RAPB_sig_type = sig_type
                self.RAPB_start_pos = n
                self.RAPB_start_price = self.y[n]
                self.RAPB_peak_pos = n
                self.RAPB_peak_price = self.y[n]
                self.RAPB_strike = False
            else:
                self.RAPS_num = 1
                self.RAPS_sig_type = sig_type
                self.RAPS_start_pos = n
                self.RAPS_start_price = self.y[n]
                self.RAPS_nadir_pos = n
                self.RAPS_nadir_price = self.y[n]
                self.RAPS_strike = False

    def plotSignal(self, n, diff, color):
        self.ax.plot(self.x[n - diff: n + 1], self.y[n - diff: n + 1], color=color)
        # self.ax.plot([self.x[n], ], [self.y[n], ], marker='o', color=color, markersize=5)

    def volitility(self, ls):
        N = len(ls)
        assembled_ls = list()
        j = 0
        while j < N and ls[j] == 0:
            j += 1
        assembled_ls.append(ls[j])
        sign = np.sign(ls[j])
        for item in ls[j + 1:]:
            if np.sign(item) * sign >= 0:
                assembled_ls[-1] += item
            else:
                assembled_ls.append(item)
                sign *= -1
        if len(assembled_ls) == 0:
            raise ValueError("Empty assembled list!!!")
        var_ls = list()
        for item in assembled_ls:
            if abs(item) < 10:
                var_ls.append(item)
        if len(var_ls) == 0:
            return 0
        else:
            return round(np.std(var_ls), 1)

    def PNLcurve(self):
        df = self.re_df
        df.columns = ["date", "100return"]
        df.sort_values(by="date", inplace=True)
        df["daily_com"] = df.loc[:, "100return"].apply(lambda x: 1 + x / 100)
        df["asset"] = df["daily_com"].cumprod()
        final_asset = round(df["asset"].tolist()[-1], 2)
        plt.plot(range(len(df)), df["asset"])
        plt.savefig("asset_" + str(final_asset) + ".png")
        plt.close()
        df["level_return"] = df["100return"] * 5
        df["level_com"] = df["level_return"].apply(lambda x: 1 + x / 100)
        df["level_asset"] = df["level_com"].cumprod()
        level_asset = round(df["level_asset"].tolist()[-1], 2)
        plt.plot(range(len(df)), df["level_asset"])
        plt.savefig("level_asset_" + str(level_asset) + ".png")
        plt.close()
        df["log_asset"] = df["asset"].apply(lambda x: math.log(x))
        log_final_asset = round(df["log_asset"].tolist()[-1], 2)
        plt.plot(range(len(df)), df["log_asset"])
        plt.savefig("log_asset_" + str(log_final_asset) + ".png")
        plt.close()
        df["log_level_asset"] = df["level_asset"].apply(lambda x: math.log(x))
        log_level_asset = round(df["log_level_asset"].tolist()[-1], 2)
        plt.plot(range(len(df)), df["log_level_asset"])
        plt.savefig("log_level_asset_" + str(log_level_asset) + ".png")
        plt.close()
        print(final_asset, level_asset)


if __name__ == "__main__":
    obj = Strategy("btc")
    obj.multi_backtest(plot=True)
    obj.PNLcurve()