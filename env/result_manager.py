import os, sys
from datetime import datetime
import matplotlib.pyplot as plt
from .import script_runner, values_delivery
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

class result_manager:
    result_txt_name = "result.txt"
    result_graph_dir_name = "result_graphs"
    def __init__(self, proj_path, conf):
        self.proj_path = proj_path
        self.result_path = os.path.join(
            proj_path, result_manager.result_txt_name)
        self.result_graph_path = os.path.join(
            proj_path, result_manager.result_graph_dir_name)
        self.conf = conf

    def add_timestamp(self):
        script_runner.append_to_file(
            f"### Executed at {datetime.now()} ###\n",
            self.result_path)

    def create_graph_directory(self):
        script_runner.mkdir(self.result_graph_path)

    def get_fvalues(self):
        return values_delivery.extract_values_from_txt(self.result_path)

    def update(self, test, time):
        tname = test.name
        bin_dir = os.path.join(self.proj_path, 'build/',
            self.conf.manufacturer(), tname)
        report_path = os.path.join(bin_dir, "report.txt")
        script_runner.cat(report_path)
        # copy from report.txt to result.txt
        try:
            report = open(report_path, 'r')
        except FileNotFoundError:
            print(f"Report file {report_path} doesn't exist.")
            return
        result = open(self.result_path, 'a')
        for l in report.readlines():
            result.write(l)
            if l[0] == '@':
                figure_name = l[1:l.find(':')].\
                    replace(' ', '_').replace('(', '_').replace(')', '_')
                self.draw_graph(l, os.path.join(
                    self.result_graph_path,
                    f"{test.name}_{figure_name}.png"))
        result.write(f"Time consumed: {time}\n\n")
        result.close() ; report.close()

    def draw_graph(self, line, out_path):
        # format: @title:dataN:xlabel:x0:dx:ylabel:y0,y1,...,yn\n
        # or, instead of :x0:dx:, xdata manually with :p:x0,x1,...,xn:
        # optional line drawings: ..., yn:a,b,label\n where y=ax+b
        plt.clf()
        if line[0] != '@':
            exit(1)
        tokens = line[1:].replace('\n', '').split(':')
        title = tokens[0] ; dataN = int(tokens[1])
        # xdata
        xlabel = tokens[2]
        max_x = 0 ; max_y = 0 ; xdata = []
        if tokens[3] != 'p':
            x0 = float(tokens[3]) ; dx = float(tokens[4])
            for i in range(dataN):
                xdata += [x0 + i * dx]
            max_x = x0 + (dataN-1) * dx
        else:
            xstrs = tokens[4].split(',')
            for i in range(dataN):
                xdata += [float(xstrs[i])]
                max_x = max(max_x, xdata[i])
        # ydata
        ylabel = tokens[5]
        data_connect_with_line = False
        if ylabel[0:2] == "--":
            # should connect dots with lines
            data_connect_with_line = True
            ylabel = ylabel[2:]
        ydata = [] ; ystrs = tokens[6].split(',')
        for i in range(dataN):
            ydata += [float(ystrs[i])]
            max_y = max(max_y, ydata[i])
        # additional line
        if len(tokens) > 7:
            slope = float(tokens[7].split(',')[0])
            y_itcpt = float(tokens[7].split(',')[1])
            linelabel = tokens[7].split(',')[2]
            addline_x = []; addline_y = []
            if slope == 0:
                addline_x = [0, max_x] ; addline_y = [y_itcpt, y_itcpt]
            else:
                x_at_maxy = (max_y - y_itcpt) / slope
                if x_at_maxy < max_x:
                    addline_x = [0, x_at_maxy] ; addline_y = [y_itcpt, max_y]
                else:
                    addline_x = [0, max_x] ; addline_y = [y_itcpt, y_itcpt+slope*max_x]
            plt.plot(addline_x, addline_y, 'k--', label=linelabel)
        # plot
        if data_connect_with_line:
            plt.plot(xdata, ydata, 'ko--')
        else:
            plt.plot(xdata, ydata, 'ko')
        if len(tokens) > 7:
            plt.legend()
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.savefig(out_path)
        plt.clf()