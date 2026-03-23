import sys
import matplotlib.pyplot as plt
import numpy as np

def line_plot(file_path):
    data = np.loadtxt(file_path)
    t = data[:,0]
    _, ax = plt.subplots()
    for col in data[:,1:].T:
        ax.plot(t,col)
    plt.show()

def main(argv):
    if len(argv) == 3:
        plot_type = argv[1]
        file_path = argv[2]
        if plot_type == 'lineplot':
            line_plot(file_path)
        else:
            print("Plot type not supported yet.")
    else:
        print("Wrong number of arguments. Expected 2 (plot_type, file_path) got {}".format(len(argv)-1))

if __name__ == "__main__":
    main(sys.argv)
