import sys
import matplotlib.pyplot as plt
import simutils as su
import numpy as np
import matplotlib.image as mpimg


def ground_track_plot(lat_deg, lon_deg, img_path='3DModels/earth.jpg', save_path=None):
    # Ensure longitude in [-180, 180]
    lon_deg = (lon_deg + 180) % 360 - 180

    # Detect jumps > 180 deg (crossing the map edge)
    dlon = np.diff(lon_deg)
    jump_idx = np.where(np.abs(dlon) > 180)[0]

    # Insert NaN to break the line
    lat_fixed = lat_deg.copy()
    lon_fixed = lon_deg.copy()
    for idx in reversed(jump_idx):
        lat_fixed = np.insert(lat_fixed, idx + 1, np.nan)
        lon_fixed = np.insert(lon_fixed, idx + 1, np.nan)

    # Plot
    img = mpimg.imread(img_path)
    plt.figure(figsize=(12, 6))
    plt.imshow(img, extent=(-180, 180, -90, 90))
    plt.plot(lon_fixed, lat_fixed, 'r', linewidth=1)
    plt.xlabel("Longitude [deg]")
    plt.ylabel("Latitude [deg]")
    plt.title("Satellite Ground Track")
    plt.xlim([-180, 180])
    plt.ylim([-90, 90])

    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()

def log_ground_track(scenario, t, r_i):
    # Convert ECI -> ECEF using scenario.q_E
    q_conj = su.Quaternion(scenario.q_E)
    q_conj.conjugate()
    r_ecef = q_conj.rotate(r_i)

    lat = np.arcsin(r_ecef[2] / np.linalg.norm(r_ecef))
    lon = np.arctan2(r_ecef[1], r_ecef[0])
    lon = (lon + np.pi) % (2 * np.pi) - np.pi

    entry = np.array([t, lat, lon])

    if scenario.ground_track is None:
        scenario.ground_track = entry.reshape(1, -1)
    else:
        scenario.ground_track = np.vstack((scenario.ground_track, entry))

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
