import gpxpy
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from vincenty import vincenty
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import random
import sys
import easygui
import os


class Process_GPX(object):
    def __init__(self, gpx_file, my_dir):
        self.gpx_file = gpx_file
        self.my_dir = my_dir
            
        "Import GPS Data"
        with open(self.gpx_file) as fh:
            gpx_data = gpxpy.parse(fh)
            self.segment = gpx_data.tracks[0].segments[0]
            self.coords = pd.DataFrame([
            {'lat': p.latitude,
            'lon': p.longitude,
            'ele': p.elevation,
            } for p in self.segment.points])

    def times(self):
        "Compute delta between timestamps"
        self.times = pd.Series([p.time for p in self.segment.points], name='time')
        self.dt = np.diff(self.times.values) / np.timedelta64(1, 's')
        
    def distances(self):
        "Find distance between points using Vincenty and Geodesic methods"
        self.vx = []
        for i in range(len(self.coords.lat)-1):
                vincenty_distance = vincenty([self.coords.lat[i], self.coords.lon[i]],[self.coords.lat[i+1], self.coords.lon[i+1]])*1000 ## ???? why 1000?
                self.vx.append(vincenty_distance)

        self.vy = []
        for i in range(len(self.coords.lat)-1):
                geodesic_distance = geodesic([self.coords.lat[i], self.coords.lon[i]],[self.coords.lat[i+1], self.coords.lon[i+1]])*1000 ## ???? why 1000?
                self.vy.append(geodesic_distance)

        self.ele = []
        for i in range(len(self.coords.lat)-1):
                self.ele.append(self.coords.ele[i])

    def remove_outliers(self):
        #remove outliers
        bad_values = np.nonzero(self.dt > 10) ## removing anytime timestep with dt > 10s
        self.dt = np.delete(self.dt, bad_values)
        self.vx = np.delete(self.vx, bad_values)
        self.ele = np.delete(self.ele, bad_values)
        
    def totals(self):
        self.total_distance = sum(self.dt*self.vx)
        self.total_time = sum(self.dt)
        self.mean_time_step = np.mean(self.dt)

    def compute_velocity_and_remove_outliers(self):
        "Compute and plot velocity"
        self.velocity = self.vx/self.dt
        bad_values = np.nonzero(self.velocity > 8)
        self.dt = np.delete(self.dt, bad_values)
        self.vx = np.delete(self.vx, bad_values)
        self.ele = np.delete(self.ele, bad_values)
        self.velocity = np.delete(self.velocity, bad_values)
        
    def define_time(self):
        self.time = [self.total_time*i/len(self.dt) for i in range(len(self.dt))]
        self.time_ = np.zeros(len(self.dt))
        for i in range(len(self.dt)-1):
            self.time_[i+1] = self.time_[i] + self.dt[i]
        self.save_file(self.time, 'time')
        self.save_file(self.time_, 'time_')
        
    def smoothen_velocity(self, window):
        self.velocity_1 = []
        self.time_1 = []
        for i in range(len(self.velocity) - window):
            self.velocity_1.append(np.mean(self.velocity[i:i+window-1]))
            self.time_1.append(np.mean(self.time[i:i+window-1]))
        self.save_file(self.velocity_1, 'velocity_1')
        self.save_file(self.time_1, 'time_1')
        
    def define_accel(self, window):
        self.accel = []
        self.time_2 = []
        for i in range(len(self.velocity) - window):
            data_y = np.array(self.velocity[i:i+window-1])
            data_x = np.array(self.time[i:i+window-1])
            model = LinearRegression().fit(data_x.reshape((-1, 1)), data_y)
            self.accel.append(model.coef_)
            self.time_2.append(np.mean(self.time[i:i+window-1]))
        self.save_file(self.accel, 'accel')
        self.save_file(self.time_2, 'time_2')
            
    def define_cad(self):
        self.cad = np.ones(len(self.time_))
        walk = 120
        run = 180
        self.cad[:70] *= walk
        self.cad[70:] *= run
        for i in range(len(self.cad)):
            self.cad[i] += 5*np.random.normal()
        self.save_file(self.cad, 'cad')
    
    def make_plots(self):
        
        # Plot velocity
        plt.figure(1, figsize=(10,2))
        plt.plot(self.time, self.velocity, '.')
        plt.xlabel('time')
        plt.ylabel('velocity')
        plt.title('Plot of Velocity vs Time')
        plt.plot(self.time_1, self.velocity_1)
        plt.xlabel('time')
        plt.ylabel('velocity')
        plt.title('Plot of Velocity vs Time (Smooth)')
        plt.xlim([0, 2500])
        plt.savefig(self.my_dir + '/vel_v_time.png', dpi=600, format='png')
        
        # Plot acceleration
        plt.figure(2, figsize=(10,2))
        plt.plot(self.time_2, self.accel,'.')
        plt.xlabel('time')
        plt.ylabel('accel')
        plt.title('Plot of Accel vs Time (Smooth)')
        plt.xlim([0,2500])
        plt.plot([0,2500], [0,0],':',color='darkgreen')
        plt.savefig(self.my_dir + '/accel_v_time.png', dpi=600, format='png')
        
        # Plot elevation
        plt.figure(3, figsize=(10,2))
        plt.plot(self.time, self.ele, color='orange')
        plt.xlabel('time')
        plt.ylabel('Elevation')
        plt.title('Plot of elevation vs Time')
        plt.xlim([0,2500])
        plt.savefig(self.my_dir + '/ele_v_time.png', dpi=600, format='png')
        
        # Plot Cadence
        plt.figure(4, figsize=(10,2))
        plt.plot(self.time_, self.cad, '.',color='g')
        plt.xlabel('time')
        plt.ylabel('Cadence')
        plt.title('Mock Cadence vs Time')
        plt.xlim([0,2500])
        plt.savefig(self.my_dir + '/cad_v_time.png', dpi=600, format='png')
        
        plt.show()
        
    def save_file(self, my_list, list_name):
        if type(my_list) != list:
            my_list = my_list.tolist()
        with open(self.my_dir + '/' + list_name+'.txt', 'w') as f:
            for item in my_list:
                f.write("%s\n" % item)

def main(gpx_file, my_dir):
    gpx_processor = Process_GPX(gpx_file, my_dir)
    gpx_processor.times()
    gpx_processor.distances()
    gpx_processor.remove_outliers()
    gpx_processor.totals()
    gpx_processor.compute_velocity_and_remove_outliers()
    gpx_processor.define_time()
    window_vel = 30
    gpx_processor.smoothen_velocity(window_vel)
    window_accel = 30
    gpx_processor.define_accel(window_accel)
    gpx_processor.define_cad()
    gpx_processor.make_plots()
    

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        
        file_path = easygui.fileopenbox()
        
    file_name = os.path.basename(file_path)
    head, tail = os.path.splitext(file_name)
    my_dir = os.path.dirname(os.path.realpath(__file__)) + '/' + head
    
    if not os.path.exists(my_dir):
        sys_command = 'mkdir ' + my_dir
        os.system(sys_command)
    print(my_dir)
    sys_command = 'cp ' + file_name + ' ' + my_dir
    os.system(sys_command)
    
    main(file_path, my_dir)
