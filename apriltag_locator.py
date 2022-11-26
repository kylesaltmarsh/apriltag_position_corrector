import numpy as np
from scipy.optimize import least_squares
import math
import re
import os
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import json
import pandas as pd

class GetAprilTagPose:
    """
    Take a csv of surveyed target positions on april tag plates in an (X,Y,Z) coordinate system
    Perform a least squares fit to find the pose and quaternion for each april tag by minimising the residuals of the surveyed target positions
    """
    def __init__(self, targets_csv_name):
        """
        Create translation vectors in april tag and targets local xyz coordinate system
        Create dictionary object stores
        """
        self.apriltag_to_target_xyz = {}
        self.apriltag_to_target_xyz['Centre'] = [0,0.2,0]
        self.apriltag_to_target_xyz['Top Right'] = [0.08,0.28,0]
        self.apriltag_to_target_xyz['Bottom Right'] = [0.08,-0.08,0]
        self.apriltag_to_target_xyz['Bottom Left'] = [-0.08,-0.08,0]
        self.apriltag_to_target_xyz['Top Left'] = [-0.08,0.28,0]     
                        
        self.targets_csv_name = targets_csv_name
        measured_targets = pd.read_csv(targets_csv_name + '.csv')
        measured_targets['Coordinates'] = measured_targets.apply(lambda row: [row['X'], row['Y'], row['Z']], axis=1)
        measured_targets['Name'] = measured_targets['Name'].apply(lambda x: re.findall(r'\d+',x)[0])

        self.measured_targets={}
        for key in measured_targets['Name'].unique():
            self.measured_targets[key] = dict(zip(measured_targets[measured_targets['Name']==key].Description, measured_targets[measured_targets['Name']==key].Coordinates))
        
        self.apriltag_poses = {}

        if not os.path.exists(targets_csv_name):
            os.mkdir(targets_csv_name)

    def apriltag_to_targets_quaternion_cost_function(self, pose):
        """
        Cost function that calculates the residual vector between the surveyed target locations and the predicted target locations in (X,Y,Z) using quaternions
        """  
        q = Quaternion(pose[3:7]).normalised

        residuals = {}

        for key in self.apriltag_to_target_xyz:
            # find difference between corrected and extracted target location for each target
            residuals[key] = np.subtract(self.measured_targets[self.apriltag_name][key],np.add(pose[0:3],q.rotate(self.apriltag_to_target_xyz[key])))

        residuals = [residuals[k].tolist() for k in residuals]
        residuals = np.array([item for sublist in residuals for item in sublist])

        return residuals

    def get_apriltag_pose(self,apriltag_name):
        """
        Return the pose (position and rotation) of the april tag
        """
        self.apriltag_name = apriltag_name
        pose0 = np.array([0,0,0,1,0,0,0])

        sol = least_squares(self.apriltag_to_targets_quaternion_cost_function, pose0, jac="3-point")

        residuals = self.apriltag_to_targets_quaternion_cost_function(sol['x'])

        residual_rms = math.sqrt(sum(residuals**2))
        residual_std = np.std(residuals)

        self.build_3d_plot(sol['x'],apriltag_name)

        q = Quaternion(sol['x'][3:7]).normalised.elements

        self.apriltag_poses[apriltag_name] = {
            'X': sol['x'][0],
            'Y': sol['x'][1],
            'Z': sol['x'][2],
            'qw': q[0],
            'qx': q[1],
            'qy': q[2],
            'qz': q[3],
            'Residual_RMS': residual_rms,
            'Residual_STD': residual_std
        }

    def get_all_apriltag_pose(self):
        """
        Run get_apriltag_pose for all april tags
        """ 
        for target_name in self.measured_targets.keys():
            self.get_apriltag_pose(target_name)

        with open(self.targets_csv_name + '/AprilTag_Poses.json', 'w') as fp:
            json.dump(self.apriltag_poses, fp)

        self.build_full_2d_plot()

    def build_3d_plot(self, apriltag_pose, apriltag_name):
        """
        Plot the pose and residuals of all the april tags in 3D
        """  
        x_real = [i[0] for i in self.measured_targets[apriltag_name].values()]
        y_real = [i[1] for i in self.measured_targets[apriltag_name].values()]
        z_real = [i[2] for i in self.measured_targets[apriltag_name].values()]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x_real, y_real, z_real, c='blue', label='Real')

        residuals = self.apriltag_to_targets_quaternion_cost_function(apriltag_pose)
        x_predicted = x_real + residuals[0::3]
        y_predicted = y_real + residuals[1::3]
        z_predicted = z_real + residuals[2::3]
        ax.scatter(x_predicted,y_predicted,z_predicted, c='red', s=20, alpha=0.25, label='Estimate')
        
        scale = -100
        for i in range(0, 5, 1):
            if i == 0:
                ax.plot3D([x_predicted[i],x_predicted[i] + scale*residuals[0::3][i]], [y_predicted[i],y_predicted[i] + scale*residuals[1::3][i]],[z_predicted[i],z_predicted[i] + scale*residuals[2::3][i]], 'r-', label='100 x Residual')
            else:
                ax.plot3D([x_predicted[i],x_predicted[i] + scale*residuals[0::3][i]], [y_predicted[i],y_predicted[i] + scale*residuals[1::3][i]],[z_predicted[i],z_predicted[i] + scale*residuals[2::3][i]], 'r-')

        x_mid = (max(x_real) + min(x_real))/2
        ax.set_xlim(x_mid - 0.6,x_mid + 0.6)
        y_mid = (max(y_real) + min(y_real))/2
        ax.set_ylim(y_mid - 0.6,y_mid + 0.6)
        z_mid = (max(z_real) + min(z_real))/2
        ax.set_zlim(z_mid - 0.3,z_mid + 0.3)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        ax.legend()

        plt.savefig(self.targets_csv_name + '/' + apriltag_name + '.png', dpi=200)

    def build_full_2d_plot(self):
        """
        Plot the locations of all the april tags in 2D
        """  
        X = [self.apriltag_poses[key]['X'] for key in self.apriltag_poses.keys()]
        Y = [self.apriltag_poses[key]['Y'] for key in self.apriltag_poses.keys()]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(X, Y, c='blue')

        for i in range(len(X)): 
            ax.text(X[i],Y[i], '%s' % (list(self.measured_targets.keys())[i]), size=12, zorder=1, color='k') 

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')

        plt.savefig(self.targets_csv_name + '/All_AprilTags_2D.png', dpi=200)

if __name__ == '__main__':
    run = GetAprilTagPose("Tests/TargetCoordinates")
    run.get_all_apriltag_pose()