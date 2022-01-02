import json
import time
import pickle
import numpy as np
from scipy import linalg
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import time

# Indexed pairs of joints that form a bone/link in a skeleton, 
# from the 32 joints cataloged here: https://docs.microsoft.com/en-us/azure/kinect-dk/body-joints
SEGMENT_PAIRS =[[1, 0], 
                [2, 1], 
                [3, 2], 
                [4, 2], 
                [5, 4], 
                [6, 5], 
                [7, 6], 
                [8, 7], 
                [9, 8],
                [10, 7], 
                [11, 2], 
                [12, 11], 
                [13, 12], 
                [14, 13], 
                [15, 14], 
                [16, 15], 
                [17, 14], 
                [18, 0], 
                [19, 18],
                [20, 19], 
                [21, 20], 
                [22, 0], 
                [23, 22], 
                [24, 23], 
                [25, 24], 
                [26, 3], 
                [27, 26],
                [28, 26], 
                [29, 26], 
                [30, 26], 
                [31, 26]]

# makeshift index bins for 4 bodies in the data file we are working with
# 32 joints * 4 bodies = 128 joints
all_indexes = range(len(SEGMENT_PAIRS)*4)


def json_maker(body3dObj, body_id, frame):
    """
    Take body object and provide data into JSON format structure
    """
    dict = {
            "frame": frame,
            "BodyID": body_id,
            "Joints" : [                   
                {
                    "name": joint.get_name(),
                    "pos" : {
                        "x" : joint.get_coordinates()[0],
                        "y" : joint.get_coordinates()[1],
                        "z" : joint.get_coordinates()[2]
                    },
                    "rot" : {
                        "w" : joint.get_orientations()[0],
                        "x" : joint.get_orientations()[1],
                        "y" : joint.get_orientations()[2],
                        "z" : joint.get_orientations()[3]
                    },
                    "confidence": joint.confidence_level,
                    
                } for joint in body3dObj.joints
            ]
    }
    return json.dumps(dict)


def unjson_body_ref(bodyJsons):
    """
    Convert JSON format dictionary into body object
    For plot tests, only implemented for first body
    """
    bodyRefs = []

    for strObj in bodyJsons:
        bodyJson = json.loads(strObj)
        # only first body
        if bodyJson["BodyID"] == 0:
            bodyJoints = bodyJson["Joints"]
            bodyFrame = bodyJson["frame"]
            bodyTuple = (bodyFrame, bodyJoints)
            bodyRefs.append(bodyTuple)
    return bodyRefs


def get_bones(joints):
    """
    Provide 3D line coords for pairs of joints in SEGMENT_PAIRS
    """
    xs_list = []
    ys_list = []
    zs_list = []
    joints = np.transpose(joints)
    # Draws bones
    for segmentId in range(len(SEGMENT_PAIRS)):
        segment_pair = SEGMENT_PAIRS[segmentId]
        x1 = joints[segment_pair[0]][0]
        x2 = joints[segment_pair[1]][0]
        #xs = np.linspace(x1, x2)

        y1 = joints[segment_pair[0]][1]
        y2 = joints[segment_pair[1]][1]
        #ys = np.linspace(y1, y2)

        z1 = joints[segment_pair[0]][2]
        z2 = joints[segment_pair[1]][2]
        #zs = np.linspace(z1, z2)
            
        xs = (x1, x2)
        ys = (y1, y2)
        zs = (z1, z2)
        xs_list.append(xs)
        ys_list.append(ys)
        zs_list.append(zs)

    return (xs_list, ys_list, zs_list)


def set_axes_equal(ax):
    '''
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    axes = _set_axes_radius(ax, origin, radius)
    return axes


def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])
    return ax


def initialize_plots():
    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    (x_min, x_max), (y_min, y_max), (z_min, z_max) = ((-15000, 2600), (-3000, 2500), (0, 6000))
    axes.set_xlabel('X')
    axes.set_ylabel('Y')
    axes.set_zlabel('Z')
    axes.set_xlim(left=x_max, right=x_min)
    axes.set_ylim(bottom=y_min, top=y_max)
    axes.set_zlim(bottom=z_max, top=z_min)
    axes.grid(False)
    axes.set_box_aspect([0.5, 0.5, 1]) 
    axes.set_proj_type('ortho') 
    axes = set_axes_equal(axes)
    axes.set_axis_off()
    axes.view_init(azim = -90, elev = -60)
    
    plots = {index: axes.plot([0], [0], [0], 'bo-', markersize = 0)[0] for index in all_indexes}

    return fig, axes, plots

 
def get_joint_positions(bodyJoints):
    jointPositions = []
    for joint in bodyJoints:
        jointPosx = joint["pos"]["x"]
        jointPosy = joint["pos"]["y"]
        jointPosz = joint["pos"]["z"]
        jointPos = [jointPosx, jointPosy, jointPosz]
        jointPositions.append(jointPos)
    return np.transpose(np.array(jointPositions))


def animate3D(all_body_refs, fig, plots, out_filename = "outfile.mp4", show = True):
    '''
    Creates an animation from skeleton data
    Args:
        all_body_refs ((frame, skeleton_list)): The data to animate
        out_filename (string): The file in which to save the video
        show (bool): Whether or not to preview the video before saving
    '''
    # Note that we make the assumption that the interval between frames is equal throughout. This
    interval_microseconds = 100
    interval = int(interval_microseconds / 1e3)
    (body_frames1, body_frames2, body_frames3, body_frames4) = all_body_refs

    def init():
        '''Initializes the animation'''
        for index in all_indexes:
            plots[index].set_xdata(np.asarray([0]))
            plots[index].set_ydata(np.asarray([0]))
            plots[index].set_3d_properties(np.asarray([0]))
        return iter(tuple(plots.values()))

    def animate(i):
        global frame_count
        global c1
        global c2 
        global c3 
        global c4

        global R32
        global T32
        global R21
        global T21
        global R14
        global T14

        '''Render each frame'''
        time_start = time.time()

        body1_ref = int(body_frames1[c1][0])
        body2_ref = int(body_frames2[c2][0])
        body3_ref = int(body_frames3[c3][0])
        body4_ref = int(body_frames4[c4][0])

        joints1 = []
        joints2 = []
        joints3 = []
        joints4 = []

        if frame_count == body1_ref:
            joint_positions1 = get_joint_positions(body_frames1[c1][1])
            c1+=1
            if np.all(joint_positions1):
                joints1.append(joint_positions1)

        if frame_count == body2_ref:
            joint_positions2 = get_joint_positions(body_frames2[c2][1])
            c2+=1
            if np.all(joint_positions2):
                _joints2 = R21.dot(joint_positions2) + T21
                joints2.append(_joints2)

        if frame_count == body3_ref:
            joint_positions3 = get_joint_positions(body_frames3[c3][1])
            c3+=1
            if np.all(joint_positions3):
                __joints4 = R32.dot(joint_positions3) + T32
                _joints4 = R21.dot(__joints4) + T21
                joints3.append(_joints4)

        if frame_count == body4_ref:
            joint_positions4 = get_joint_positions(body_frames4[c4][1])
            c4+=1
            if np.all(joint_positions4):
                _joints4 = np.transpose(R14).dot(joint_positions4) - 10*T14 
                joints4.append(_joints4)

        frame_count+=1

        bones1 = []
        bones2 = []
        bones3 = []
        bones4 = []

        if len(joints1) > 0:
            bones1 = get_bones(joints1[0])
        if len(joints2) > 0:
            bones2 = get_bones(joints2[0])
        if len(joints3) > 0:
            bones3 = get_bones(joints3[0])
        if len(joints4) > 0:
            bones4 = get_bones(joints4[0])
        
        # Draws bones
        bones_from_all_cameras = [bones1, bones2, bones3, bones4]

        for b, set_of_bones in enumerate(bones_from_all_cameras):
            color = ((b + 1)*0.25, (b + 1)* 0.2, (b + 1)*0.1)

            if set_of_bones != []:
                x_orientations = 3*np.asarray(set_of_bones[0])
                y_orientations = 3*np.asarray(set_of_bones[1])
                z_orientations = 3*np.asarray(set_of_bones[2])

                for bone_id in range(len(x_orientations)):
                    if b == 0:
                        index = bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)
                    if b == 1:
                        index = 31 + bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)
                    if b == 2:
                        index = 31*2 + bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)
                    if b == 3:
                        index = 31*3 + bone_id
                        plots[index].set_xdata(x_orientations[bone_id])
                        plots[index].set_ydata(y_orientations[bone_id])
                        plots[index].set_3d_properties(z_orientations[bone_id])
                        plots[index].set_markersize(1)
                        plots[index].set_color(color)

        output = iter(tuple(plots.values()))
        #print("it takes {} seconds to plot a frame".format(time.time()-time_start))
        return output 

    #log.info('Creating animation')
    video = animation.FuncAnimation(
        fig,
        animate, init,
        interval=interval,
        blit=True
    )

    if show:
        plt.show()

    
    #log.info(f'Saving video to {out_filename}')
    #video.save(out_filename, fps=30, extra_args=['-vcodec', 'libx264'])

# Plot frame global vars
frame_count = 1
c1, c2, c3, c4 = 0, 0, 0, 0


def DLT(P1, P2, point1, point2):
    """
    Use projection matrices and 2D points from two views to get 3D points from left view
    """
    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]]
    A = np.array(A).reshape((4,4))
    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)
    return Vh[3,0:3]/Vh[3,3]

"""
Transformation matrices to go from Cam U to Cam V,
    XV = RUV*XU + TUV 
              WINDOW
            C4 ---- C1
            |        |
         P  |        | S
         I  |        | E
         C  |        | R       
         T  |        | V
         U  |        | E
         R  |        | R
         E  |        |
            |        |
            C3 ---- C2
               BOARD

These calibrated values need to be replaced with new ones if cameras move, or ideally everytime, new data is taken for testing purposes
"""
R32 = np.array([[-0.05362146, -0.40490901,  0.91278334],
       [ 0.52123706,  0.76833319,  0.37145126],
       [-0.8517257 ,  0.49569427,  0.16985444]])

T32 = np.array([[-2589.03803651],
       [-1410.73910838],
       [ 2072.55890949]])

R21 = np.array([[ 0.50775171, -0.48538798,  0.71174905],
       [ 0.35416218,  0.87073587,  0.34115714],
       [-0.78533901,  0.07885148,  0.61402369]])

T21 = np.array([[-9430.83543993],
       [-2645.29587609],
       [ 9951.04442354]])

R14 = np.array([[-0.00594625, -0.42046097,  0.90729114],
    [ 0.48471258,  0.79238132,  0.37038568],
    [-0.87465328,  0.44197784,  0.19909102]])

T14 = np.array([[-34.06293941],
    [-18.73061499],
    [ 26.62438133]])

# Camera matrix, ideally same among all cameras, but can be different, 
# so use separate for more accurate outputs
K = np.array([[504.399017334, 0,            327.475708008],
              [0,            504.499676514, 339.489501195],
              [0,            0,             1            ]])


if __name__ == "__main__":
    matplotlib.use("TkAgg")

    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0],[0],[0]]], axis = -1)
    P1 = K @ RT1 #projection matrix for C1
    
    # RT matrix for C4 are obtained from stereo calibration.
    RT4 = np.concatenate([R14, T14], axis = -1)
    P4 = K @ RT4 #projection matrix for C4

    with open("./all_body_data.pkl", "rb") as f:
        allBodyJsons = pickle.load(f)
    
    [body1Jsons, body2Jsons, body3Jsons, body4Jsons] = allBodyJsons

    bodyRef1 = unjson_body_ref(body1Jsons)
    bodyRef2 = unjson_body_ref(body2Jsons)
    bodyRef3 = unjson_body_ref(body3Jsons)
    bodyRef4 = unjson_body_ref(body4Jsons)
    allBodyRefs = [bodyRef1, bodyRef2, bodyRef3, bodyRef4]

    fig, axes, plots = initialize_plots()
    animate3D(allBodyRefs, fig, plots)

