# - - - - - - - - - - - - - - - -
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Pierre Nagorny, PhD student at SYMME labs
# 
# Experimental setup :
# 400 pictures per full rotation
# - - - - - - - - - - - - - - - -

"""
=====
tomo_prod
=====
tomo_prod combine multiple library for tomographic reconstruction on a specific experimental setup
"""


# for Python 2-3 compatibility:
from __future__ import print_function, division

try:
    import re  # for string processing
    import os
    from pathlib import Path
    import sys  # for folder scanning
    import joblib  # joblib version: 0.9.4
    import time  # for performance timing
    import numpy as np  # for matrix works
    import PIL  # for image resizing
    from PIL import Image
    import matplotlib.pyplot as plt
    import progressbar

    import tomopy  # for inverse tomography transformation

    # import vtk #for 3D visualization but can't compile it on my Sierra
    # import PyOpenGL #for 3D visualization using full OpenGL functionality available on your system
    import PyQt5  # for vispy visualisation
    from itertools import cycle
    import vispy  # for interractive 3D plotting
    from vispy import app, scene, io
    from vispy.scene.visuals import Text
    from vispy.color import get_colormaps, BaseColormap
    from vispy.visuals.transforms import (
        STTransform, PolarTransform, AffineTransform)
    # from vispy.visuals.transforms import MatrixTransform #MatrixTransform is used for volume rotation #TODO: wait VisPy 0.5.0 to re-add MatrixTransform
    import imageio  # for .gif recording

except ImportError as exc:
    sys.stderr.write("Error importing dependancies: ({})".format(exc))
    m = re.search('\'(.+?)\'', format(exc))
    print('You need some librairies ! Please try to install it by runing:')
    print('pip install %s' % m.group(1))


def get_images():
    # CREATE TOMOGRAPHIC IMAGES MATRIX FROM FILES
    input_dir = './raw_input/'
    print('Let\'s create tomomatrix from raw images')
    print('input directory:' + input_dir)
    print('Reshaping image to size %d x %d pixels' % (img_size, img_size))

    tclass = [d for d in os.listdir(input_dir)]
    # tclass = tclass[0:202] #we only need 202 images to do a 180deg rotation
    measurements_matrix = []
    counter = 0
    writer = imageio.get_writer('raw_tomo.mp4', fps=50, quality=8)
    bar_crop = progressbar.ProgressBar(
        redirect_stdout=True, max_value=len(tclass))
    bar_crop.start()
    for x in tclass:
        try:
            img = Image.open(os.path.join(input_dir + '/' + x))
            # Hand crafted cropping values : #TODO: automatize or user selectable
            # bottom-500
            img = img.crop((1300, 1700, img.size[0] - 550, img.size[1] - 600))
            img = img.resize((int(img_size), int(img_size)), Image.ANTIALIAS)
            measurements_matrix.append(np.asarray(img))
            writer.append_data(im)
        except Exception:
            print("Error resize file : %s - %s " % x)
        counter += 1
        bar_crop.update(counter)
    bar_crop.finish()
    writer.close()

    # Now we have 400 images (256,256)
    mat = np.array(measurements_matrix)
    print('tomo matrix created! %d image resized. A rotation .mp4 was also created ;!' % counter)


    # Let's save our measurements_matrix
    joblib.dump(mat, 'TOMO_360_angle_images.pkl')
    print('tomo matrix saved to TOMO_360_angle_images.pkl')
    # First dimension: Capture angles
    # Second dimension: Image width X
    # Third dimension: Image height Y


def sample_stack_Angles(stack, rows=5, cols=5, start_with=0, show_every=16, algo='defaut'):
    # Displaying all images :
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('Angle %g°' % (ind * 0.9))
        ax[int(i / rows), int(i % rows)
           ].imshow(stack[ind, :, :], cmap=plt.cm.Greys_r)
        ax[int(i / rows), int(i % rows)].axis('off')
    #plt.suptitle('%s images' % algo, fontsize=20)
    plt.tight_layout()
    plt.savefig('%s_360_angles.png' % algo, dpi=300)
    plt.close()


def sample_stack_proj_Angles(obj, angles, rows=5, cols=5, start_with=0, show_every=16, algo='defaut'):
    sim = tomopy.project(obj, angles)  # Calculate projections
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('Angle %g°' % (ind * 0.9))
        ax[int(i / rows), int(i % rows)
           ].imshow(sim[ind, :, :], cmap=plt.cm.Greys_r)
        ax[int(i / rows), int(i % rows)].axis('off')
    #plt.suptitle('%s images' % algo, fontsize=20)
    plt.tight_layout()
    plt.savefig('%s_proj_360_angles.png' % algo, dpi=300)
    plt.close()

def sample_stack_Z(stack, rows=4, cols=4, start_with=0, show_every=16, algo='default'):
    # Slice display from https://www.raddq.com/dicom-processing-segmentation-visualization-in-python/
    fig, ax = plt.subplots(rows, cols, figsize=[12, 12])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)
           ].imshow(stack[ind, :, :], cmap=plt.cm.Greys_r)
        ax[int(i / rows), int(i % rows)].axis('off')
    #plt.suptitle('Reconstructed Z slice', fontsize=20)
    plt.tight_layout()
    plt.savefig('%s_Z_stack_360_.png' % algo, dpi=300)
    plt.close()

# - - - - - - - - - - - - - - VisPy 3D visualization - - - - - - - - - - - - - - - -
# create colormaps that work well for translucent and additive volume rendering


class TransFire(BaseColormap):
    glsl_map = """
    vec4 translucent_fire(float t) {
        return vec4(pow(t, 0.5), t, t*t, max(0, t*1.05 - 0.05));
    }
    """


class TransGrays(BaseColormap):
    glsl_map = """
    vec4 translucent_grays(float t) {
        return vec4(t, t, t, t*0.05);
    }
    """


def scene_building(recon_algos):
    # Prepare canvas
    canvas = scene.SceneCanvas(keys='interactive', size=(1024, 768), show=True)
    # canvas.measure_fps()

    # Set up a viewbox to display the image with interactive pan/zoom
    view = canvas.central_widget.add_view()

    # Set whether we are emulating a 3D texture
    emulate_texture = False

    # Create the volume visuals for the different reconstructions
    volume1 = scene.visuals.Volume(recon_algos[0], parent=view.scene, threshold=0.225,
                                   emulate_texture=emulate_texture)
    volume2 = scene.visuals.Volume(recon_algos[1], parent=view.scene, threshold=0.225,
                                   emulate_texture=emulate_texture)
    volume3 = scene.visuals.Volume(recon_algos[2], parent=view.scene, threshold=0.225,
                                   emulate_texture=emulate_texture)
    volume4 = scene.visuals.Volume(recon_algos[3], parent=view.scene, threshold=0.225,
                                   emulate_texture=emulate_texture)
    volume5 = scene.visuals.Volume(recon_algos[4], parent=view.scene, threshold=0.225,
                                   emulate_texture=emulate_texture)
    #volume1.transform = scene.STTransform(translate=(64, 64, 0))
    
    # Hacky cyclic volume display setup:
    volume1.visible = True  # set first volume as visible, then switch with 3
    volume2.visible = False
    volume3.visible = False
    volume4.visible = False
    volume5.visible = False
    
    t1 = Text('ART reconstruction', parent=canvas.scene, color='white')
    t1.font_size = 18
    t1.pos = canvas.size[0] // 2, canvas.size[1] - 10
    t2 = Text('fbp reconstruction', parent=canvas.scene, color='white')
    t2.font_size = 18
    t2.pos = canvas.size[0] // 2, canvas.size[1] - 10
    t3 = Text('sirt reconstruction', parent=canvas.scene, color='white')
    t3.font_size = 18
    t3.pos = canvas.size[0] // 2, canvas.size[1] - 10
    t4 = Text('ospml_quad reconstruction', parent=canvas.scene, color='white')
    t4.font_size = 18
    t4.pos = canvas.size[0] // 2, canvas.size[1] - 10
    t5 = Text('pml_quad reconstruction', parent=canvas.scene, color='white')
    t5.font_size = 18
    t5.pos = canvas.size[0] // 2, canvas.size[1] - 10
    t1.visible = True
    t2.visible = False
    t3.visible = False
    t4.visible = False
    t5.visible = False
    
    # Implement axis connection with cam2
    @canvas.events.mouse_move.connect
    def on_mouse_move(event):
        if event.button == 1 and event.is_dragging:
            axis.transform.reset()

            axis.transform.rotate(cam2.roll, (0, 0, 1))
            axis.transform.rotate(cam2.elevation, (1, 0, 0))
            axis.transform.rotate(cam2.azimuth, (0, 1, 0))

            axis.transform.scale((50, 50, 0.001))
            axis.transform.translate((50., 50.))
            axis.update()

    # Implement key presses
    @canvas.events.key_press.connect
    def on_key_press(event):
        global opaque_cmap, translucent_cmap
        if event.text == '1':
            cam_toggle = {cam1: cam2, cam2: cam3, cam3: cam1}
            view.camera = cam_toggle.get(view.camera, cam2)
            print(view.camera.name + ' camera')
            if view.camera is cam2:
                axis.visible = True
            else:
                axis.visible = False
        elif event.text == '2':
            methods = ['mip', 'translucent', 'iso', 'additive']
            method = methods[(methods.index(volume1.method) + 1) % 4]
            print("Volume render method: %s" % method)
            cmap = opaque_cmap if method in [
                'mip', 'iso'] else translucent_cmap
            volume1.method = method
            volume1.cmap = cmap
        elif event.text == '3':  # hacky toogle between different reconstructed volumes
            if(volume1.visible):
                volume1.visible = False
                volume2.visible = True
                t1.visible = False
                t2.visible = True
                # t1.update()
            elif(volume2.visible):
                volume2.visible = False
                volume3.visible = True
                t2.visible = False
                t3.visible = True
            elif(volume3.visible):
                volume3.visible = False
                volume4.visible = True
                t3.visible = False
                t4.visible = True
            elif(volume4.visible):
                volume4.visible = False
                volume5.visible = True
                t4.visible = False
                t5.visible = True
            else:
                volume5.visible = False
                volume1.visible = True
                t5.visible = False
                t1.visible = True
        elif event.text == '4':
            if volume1.method in ['mip', 'iso']:
                cmap = opaque_cmap = next(opaque_cmaps)
            else:
                cmap = translucent_cmap = next(translucent_cmaps)
            volume1.cmap = volume2.cmap = volume3.cmap = volume4.cmap volume5.cmap = cmap
        elif event.text == '0':
            cam1.set_range()
            cam3.set_range()
        elif event.text != '' and event.text in '[]':
            s = -0.025 if event.text == '[' else 0.025
            volume1.threshold += s
            #volume2.threshold += s
            th = volume1.threshold if volume1.visible else volume2.threshold
            print("Isosurface threshold: %0.3f" % th)

    # for testing performance
    #@canvas.connect
    # def on_draw(ev):
    #    canvas.update()

    # @canvas.connect
    # def on_timer(ev):
    #         # Animation speed based on global time.
    #         t = event.elapsed
    #         c = Color(self.color).rgb
    #         # Simple sinusoid wave animation.
    #         s = abs(0.5 + 0.5 * math.sin(t))
    #         self.context.set_clear_color((c[0] * s, c[1] * s, c[2] * s, 1))
    #         self.update()

    # Create three cameras (Fly, Turntable and Arcball)
    fov = 60.
    cam1 = scene.cameras.FlyCamera(parent=view.scene, fov=fov, name='Fly')
    cam2 = scene.cameras.TurntableCamera(parent=view.scene, fov=fov,
                                         name='Turntable')
    cam3 = scene.cameras.ArcballCamera(
        parent=view.scene, fov=fov, name='Arcball')
    view.camera = cam2  # Select turntable at first

    # Create an XYZAxis visual
    axis = scene.visuals.XYZAxis(parent=view)  # view.scene tout court

    s = STTransform(translate=(50, 50), scale=(50, 50, 50, 1))
    affine = s.as_affine()
    axis.transform = affine

    # Setup colormap iterators
    opaque_cmaps = cycle(get_colormaps())
    translucent_cmaps = cycle([TransFire(), TransGrays()])
    opaque_cmap = next(opaque_cmaps)
    translucent_cmap = next(translucent_cmaps)

    # TODO: add a colorbar, fix error: AttributeError: module 'vispy.scene' has no attribute 'ColorBarWidget'
    #grid = canvas.central_widget.add_grid(margin=10)
    # cbar_widget = scene.ColorBarWidget(cmap=translucent_cmaps, orientation="right") #cmap="cool"
    # grid.add_widget(cbar_widget)
    # cbar_widget.pos = (800, 600)#(300, 100)
    # cbar_widget.border_color = "#212121"
    # grid.bgcolor = "#ffffff"

    # Create a rotation transform:
    # tr = MatrixTransform() #TODO: wait 0.5.0 to re-add MatrixTransform

    # Let's record a .gif:
    gif_file = Path("reconstruct_animation.mp4")
    if gif_file.is_file():
        print('reconstruct_animation.mp4 is already in the folder, please delete it to get a new one')
    else:
        print('Let\'s record a .mp4 of the reconstructed volume:')
        n_steps = 450  # 360
        step_angle = 0.8  # 1.
        # [] #0.1 fail, 0, 1 fail, (0.5, 0.5, 0.5)ok
        rotation_axis = np.array([0, 0, 1])
        # rt=scene.AffineTransform()
        #volume1.transform = rt
        #volume1.transform.rotate(angle=step_angle, axis=rotation_axis)
        #volume1.transform.translate([1, 1, 0])
        axis.visible = False
        #view.camera.set_range(x=[-3, 3])
        writer = imageio.get_writer(
            'reconstruct_animation.mp4', fps=50, quality=8)
        # TODO: add a progress bar
        gif_bar = progressbar.ProgressBar(
            redirect_stdout=True, max_value=n_steps)
        gif_bar.start()
        for i in range(n_steps):  # * 2):
            im = canvas.render()
            writer.append_data(im)
            view.camera.transform.translate([1.8, -1.8, 0])
            view.camera.transform.rotate(step_angle, rotation_axis)
            #volume1.transform.rotate(angle=step_angle, axis=rotation_axis)
            gif_bar.update(i)
        gif_bar.finish()
        writer.close()
        axis.visible = True

# from https://stackoverflow.com/questions/5376837/how-can-i-do-an-if-run-from-ipython-test-in-python


def run_from_ipython():
    try:
        __IPYTHON__
        print('Running inside IPython: Let\'s use VisPy Interactive mode')
        return True
    except NameError:
        return False


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - IMPERATIVE PROGRAM - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
if __name__ == '__main__':
    # Setting tomographic images size :
    img_size = 256  # 256

    # First we need to create or load the tomo matrix :
    tomo_file = Path("TOMO_360_angle_images.pkl")
    if tomo_file.is_file():
        print('TOMO_360_angle_images.pkl is in the folder, let\'s load it:')
    else:
        get_images()

    # Loading dataset
    # , protocol=2)#protocol=2 for python2, 3 by defaut with python3
    mat_360 = joblib.load('TOMO_360_angle_images.pkl')
    print("TOMO_360_angle_images.pkl loaded !")

    # Let's compute the inverse radon transform, reconstruct from all images
    # from http://scikit-image.org/docs/dev/auto_examples/transform/plot_radon_transform.html

    # Set data collection angles as equally spaced between 0-180 degrees.
    # 0.9deg rotation between images
    # for 180deg rotation, only taking images 1 to 201:
    # total_rotation = 180 # 200 images *0.9deg, (full image set = 255 images * 0.9deg)
    total_rotation = 360  # for 360deg imaging
    nb_angles = int(total_rotation / 0.9)
    # *2 because tomopy.angles() compute angles for a full 180° rotation, not 360°
    theta = tomopy.angles(nb_angles) * 2

    # Flat-field correction of raw data.
    # proj = tomopy.normalize(proj, flat, dark) #NEED FLAT AND DARK VALUES
    # norm_mat_180 = tomopy.normalize_bg(mat_180, theta) # Normalization from background TODO: FIX THIS PYTHON REBOOT!
    norm_mat_360 = mat_360 / mat_360.mean()  # simplest normalization

    # Plotting raw measurements by angles:
    raw_file = Path("Raw_360_angles.png")
    if raw_file.is_file():
        print('Raw_360_angles.png already in folder, please delete to get a new-one')
    else:
        sample_stack_Angles(mat_360, algo='Raw')

    # Find rotation center #FAILURE - TODO: FIX THIS PYTHON REBOOT!
    # rot_center = tomopy.find_center(sim, theta, init=125,# mask=True, #initial gess:256/2
    #                                ind=0, tol=0.5)

    #print("Center of rotation: ", rot_center)
    # sim_log = tomopy.minus_log(mat_180) # TODO: Why use this ? Getting sim_log.max=inf...

    # Simple reconstruct with ART algorithm:
    # recon = tomopy.recon(mat_180, theta, algorithm='art')#, center=rot_center,)
    # algorithm='bart'BOF, 'fbp'SUPERBOF, 'mlem'BOF, 'osem'BOF, 'ospml_hybrid'MOUAIS, 'ospml_quad'BOF, 'pml_hybrid'BOF, 'pml_quad'MOUAIS, 'sirt'MOUAIS
    #filter_name = 'shepp', 'cosine', 'hann', 'hamming', 'ramlak', 'parzen', 'butterworth'

    # recon is (z, x, y)
    # masked_recon = tomopy.circ_mask(recon, axis=0, ratio=0.9)
    # plt.imshow(masked_recon[:,120,:], cmap=plt.cm.Greys_r)
    # plt.show()

    # Reconstructing with all avaible algorithms :
    recon_file = Path("360_recon_all_algo_noCentering.pkl")
    if recon_file.is_file():
        print('360_recon_all_algo_noCentering.pkl is in the folder, let\'s load it')
        recon_algos = joblib.load('360_recon_all_algo_noCentering.pkl')
        recon = recon_algos[0]  # select ART algorithm
    else:
        # We only select algortihms which held to differences in the reconstruction:
        # ,'ospml_hybrid', 'pml_hybrid', 'bart', 'mlem', 'osem'}
        algorithms = {'art', 'fbp', 'sirt', 'ospml_quad', 'pml_quad'}
        print('Let\'s reconstruct with: ' + format(algorithms) + ' algorithms')
        # Display a progressbar
        bar = progressbar.ProgressBar(redirect_stdout=True, term_width=2, max_value=len(
            algorithms) * 15)  # widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()]
        bar.start()
        start = time.time()
        recon_algos = []
        i = 0
        for algo in algorithms:
            # , center=rot_center,)
            recon = tomopy.recon(mat_360, theta, algorithm=algo)
            i = i + 10
            bar.update(i)
            # Plotting reconstructed projections by angles
            sample_stack_proj_Angles(recon, theta, algo=algo)
            i = i + 4
            bar.update(i)
            # Plotting reconstructed Z slices by Y height
            sample_stack_Z(recon, algo=algo)
            recon_algos.append(recon)
            i = i + 1
            bar.update(i)
        bar.finish()
        joblib.dump(recon_algos, '360_recon_all_algo_noCentering.pkl')  # 20 Mo
        end = time.time()
        print("Done in " + str(end - start) + "ms")
        recon = recon_algos[0]  # select ART algorithm

    # 3D rotational animation :
    print('Let\'s display the reconstructed 3D volume')
    # Using VisPy as I can't compile VTK 8.0.0 : http://vispy.org/plot.html
    # print(vispy.sys_info())
    # https://github.com/vispy/vispy/blob/master/examples/basics/scene/volume.py
    """
    Example volume rendering
    Controls:
    * ESC  - to quit
    * 1  - toggle camera between first person (fly), regular 3D (turntable) and
           arcball
    * 2  - toggle between volume rendering methods
    * 3  - toggle between stent-CT / brain-MRI image
    * 4  - toggle between colormaps
    * 0  - reset cameras
    * [] - decrease/increase isosurface threshold
    With fly camera:
    * WASD or arrow keys - move around
    * SPACE - brake
    * FC - move up-down
    * IJKL or mouse - look around
    """
    # if run_from_ipython():
    #     #app.use_app('glfw')  # for testing specific backends
    #     app.set_interactive() #to use in IPython
    #     scene_building()
    #     # All variables listed in this scope are accessible via the console.
    #     #app.Timer(interval=0.0, connect=None, iterations=-1, start=True)#, app=None)
    #     #app.Timer('auto', connect=on_timer(), start=True) #TODO
    #     # In IPython, try typing any of the following:
    #     #   canvas.color = (1.0, 0.0, 0.0)
    #     canvas.color = 'white'#'red'
    #     canvas.show()
    # else:
    print(__doc__)
    # 'ql+' If PyOpenGL is not avaible please use gl='gl2'
    vispy.use(app='glfw', gl='gl2')
    # app='PyQt5'(framebuffer quarter on HiDPi), 'pyglet'(slow, bug), 'glfw',
    scene_building(recon_algos)
    app.run()

# # VisPy point by point = SLOW AS HELL - - - - - - - - - - - - - - - - - - - -
# #from https://github.com/vispy/vispy/issues/1189
# # build your visuals
# Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
#
# # The real-things : plot using scene
# # build canvas
# canvas = scene.SceneCanvas(keys='interactive', show=True)
#
# # Add a ViewBox to let the user zoom/rotate
# view = canvas.central_widget.add_view()
# view.camera = 'turntable'
# view.camera.fov = 45
# view.camera.distance = 300 #500
#
# n = recon.size #near 16 millions points: 256*256*256=16777216
# pos = np.zeros((n, 3))
# for i in range(0,recon.shape[0]):
#     for j in range(0,recon.shape[1]):
#         for k in range(0,recon.shape[2]):
#             pos[i+j+k] = (i,j,k)
#
# #xyz = []
# #xyz = np.mgrid[0:recon.shape[0], 0:recon.shape[1], 0:recon.shape[2]]
# #xyz.append(np.arange(0,215,1))
#
# colors = np.ones((n, 4), dtype=np.float32)
# norm = plt.Normalize()
# colors = plt.cm.jet(norm(recon.reshape(-1)))
#
# # plot
# p1 = Scatter3D(parent=view.scene)
# p1.set_gl_state('translucent', blend=True, depth_test=True)
# p1.set_data(pos, face_color=colors)
# p1.symbol = visuals.marker_types[10]
#
# # run
# app.run()


# 3D visualization using matplotlib... slow ! - - - - - - - - - - - - - - - - -
# import itertools
# from mpl_toolkits.mplot3d import axes3d
# from matplotlib import animation
#
# N = 10 # nb of images
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# #ax = plt.figure().gca(projection='3d')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# #pcm = ax.pcolormesh(x, y, Z, vmin=-1., vmax=1., cmap='RdBu_r')
#
# # load some test data for demonstration and plot a wireframe
# #X, Y, Z = axes3d.get_test_data(0.1)
# #ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)
#
# recon[:,:,-1].shape
#
# xyz = recon.reshape(-1)
# xyz = plt.cm.Greys_r(xyz)
#
# x = np.linspace(0,256, 256)
# y = np.linspace(0,256, 256)
# z = np.linspace(0,256, 256)
# points = []
# for element in itertools.product(x, y, z):
#     points.append(element)
# fxyz = map(xyz, points)
# xi, yi, zi = zip(*points)
#
# ax.scatter(xi, yi, zi, c=fxyz, alpha=0.5) #,cmap=plt.cm.Greys_r
# fig.tight_layout()
# plt.show()
#
# # rotate the axes and update
# #for angle in range(0, 360):
# #    ax.view_init(30, angle)
# #    plt.draw()
# #    plt.pause(.001)
#
# def update(num):
#     ax.view_init(30, num*(360/N))
#
# ani = animation.FuncAnimation(fig, update, N, interval=10000/N, blit=False)
# ani.save('matplot002.gif', writer='imagemagick')
#
#
#
#
# import mpl_toolkits.mplot3d.axes3d as p3
# from matplotlib import animation
#
# fig = plt.figure()
# ax = p3.Axes3D(fig)
#
# def gen(n):
#     phi = 0
#     nb_rotation = 1
#     while phi < nb_rotation*np.pi:
#         yield np.array([np.cos(phi), np.sin(phi), phi])
#         phi += nb_rotation*np.pi/n
#
# def update(num, data, line):
#     line.set_data(data[:2, :num])
#     line.set_3d_properties(data[2, :num])
#
# N = 100
# data = np.array(list(gen(N))).T
# line, = ax.plot(data[0, 0:1], data[1, 0:1], data[2, 0:1])
#
# # Setting the axes properties
# ax.set_xlim3d([-1.0, 1.0])
# ax.set_xlabel('X')
#
# ax.set_ylim3d([-1.0, 1.0])
# ax.set_ylabel('Y')
#
# ax.set_zlim3d([0.0, 10.0])
# ax.set_zlabel('Z')
#
# ani = animation.FuncAnimation(fig, update, N, fargs=(data, line), interval=10000/N, blit=False)
# #ani.save('matplot003.gif', writer='imagemagick')
# plt.show()
#
#
# # - - - - - -
# # Visualization :
#
# from skimage import measure
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# def make_mesh(image, threshold=-300, step_size=1):
#     print("Transposing surface")
#     #p = image.transpose(2,1,0) #TODO: don't transpose as we have (x, y, z) images
#     p = image.transpose(0,2,1) # simply inverse X and Y
#
#     print("Calculating surface")
#     verts, faces, norm, val = measure.marching_cubes(p, threshold, step_size=step_size, allow_degenerate=True)
#     return verts, faces
#
# def plotly_3d(verts, faces):
#     x,y,z = zip(*verts)
#     print("Drawing")
#     # Make the colormap single color since the axes are positional not intensity.
#     #colormap=['rgb(255,105,180)','rgb(255,255,51)','rgb(0,191,255)']
#     colormap=['rgb(236, 236, 212)','rgb(236, 236, 212)']
#
#     fig = FF.create_trisurf(x=x,
#                         y=y,
#                         z=z,
#                         plot_edges=False,
#                         colormap=colormap,
#                         simplices=faces,
#                         backgroundcolor='rgb(64, 64, 64)',
#                         title="Interactive Visualization")
#     iplot(fig)
#
# def plt_3d(verts, faces):
#     print("Drawing")
#     x,y,z = zip(*verts)
#     fig = plt.figure(figsize=(10, 10))
#     ax = fig.add_subplot(111, projection='3d')
#
#     # Fancy indexing: `verts[faces]` to generate a collection of triangles
#     mesh = Poly3DCollection(verts[faces], linewidths=0.05, alpha=1)
#     face_color = [1, 1, 0.9]
#     mesh.set_facecolor(face_color)
#     ax.add_collection3d(mesh)
#
#     ax.set_xlim(0, max(x))
#     ax.set_ylim(0, max(y))
#     ax.set_zlim(0, max(z))
#     ax.set_axis_bgcolor((0.7, 0.7, 0.7))
#     plt.show()
# v, f = make_mesh(recon, 0)#
# plt_3d(v, f)
#
#
#
#
#
# #Animation from : http://jakevdp.github.io/blog/2012/08/18/matplotlib-animation-tutorial/
# from matplotlib import animation
# # First set up the figure, the axis, and the plot element we want to animate
# fig = plt.figure()
# ax = plt.axes(xlim=(0, 255), ylim=(0, 255))
# line, = ax.plot([], [], lw=2)
#
# # initialization function: plot the background of each frame
# def init():
#     line.set_data([], [])
#     return line,
#
# # animation function.  This is called sequentially
# def animate(i):
#     x = np.linspace(0, 2, 1000)
#     y = np.sin(2 * np.pi * (x - 0.01 * i))
#     line.set_data(x, y)
#     return line,
#
# # call the animator.  blit=True means only re-draw the parts that have changed.
# anim = animation.FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)
#
# # save the animation as an mp4.  This requires ffmpeg or mencoder to be
# # installed.  The extra_args ensure that the x264 codec is used, so that
# # the video can be embedded in html5.  You may need to adjust this for
# # your system: for more information, see
# # http://matplotlib.sourceforge.net/api/animation_api.html
# anim.save('basic_animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])
# plt.show()
