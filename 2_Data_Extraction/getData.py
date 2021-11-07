
import vessel_analysis_unmod
import nibabel as nib
import networkx as nx
from scipy.io import savemat
import numpy as np
from scipy.sparse import csr_matrix
import os
import glob
import numpy
from unet_core.io import ImageReader, load_image
from unet_core.vessel_analysis_unmod import VesselTree



def getVesselData(img_path, img_name, seg_threshold):

	img_str = nib.load(img_path)
	img_vol = img_str.dataobj[:]
	seg = img_vol > seg_threshold

	vt = vessel_analysis_unmod.VesselTree(input_image=seg)
	ves = vt.analyse_vessels()

	filename = img_name + 'AdjMatrix.txt'
	i = 0;

	for c in ves.components:
		j = 1
		nx_graph = c.graph # this is a NetworkX graph (https://networkx.github.io/documentation/networkx-1.9/index.html)

		adj = nx.to_scipy_sparse_matrix(nx_graph)
		print(adj)

		i = i + 1

		#filename2 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/AdjMatrix' + str(i) + '.mat'
		filename2 = '/home/stolz/Desktop/Vessel Networks/unet-test-master/Data Bostjan Output/' +img_name + '/AdjMatrix' + str(i) + '.mat'

		b = csr_matrix(adj, dtype='double')
		savemat(filename2, {'b': b}, format='5')

		for n1, n2, b in c.branch_iter():
			p = b.get_smoothed_points()
			print(p)
			j = j + 1
			#filename3 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
			filename3 ='/home/stolz/Desktop/Vessel Networks/unet-test-master/Data Bostjan Output/'+ img_name + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
			numpy.savetxt(filename3, p, fmt='%.18e', delimiter=' ', newline='\n')

		#scipy.io.savemat(filename2, adj)
	   # scipy.sparse.save_npz(filename2, adj, compressed=True)
	   # print(nx.to_dict_of_dicts(adj))
	   # f.write(adj)
		for u, v, b in c.branch_iter():
			pass # iterate over each branch in the skeleton component, branch `b` connects nodes `u` and `v`.

	#vt.plot_vessels(write_location='/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' + img_name + 'vt_viz_example.png', metric='clr')

	vt.plot_vessels(write_location= '/home/stolz/Desktop/Vessel Networks/unet-test-master/Data Bostjan Output/' + img_name + 'vt_viz_example.png', metric='clr')




def getSkeletonVesselData(skeleton_path, image_path, img_name):

	min_branch_length = 10
	min_object_size = 100
	image = load_image(image_path)
	image_dims = image.shape

	v2 = vessel_analysis_unmod.VesselTree(image_dimensions=image_dims)
	v2.load_skeleton(skeleton_path)

	output_dir = '/home/stolz/Desktop/Vessel Networks/unet-test-master/Data Bostjan Output/Skeleton Output/' + img_name

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	v2.plot_vessels(metric='diameter', metric_scaling=2, write_location=output_dir + "diameterPlot.png", width_scaling=0.2)
	v2.plot_vessels(metric='length', metric_scaling=0.5, write_location=output_dir + "lengthPlot.png", width_scaling=0.2)
	v2.plot_vessels(metric='clr', metric_scaling=255, write_location=output_dir + "clrPlot.png", width_scaling=0.2)

	j = 0

	for n1, n2, b in v2.branch_iter():
		p = np.array(b.get_smoothed_points())
		#p2 = np.array([[n1.x, n1.y, n1.z]])
		#p3 = np.array([[n2.x, n2.y, n2.z]])
		#print(p2)
		#print(p3)
		#nodes1 = np.concatenate((p2,p))
		#nodes2 = np.concatenate((nodes1,p3))
		j = j + 1
		#filename3 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
		filename3 = output_dir + 'Branch' + str(j) + '.txt'
		numpy.savetxt(filename3, p, fmt='%.18e', delimiter=' ', newline='\n')


def getSkeletonVesselDataComponents(skeleton_path, image_path, img_name):

	min_branch_length = 10
	min_object_size = 100
	image = load_image(image_path)
	image_dims = image.shape

	v2 = vessel_analysis_unmod.VesselTree(image_dimensions=image_dims)
	v2.load_skeleton(skeleton_path)

	output_dir = '/home/stolz/Desktop/Vessel Networks/unet-test-master/Data Bostjan Output/Skeleton Output/' + img_name

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	#v2.plot_vessels(metric='diameter', metric_scaling=2, write_location=output_dir + "diameterPlot.png", width_scaling=0.2)
	#v2.plot_vessels(metric='length', metric_scaling=0.5, write_location=output_dir + "lengthPlot.png", width_scaling=0.2)
	#v2.plot_vessels(metric='clr', metric_scaling=255, write_location=output_dir + "clrPlot.png", width_scaling=0.2)

	#print(v2.skeleton.num_components)

	i = 0;

	for c in v2.skeleton.components:
		nx_graph = c.graph
		adj = nx.adjacency_matrix(nx_graph)
		print(adj)

		j = 0;
		i = i + 1

		filename2 = output_dir + '/AdjMatrix' + str(i) + '.mat'
		b = csr_matrix(adj, dtype='double')
		savemat(filename2, {'b': b}, format='5')

		for n1, n2, b in c.branch_iter():
			p = np.array(b.get_smoothed_points())
			print(p)
		#p2 = np.array([[n1.x, n1.y, n1.z]])
		#p3 = np.array([[n2.x, n2.y, n2.z]])
		#print(p2)
		#print(p3)
		#nodes1 = np.concatenate((p2,p))
		#nodes2 = np.concatenate((nodes1,p3))
			j = j + 1
		#filename3 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
			filename3 = output_dir + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
			numpy.savetxt(filename3, p, fmt='%.18e', delimiter=' ', newline='\n')

def getSkeletonVesselData_DiametersClrAndLenghts(skeleton_path, image_path, img_name):

	min_branch_length = 10
	min_object_size = 100
	image = load_image(image_path)
	image_dims = image.shape

	v2 = vessel_analysis_unmod.VesselTree(image_dimensions=image_dims)
	v2.load_skeleton(skeleton_path)

	output_dir = '/mi/share/scratch/Stolz/Summer17_Dataset_RT_AntiAngio/' + img_name

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	#v2.plot_vessels(metric='diameter', metric_scaling=2, write_location=output_dir + "diameterPlot.png", width_scaling=0.2)
	#v2.plot_vessels(metric='length', metric_scaling=0.5, write_location=output_dir + "lengthPlot.png", width_scaling=0.2)
	#v2.plot_vessels(metric='clr', metric_scaling=255, write_location=output_dir + "clrPlot.png", width_scaling=0.2)

	#print(v2.skeleton.num_components)

	i = 0;

	for c in v2.skeleton.components:
		nx_graph = c.graph
		adj = nx.adjacency_matrix(nx_graph)
		print(adj)

		j = 0;
		i = i + 1

		#filename2 = output_dir + '/AdjMatrix' + str(i) + '.mat'
		#b = csr_matrix(adj, dtype='double')
		#savemat(filename2, {'b': b}, format='5')

		#for p in b.points:
	       # print('({}, {}, {}), diameter: {}'.format(p.x, p.y, p.z, p.diameter))

		for n1, n2, b in c.branch_iter():
			#p = np.array(b.get_smoothed_points())
			print(b.diameter)
			print(b.length)
			print(b.soam)
			print(b.clr)
		#p2 = np.array([[n1.x, n1.y, n1.z]])
		#p3 = np.array([[n2.x, n2.y, n2.z]])
		#print(p2)
		#print(p3)
		#nodes1 = np.concatenate((p2,p))
		#nodes2 = np.concatenate((nodes1,p3))
			j = j + 1
		#filename3 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
			filename3 = output_dir + '/Diameter_Component' + str(i) + 'Branch' + str(j) + '.txt'
			filename4 = output_dir + '/Length_Component' + str(i) + 'Branch' + str(j) + '.txt'
			filename5 = output_dir + '/SOAM_Component' + str(i) + 'Branch' + str(j) + '.txt'
			filename6 = output_dir + '/Clr_Component' + str(i) + 'Branch' + str(j) + '.txt'


			f = open(filename3,'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
			f.write('{}'.format(b.diameter))
			f.close()

			f = open(filename4,'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
			f.write('{}'.format(b.length))
			f.close()

			f = open(filename5,'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
			f.write('{}'.format(b.soam))
			f.close()

			f = open(filename6,'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
			f.write('{}'.format(b.clr))
			f.close()


def getSkeletonVesselData_DiametersClrAndLenghtsOneFile(skeleton_path, image_path, img_name):

	min_branch_length = 10
	min_object_size = 100
	image = load_image(image_path)
	image_dims = image.shape

	v2 = vessel_analysis_unmod.VesselTree(image_dimensions=image_dims)
	v2.load_skeleton(skeleton_path)

	output_dir = '/home/stolz/Desktop/Vessel Networks/Network_for_Jakub/' + img_name

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	#v2.plot_vessels(metric='diameter', metric_scaling=2, write_location=output_dir + "diameterPlot.png", width_scaling=0.2)
	#v2.plot_vessels(metric='length', metric_scaling=0.5, write_location=output_dir + "lengthPlot.png", width_scaling=0.2)
	#v2.plot_vessels(metric='clr', metric_scaling=255, write_location=output_dir + "clrPlot.png", width_scaling=0.2)

	#print(v2.skeleton.num_components)

	i = 0;

	for c in v2.skeleton.components:
		nx_graph = c.graph

		j = 0;
		i = i + 1

		#filename2 = output_dir + '/AdjMatrix' + str(i) + '.mat'
		#b = csr_matrix(adj, dtype='double')
		#savemat(filename2, {'b': b}, format='5')

		#for p in b.points:
	       # print('({}, {}, {}), diameter: {}'.format(p.x, p.y, p.z, p.diameter))

		filenameComponent = output_dir + '/Summary_Component' + str(i) + '.txt'

		arr = []


		for n1, n2, b in c.branch_iter():

			j = j + 1

			arr = np.append(arr,np.array([n1.point.x, n1.point.y, n1.point.z, n2.point.x, n2.point.y, n2.point.z, b.diameter, b.length, b.soam, b.clr]),axis=0)
			#print([n1.point.x, n1.point.y, n1.point.z, n2.point.x, n2.point.y, n2.point.z, b.diameter, b.length, b.soam, b.clr])

		print(arr)

		f = open(filenameComponent,'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
		numpy.savetxt(f, arr, fmt='%.18e', delimiter= ',', newline= '\n')
		#f.write('{}'.format([n1.point, n2.point, b.diameter, b.length, b.soam, b.clr]),'\n')
		f.close()


def getVesselDataComponents(img_name):

	initial_path = '/scratch/stolz/Roche_Data/nii_files/'
	img_path = initial_path + img_name + '.nii'

	min_branch_length = 0 #10
	min_object_size = 0 #100

	output_dir = '/home/stolz/Desktop/Vessel Networks/Roche_Output/' +img_name+ '/'
	skeleton_path = output_dir + 'skeleton.pkl'

	if not os.path.exists(output_dir):
		os.makedirs(output_dir)


	image = load_image(img_path)
	image_dims = image.shape

	vt = VesselTree(image, min_branch_length=min_branch_length, min_object_size=min_object_size)
	v = vt.analyse_vessels(verbose=True)

	vt.save_skeleton(skeleton_path)

	i = 0;

	for c in v.components:
		j = 0
		nx_graph = c.graph # this is a NetworkX graph (https://networkx.github.io/documentation/networkx-1.9/index.html)

		adj = nx.to_scipy_sparse_matrix(nx_graph)
		print(adj)

		i = i + 1

		#filename2 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/AdjMatrix' + str(i) + '.mat'
		filename2 = output_dir + 'AdjMatrix' + str(i) + '.mat'

		b = csr_matrix(adj, dtype='double')
		savemat(filename2, {'b': b}, format='5')

		for n1, n2, b in c.branch_iter():
			p = b.get_smoothed_points()
			print(p)
			j = j + 1
			#filename3 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
			filename3 = output_dir + 'Component' + str(i) + 'Branch' + str(j) + '.txt'
			numpy.savetxt(filename3, p, fmt='%.18e', delimiter=' ', newline='\n')

	vt.plot_vessels(metric='diameter', metric_scaling=10, write_location=output_dir + "diameterPlot.png", width_scaling=0.4)
	vt.plot_vessels(metric='length', metric_scaling=1, write_location=output_dir + "lengthPlot.png", width_scaling=0.4)
	vt.plot_vessels(metric='clr', metric_scaling=255, write_location=output_dir + "clrPlot.png", width_scaling=0.4)


def getVesselDataComponentsDiametersLengths(img_name, folder, path_in, path_out,
                                            pix_dim=(20,20,20)):

    
    # print(img_name)

    initial_path = path_in
    # for file in os.listdir(path_in):
    #     if file.endswith(".nii"):
    #         img_name = file
    
    # Removed '.nii' extension as this is included in main script
    img_path = os.path.join(initial_path, img_name)
    
    min_branch_length = 0 #10
    min_object_size = 0 #100
    
    img_name = (os.path.splitext(img_name))[0]
    output_dir = os.path.join(folder,'Skeletons', img_name)
    output_dir = output_dir + '/'
    
    skeleton_path = output_dir + 'skeleton.pkl'
    
    os.makedirs(output_dir + 'Components/')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    
    image = load_image(img_path)
    image[image >= 127] = 255
    image[image < 127] = 0
    
    image_dims = image.shape
    
    vt = VesselTree(image, min_branch_length=min_branch_length, min_object_size=min_object_size,
                    pix_dim=pix_dim)
    v = vt.analyse_vessels(verbose=True)
    
    vt.save_skeleton(skeleton_path)
    
    i = 0;
    
    filenameNetwork = output_dir + 'Network_Summary.txt'
    ff = open(filenameNetwork,'w')

    maxdiam = 0.0
    maxlength = 0.0
    mindiam = 1e8
    minlength = 1e8
    for c in v.components:
        j = 0
        nx_graph = c.graph # this is a NetworkX graph (https://networkx.github.io/documentation/networkx-1.9/index.html)
        
        i = i + 1
        
        		#filename2 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/AdjMatrix' + str(i) + '.mat'
        
        arr = []
        
        filenameComponent = output_dir + 'Components/Summary_Component' + str(i) + '.txt'

        for n1, n2, b in c.branch_iter():
            p = b.get_smoothed_points()
            # print(p)
            j = j + 1
            #filename3 = '/Users/bernadettestolz/Dropbox/DPhil Projects/Vasculature Project/Codes from Russ/unet-test-master/' +img_name + '/Component' + str(i) + 'Branch' + str(j) + '.txt'
            filename1 = output_dir + 'Components/Component' + str(i) + 'Branch' + str(j) + '.txt'
            numpy.savetxt(filename1, p, fmt='%.18e', delimiter=' ', newline='\n')
            
            # print(b.diameter)
            # print(b.length)
            # print(b.soam)
            # print(b.clr)
            
            arr = np.append(arr,np.array([n1.point.x, n1.point.y, n1.point.z, n2.point.x, n2.point.y, n2.point.z, b.diameter, b.length, b.soam, b.clr]),axis=0)
            		#print([n1.point.x, n1.point.y, n1.point.z, n2.point.x, n2.point.y, n2.point.z, b.diameter, b.length, b.soam, b.clr])
            
            if not (b.diameter is None):
                ff.write('%.8e, %.8e\n' %(b.diameter, b.length))
                if b.diameter > maxdiam: maxdiam = b.diameter
                if b.length > maxlength: maxlength = b.length
                if b.diameter < mindiam: mindiam = b.diameter
                if b.length < minlength: minlength = b.length
            
        # print(arr)
        
        f = open(filenameComponent,'w')  # w : writing mode  /  r : reading mode  /  a  :  appending mode
        numpy.savetxt(f, arr, fmt='%.18e', delimiter= ',', newline= '\n')
        	#f.write('{}'.format([n1.point, n2.point, b.diameter, b.length, b.soam, b.clr]),'\n')
        f.close()

    ff.close()
    
    vt.plot_vessels(metric='diameter',  write_location=output_dir + "diameterPlot.png", width_scaling=0.2, threshold=None, maxval=maxdiam, minval=0.)
    vt.plot_vessels(metric='length', write_location=output_dir + "lengthPlot.png", width_scaling=0.2, threshold=None, maxval=maxlength, minval=0.)
    vt.plot_vessels(metric='clr', metric_scaling=255, write_location=output_dir + "clrPlot.png", width_scaling=0.2, threshold=None, maxval=255., minval=0.0)
