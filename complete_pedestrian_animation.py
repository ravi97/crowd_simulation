import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import pandas as pd
import numpy as np
import random
from complete_pedestrian_simulation import loops,t_int,walls,n_ped,v_desired_max

### GLOBAL VARIABLES ###

x_dim=max(walls["X1"].max(),walls["X2"].max())  #x_dim is the maximum value of x dimension of the walls
y_dim=max(walls["Y1"].max(),walls["Y2"].max())  #y_dim is the maximum value of y dimension of the walls

grid_size=1  #size of grid in the heat map (in meters)
mf=1/float(grid_size) #multiplication factor since the number of grids in the heat map increases with decrease in grip size

"""The weightage of various factors on the Crowd risk Index (CRI)"""
imp_weight=2 #impatience
density_weight=1 #pedestrian density
b_weight=0 #scalar sum of border force
p_weight=0 #scalar sum of pedestrian repulsion force

"""The threshold/maximum values of various factors influencing CRI, beyond which, it is dangerous"""
imp_threshold=1     
density_threshold=3
b_threshold=0
p_threshold=0

max_cri=imp_threshold*imp_weight+density_threshold*density_weight \
        + b_threshold*b_weight + p_threshold*p_weight        #The maximum value the CRI can have without danger


x=pd.ExcelFile("Pedestrian_details.xlsx").parse("X positions").as_matrix() #read x values from the excel file as a numpy array
y=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y positions").as_matrix() #read y values from the excel file as a numpy array
vx=pd.ExcelFile("Pedestrian_details.xlsx").parse("X velocity").as_matrix() #read velocity_x values from the excel file as a numpy array
vy=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y velocity").as_matrix() #read velocity_y values from the excel file as a numpy array
ax=pd.ExcelFile("Pedestrian_details.xlsx").parse("X acceleration").as_matrix() #read acceleration_x values from the excel file as a numpy array
ay=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y acceleration").as_matrix() #read acceleration_y values from the excel file as a numpy array
b_net=pd.ExcelFile("Pedestrian_details.xlsx").parse("border force").as_matrix() #read net border force values from the excel file as a numpy array
p_net=pd.ExcelFile("Pedestrian_details.xlsx").parse("pedestrian force").as_matrix() #read net pedestrian force values from the excel file as a numpy array

### FUNCTIONS USED ###


def update_scatter(i,ax1,scat):
	'''
	This method is used to update the scatter plot after every frame
	''' 
	ped_xy=[] #list containing the coordinates (x,y) of every pedestrian
	for j in xrange(n_ped):
		if vx[i][j]!=0 and vy[i][j]!=0: #if the velocity is zero, the pedestrian has not yet emerged
			ped_xy.append([x[i][j],y[i][j]]) 

	scat.set_offsets(ped_xy) #this method sets the data for the scatter plot
	return scat,


def update_heat(i,ax2,im):
	
	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	density=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #initial value of density
	impatience=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #initial value of impatience
	border_forces=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #initial value of border force
	ped_forces=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #initial value of pedestrian repulsion force
	
	for j in xrange(n_ped):
		if vx[i][j]!=0 and vy[i][j]!=0:  #if the velocity is zero, the pedestrian has not yet emerged
			density[int(math.floor(y[i][j]))][int(math.floor(x[i][j]))]+=1 #density value increments in the cells where the pedestrian is present 
			
			imp=1-(math.sqrt(vx[i][j]**2 + vy[i][j]**2))/v_desired_max
			impatience[int(math.floor(y[i][j]*mf))][int(math.floor(x[i][j]*mf))]+=imp #adds the value of impatience to the cell where pedestrian is present

			border_forces[int(math.floor(y[i][j]))][int(math.floor(x[i][j]))]+=b_net[i][j] #adds the value of border force to the cell where pedestrian is present
			ped_forces[int(math.floor(y[i][j]))][int(math.floor(x[i][j]))]+=p_net[i][j] #adds the value of pedestrian force to the cell where pedestrian is present


	cri=density_weight*density + imp_weight*impatience + b_weight*border_forces + p_weight * ped_forces #calculation of CRI

	im.set_data(-cri) #sets CRI to the heat map
	return im,



if __name__ == '__main__':
	

	### SETUP SCATTER PLOT ###

	fig=plt.figure(figsize=(10,5)) #create graph
	plt.suptitle("Animation graphs") #set title for graph

	ax1=fig.add_subplot(1,2,1) #create 1 x 2 subplot in the graph and assigns the first subplot to ax1
	ax1.set_title("Pedestrian path") #set title for the subplot
	ax1.set_xlim(0,x_dim)  #set x limits to first subplot
	ax1.set_ylim(0,y_dim)  #set y limits to first subplot
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax1.plot([row[1],row[3]],[row[2],row[4]]) #this will plot a line using the x and y coordinates of the wall entered in the excel
	scat=ax1.scatter([],[]) #initial value of scatterplot


	### SETUP HEAT MAP ###

	ax2=fig.add_subplot(1,2,2) #create 1 x 2 subplot in the graph and assigns the second subplot to ax2
	ax2.set_title("Heat map") #set title for the subplot
	ax2.set_xlim(0,x_dim*mf)  #set x limits to second subplot
	ax2.set_ylim(0,y_dim*mf)  #set y limits to second subplot
	ax2.get_xaxis().set_visible(False) #removes the scale display of the heat map
	ax2.get_yaxis().set_visible(False) #removes the scale display of the heat map
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax2.plot([row[1],row[3]],[row[2],row[4]]) #this will plot a line using the x and y coordinates of the wall entered in the excel
	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #initial value of CRI
	im=ax2.imshow(cri,cmap='hot',interpolation='nearest',vmin=-(max_cri),vmax=0) #imageplot used to plot the heatmap
	#we use negative of CRI since when cmap is set to hot, the heatmap takes white for maximum and black for minimum


	### RUN ANIMATION ###
	'''
	Arguements for the below functions:
	fig - figure of theplot we want to animate
	update function -  function that updates the plot after every frame
	frame- number of frames for which you need to animation to run
	interval - interval time between two frames
	'''
	anim=animation.FuncAnimation(fig,update_scatter,fargs=(ax1,scat),frames=loops,interval=int(t_int*1000)) #animates the scatter plot
	anim2=animation.FuncAnimation(fig,update_heat,fargs=(ax2,im),frames=loops,interval=int(t_int*1000)) #animates the heat map
	plt.show()
