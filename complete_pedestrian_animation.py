import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import pandas as pd
import numpy as np
import random
from complete_pedestrian_simulation import loops,t_int,walls,n_ped,v_desired_max

### GLOBAL VARIABLES ###

x_dim=max(walls["X1"].max(),walls["X2"].max())  
y_dim=max(walls["Y1"].max(),walls["Y2"].max())

grid_size=1  #size of grid in the heat map (in meters)
mf=1/float(grid_size) #multiplication factor since the number of grids in the heat map increases with decrease in grip size

"""The weightage of various factors on the Crowd risk Index (CRI)"""
imp_weight=2 #impatience factor
density_weight=1 #pedestrian density factor
b_weight=5
p_weight=5

"""The threshold values of various factors influencing CRI, beyond which, it is dangerous"""
imp_threshold=1     
density_threshold=5
b_threshold=1
p_threshold=1

max_cri=imp_threshold*imp_weight+density_threshold*density_weight \
        + b_threshold*b_weight + p_threshold*p_weight        #The maximum value the CRI can have without danger


x=pd.ExcelFile("Pedestrian_details.xlsx").parse("X positions")
y=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y positions")
vx=pd.ExcelFile("Pedestrian_details.xlsx").parse("X velocity")
vy=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y velocity")
ax=pd.ExcelFile("Pedestrian_details.xlsx").parse("X acceleration")
ay=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y acceleration")
b_net=pd.ExcelFile("Pedestrian_details.xlsx").parse("border force")
p_net=pd.ExcelFile("Pedestrian_details.xlsx").parse("pedestrian force")

### FUNCTIONS USED ###


def update_scatter(i,ax1,scat):
	ax1.set_title("Pedestrian path : "+str(i))
	ped_xy=[]
	for j in xrange(n_ped):
		if vx.loc[i][j]!=0 and vy.loc[i][j]!=0:
			ped_xy.append([x.loc[i][j],y.loc[i][j]])

	scat.set_offsets(ped_xy)
	return scat,


def update_heat(i,ax2,im):
	ax2.set_title("Heat map : "+str(i))
	
	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	density=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))
	impatience=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))
	border_forces=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))
	ped_forces=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))
	
	for j in xrange(n_ped):
		if vx.loc[i][j]!=0 and vy.loc[i][j]!=0:
			density[int(math.floor(y.loc[i][j]))][int(math.floor(x.loc[i][j]))]-=1
			
			imp=1-(math.sqrt(vx.loc[i][j]**2 + vy.loc[i][j]**2))/v_desired_max
			impatience[int(math.floor(y.loc[i][j]*mf))][int(math.floor(x.loc[i][j]*mf))]-=imp

			border_forces[int(math.floor(y.loc[i][j]))][int(math.floor(x.loc[i][j]))]-=b_net.loc[i][j]
			ped_forces[int(math.floor(y.loc[i][j]))][int(math.floor(x.loc[i][j]))]-=p_net.loc[i][j]


	cri=density_weight*density + imp_weight*impatience + b_weight*border_forces + p_weight * ped_forces

	im.set_data(cri)
	return im,



if __name__ == '__main__':
	

	### SETUP SCATTER PLOT ###

	fig=plt.figure(figsize=(10,5)) #create graph
	plt.suptitle("Animation graphs") #set title for graph

	ax1=fig.add_subplot(1,2,1) #create 1 x 2 subplot in the graph and assigns the first subplot to ax1
	ax1.set_title("Pedestrian path")
	ax1.set_xlim(0,x_dim)  #set x limits to first subplot
	ax1.set_ylim(0,y_dim)  #set y limits to first subplot
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax1.plot([row[1],row[3]],[row[2],row[4]]) #this will plot a line using the x and y coordinates of the wall entered in the excel
	scat=ax1.scatter([],[]) #initial value of scatterplot


	### SETUP HEAT MAP ###

	ax2=fig.add_subplot(1,2,2) #create 1 x 2 subplot in the graph and assigns the second subplot to ax2
	ax2.set_title("Heat map")
	ax2.set_xlim(0,x_dim*mf)  #set x limits to second subplot
	ax2.set_ylim(0,y_dim*mf)  #set y limits to second subplot
	ax2.get_xaxis().set_visible(False) #removes the scale display of the heat map
	ax2.get_yaxis().set_visible(False) #removes the scale display of the heat map
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax2.plot([row[1],row[3]],[row[2],row[4]]) #this will plot a line using the x and y coordinates of the wall entered in the excel
	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))
	im=ax2.imshow(cri,cmap='hot',interpolation='nearest',vmin=-(max_cri),vmax=0)


	### RUN ANIMATION ###
	anim=animation.FuncAnimation(fig,update_scatter,fargs=(ax1,scat),frames=loops,interval=50)
	anim2=animation.FuncAnimation(fig,update_heat,fargs=(ax2,im),frames=loops,interval=50)
	plt.show()
