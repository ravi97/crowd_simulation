import matplotlib.pyplot as plt
import matplotlib.animation as an
import math
import pandas as pd
import numpy as np
import os

""" DESCRIPTION OF THIS CODE 

This code is a hardcoded simulation of a two pedestrian system.

Two pedestrians, start from two locations inside a 5m x 3m arena.
There are two horizontal parallel walls located at yb1 and yb2.
The pedestrians move towards each other, and their locus is determined using the social force model.

The code has three outputs.
1. It displays the locus of the two pedestrians
2. It saves the coordinates of the pedestrians in an excel file
3. It generates heat maps of the location and saves it as png images in the heat_map_folder


"""




'''global variables'''
t_int=0.05     #interval time (in seconds)
t_total=5      #total running time (in seconds)
grid_size=0.1  #size of grid in the heat map (in meters)
heat_map_folder="/media/ravi/data/Acads/internships/IISc - Summer17/Crowd simulation/heat maps/" #the heat maps are saved as png in this directory
yb1=0      #y coordinate of the first border
yb2=3      #y coordinate of second border




def sfm(p1,p2,t):         #calculates the state of pedestrian after t_int seconds and updates p1 and p2 accordingly
	df1=driving_force(p1)
	df2=driving_force(p2)
	prf=pedestrian_repulsion(p1,p2)
	brf1=border_repulsion(p1)
	brf2=border_repulsion(p2)
	fnet_1=[sum(x) for x in zip(df1,prf[0],brf1)]
	fnet_2=[sum(x) for x in zip(df2,prf[1],brf2)]

	fx1=fnet_1[0]
	fy1=fnet_1[1]
	fx2=fnet_2[0]
	fy2=fnet_2[1]

	p1[0]+=p1[2]*t+0.5*fx1*t*t
	p1[1]+=p1[3]*t+0.5*fy1*t*t
	p1[2]+=fx1*t
	p1[3]+=fy1*t

	p2[0]+=p2[2]*t+0.5*fx2*t*t
	p2[1]+=p2[3]*t+0.5*fy2*t*t
	p2[2]=p2[2]+fx2*t
	p2[3]=p2[3]+fy2*t

def driving_force(p):   #returns driving force of a single pedestrian
	v_desired=p[4]
	t_relax=1
	df_x=(p[4]-p[2])/t_relax
	df_y=(p[5]-p[3])/t_relax
	return df_x,df_y

def pedestrian_repulsion(p1,p2):    #returns the pedestrian repulsion force between both the pedestrians
	
	"""CONSTANTS USED TO CALCULATE INTERPEDESTRIAN FORCE"""
	a1=3
	b1=0.5
	a2=1
	b2=0.5
	r=0.6
	lam=0.75

	dist=math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
	nx=(p1[0]-p2[0])/dist
	ny=(p1[1]-p2[1])/dist
	cos=-(p1[0]*nx+p1[1]*ny)

	f_x=a1*math.exp((r-dist)/b1)*nx*(lam+(1-lam)*(cos)/2)+a2*math.exp((r-dist)/b2)*nx
	f_y=a1*math.exp((r-dist)/b1)*ny*(lam+(1-lam)*(cos)/2)+a2*math.exp((r-dist)/b2)*ny

	prf1=[f_x,f_y]
	prf2=[-f_x,-f_y]
	
	return prf1,prf2

def border_repulsion(p):   #returns the border repulsion force experienced by a particular pedestrian
	

	"""CONSTANTS FOR CALCULATING BORDER FORCE"""
	a_b=1
	b_b=0.3
	r_p=0.3   #radius of one pedestrain (the walls have zero thickness)

	d1=abs(p[1]-yb1)
	d2=abs(p[1]-yb2)

	"""VECTORS"""
	mx1=0             #x component of vector from pedestrian to wall 1 
	my1=(p[1]-yb1)/d1 #y component of vector from pedestrian to wall 1
	mx2=0             #x component of vector from pedestrian to wall 2
	my2=(p[1]-yb2)/d2 #y component of vector from pedestrian to wall 2

	f_x=0
	f_y=a_b*math.exp((r_p-d1)/b_b)*my1 + a_b*math.exp((r_p-d2)/b_b)*my2


	return f_x,f_y

def heat_map(p1,p2):
	pedestrian_map=np.zeros(shape=(3*mf,5*mf))
	pedestrian_map[int(math.floor(p1[1]*mf))][int(math.floor(p1[0]*mf))]-=1
	pedestrian_map[int(math.floor(p2[1]*mf))][int(math.floor(p2[0]*mf))]-=1
	im=plt.imshow(pedestrian_map,cmap='autumn', interpolation='nearest',vmin=-2,vmax=0)
	plt.gca().invert_yaxis()
	plt.savefig(heat_map_folder+str(i)+".png")

def plot_locus(x1,y1,x2,y2):  #plots the locus of the entire journey of the two pedestrians 
	plt.plot(x1,y1,label="Pedestrian 1")
	plt.plot(x2,y2,label="Pedestrian 2")
	plt.suptitle("Locus of a two pedestrian system")
	plt.xlabel("X axis")
	plt.ylabel("Y axis")
	plt.xlim([0,5])
	plt.ylim([0,3])
	plt.legend()
	plt.show()

def save_excel(position_profile,velocity_profile):
	position_profile.to_excel("pedestrain positions.xlsx",sheet_name="Sheet1")
	position_profile.to_excel("pedestrain velocity.xlsx",sheet_name="Sheet1")

if __name__=="__main__":


	#p1 and p2 are the states of the pedestrian. They contain all the information of a pedestrian
	p1=[0,1.3,1,0,1.3,0]   #x,y,vx,vy,vdx,vdy (x coordinate,y coordinate,x velocity, y velocity,x desired velocity,y desired velocity)
	p2=[5,1.7,-1,0,-1.3,0] #similarly, for pedestrian 2

	mf=int(1/grid_size)  #multiplication factor (to incorporate grid size)

	loops=int(t_total/t_int)  #total number of loops to run




	x1=[]   #list containing all the x positions of pedestrian 1
	y1=[]   #list containing all the y positions of pedestrian 1
	x2=[]   #list containing all the x positions of pedestrian 2
	y2=[]   #list containing all the y positions of pedestrian 2

	position_profile=pd.DataFrame(columns=["Pedestrian 1 (x)","Pedestrian 1 (y)","Pedestrian 2 (x)","Pedestrian 2 (y)"]) #dataframe containing pedestrian coordinates 
	velocity_profile=pd.DataFrame(columns=["Pedestrian 1 (x)","Pedestrian 1 (y)","Pedestrian 2 (x)","Pedestrian 2 (y)"]) #dataframe containing pedestrian velocities

	
	for i in range(loops):
		sfm(p1,p2,t_int) #updates p1 and p2 after t_int seconds
		x1.append(p1[0])
		y1.append(p1[1])
		x2.append(p2[0])
		y2.append(p2[1])

		position_profile.loc[i]=[p1[0],p1[1],p2[0],p2[1]]
		velocity_profile.loc[i]=[p1[2],p1[3],p2[2],p2[3]]
		#heat_map(p1,p2)

	plot_locus(x1,y1,x2,y2)
	#save_excel(position_profile,velocity_profile)