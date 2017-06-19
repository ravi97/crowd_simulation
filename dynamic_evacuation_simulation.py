import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import random



""" GLOBAL VARIABLES"""
t_int=0.05     #interval time (in seconds)
t_total=5      #total running time (in seconds)
grid_size=1  #size of grid in the heat map (in meters)
n_ped=100   #number of pedestrians
m_ped=1    #mass of one pedestrian
r_p=0.3   #radius of one pedestrain
x_dim=20    #x dimension of the arena
y_dim=20    #y dimension of the arena
x_door=x_dim #x location of the door
y_door=y_dim/2 #y location of the door
l_door=1.5 #length of the door
v_desired=2 #desired speed of the pedestrian 

mf=float(1/grid_size) #multiplication factor since the number of grids in the heat map increases with decrease in grip size

class Pedestrian:
	"""
	This class is the pedestrian class, and it contains the kinematic details of a pedestrian
	"""

	def __init__(self,x,y,vx,vy):
		"""
		This is a constructor method that creates a new instance 
		and initialises it with the values given in the arguement
		"""
		self.x=x      #x position
		self.y=y      #y position
		self.vx=vx    #x velocity
		self.vy=vy    #y velocity
		self.calc_desired_velocity()
		self.init_constants()

	def calc_desired_velocity(self):
		dist=math.sqrt((x_door-self.x)**2 + (y_door-self.y)**2)
		nx=(x_door-self.x)/dist
		ny=(y_door-self.y)/dist
		self.vdx=v_desired*nx
		self.vdy=v_desired*ny

	def init_constants(self):
		self.t_relax=random.uniform(0.9,1.1)
		self.rad=random.uniform(0.5,0.6)
		self.a_b=random.uniform(0.9,1.1)
		self.b_b=random.uniform(0.27,0.33)
		self.a1=random.uniform(0.27,0.33)
		self.b1=random.uniform(0.18,0.22)
		self.a2=random.uniform(0.45,0.55)
		self.b2=random.uniform(0.18,0.22)




def sfm(ped):
	"""
	This is the social force model function.
	It takes the list containing all the pedestrian instances as arguement, 
	and updates the pedestrian kinematic values with the ones after a time of 't_int'   
	"""
	fx_total=[]   #stores the forces experienced by all the pedestrians in the x direction
	fy_total=[]   #stores the forces experienced by all the pedestrians in the y direction
	for i in xrange(n_ped):  #this loop iterates through all the pedestrians and calculates the force on the pedestrians
		#this statement calls the three force functions, and obtains the net force of each pedestrian in the x and y directions
		ped[i].calc_desired_velocity()
		f_total=[sum(x) for x in zip( driving_force(ped[i]) , border_repulsion(ped[i]) , pedestrian_repulsion(ped,ped[i]))]
		fx_total.append(f_total[0])    #net force of all the pedestrians in the x direction
		fy_total.append(f_total[1])    #net force of all the pedestrians in the y direction

	for i in xrange(n_ped):    #this loop updates the position and velocity of each pedestrian using the forces obtained 
		ped[i].x+=ped[i].vx*t_int+0.5*(fx_total[i]/m_ped)*t_int*t_int  # s = ut + 0.5 at^2 in the x direction
		ped[i].y+=ped[i].vy*t_int+0.5*(fy_total[i]/m_ped)*t_int*t_int  # s = ut + 0.5 at^2 in the y direction
		ped[i].vx+=(fx_total[i]/m_ped)*t_int  # v = u + at in the x direction
		ped[i].vy+=(fy_total[i]/m_ped)*t_int  # v = u + at in the y direction
		'''
		if ped[i].y<r_p:    #stops the pedestrian from crossing the bottom wall
			ped[i].y=r_p
			ped[i].vy=0
		if ped[i].y>(y_dim-r_p):   #stops the pedestrian from crossing the top wall
			ped[i].y=y_dim-r_p
			ped[i].vy=0
		if ped[i].x<r_p:    #stops the pedestrian from crossing the left wall
			ped[i].x=r_p
			ped[i].vx=0
		if ped[i].x>(x_dim-r_p):   #stops the pedestrian from crossing the right wall
			if ped[i].y>(y_door+l_door/2) or ped[i].y<(y_door-l_door/2):  #																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																															
				ped[i].x=x_dim-r_p
				ped[i].vx=0
		'''


def driving_force(p):
	"""
	This method calculates and returns the driving force experienced by a single pedestrian p
	"""


	"""Constants"""
	#t_relax=1  #relaxation time

	df_x=(p.vdx-p.vx)/p.t_relax  #force in the x direction
	df_y=(p.vdy-p.vy)/p.t_relax  #force in the y direction
	return df_x,df_y


def border_repulsion(p):
	"""
	This method calculates and returns the border repulsion force experienced by a single pedestrian p 
	"""

	"""Constants"""
	#a_b=1
	#b_b=0.3
	

	d1=abs(p.y)       #distance of the pedestrian from bottom wall
	d2=abs(p.x-x_dim) #distance of the pedestrian from right wall     
	d3=abs(p.y-y_dim) #distance of the pedestrian from top wall
	d4=abs(p.x)       #distance of the pedestrian from left


	"""unit vectors"""
	m1=(p.y)/d1          #unit vector from bottom wall 
	m2=(p.x-x_dim)/d2    #unit vector from right wall
	m3=(p.y-y_dim)/d3    #unit vector from top wall
	m4=(p.x)/d4          #unit vector from left wall

	if p.y<(y_door+l_door/2) and p.y>(y_door-l_door/2):
		f_x=p.a_b*math.exp((p.rad-d4)/p.b_b)*m4    #when near the door, only left wall will exert a force in the x direction
	else:
		f_x=p.a_b*math.exp((p.rad-d2)/p.b_b)*m2 + p.a_b*math.exp((p.rad-d4)/p.b_b)*m4   #else,both left and right wall exert force in the x direction
	f_y=p.a_b*math.exp((p.rad-d1)/p.b_b)*m1 + p.a_b*math.exp((p.rad-d3)/p.b_b)*m3       #top and bottom walls exert force in the y direction

	return f_x,f_y

def pedestrian_repulsion(ped,p):
	"""
	This method calculates and returns the pedestrain repulsion force experienced by a single pedestrain p
	due to all the other pedestrians ped
	"""

	"""Constants"""
	#a1=0.3
	#b1=0.2
	#a2=0.5
	#b2=0.2
	r=0.6
	lam=0.75

	f_x=0 # net force in the x direction
	f_y=0 # net force in the y direction

	for i in xrange(n_ped): #iterates through all the pedestrians from the list 
		dist=math.sqrt((ped[i].x-p.x)**2+(ped[i].y-p.y)**2) #distance of the i th pedestrian from p   
		
		if dist==0: #distance is zero indicates that i th pedestrian is p 
			continue #since the pedestrian does not exert any force on themselves
		else:
			nx=(p.x-ped[i].x)/dist  #x component of unit vector between the two pedestrians
			ny=(p.y-ped[i].y)/dist  #y component of unit vector between the two pedestrians
			cos=abs(p.x*nx+p.y*ny)  # cos of angle between the line joining two pedestrian, and the path of p

			#x component of force on p due to the i th pedestrian
			f_x+=p.a1*math.exp((r-dist)/p.b1)*nx*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*nx
			#y component of force on p due to the i th pedestrian 
			f_y+=p.a1*math.exp((r-dist)/p.b1)*ny*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*ny
			
			
			

	return f_x,f_y


def plot_locus(positionX,positionX_col,positionY,positionY_col):
	"""
	This method takes the pandas dataframe containing all the pedestrian positions at all instances of time as arguement,
	and plots the locus graph 
	"""
	plt.clf() #clears the graph before plotting
	plt.figure(figsize=(5,5))
	plt.suptitle("Locus graph")
	for i in xrange(n_ped): #iterates through each pedestrian
		x=positionX[positionX_col[i]].tolist() #take the specific column containing x positions of pedestrian i
		y=positionY[positionY_col[i]].tolist() #and convert into list. Similarly, for y positions.
		plt.plot(x,y) #plots the locus line of pedestrian i 

	plt.suptitle("Locus of a "+str(n_ped)+" pedestrian system") #adds title to graph
	plt.xlim([0,x_dim])  #sets x scale for the graph
	plt.ylim([0,y_dim])  #sets y scale for the graph
	plt.xlabel("X axis") #sets x axis label
	plt.ylabel("Y axis") #sets y axis label
	plt.show() #displays plot


def heat_maps(ped):
	"""
	This method takes the pedestrian details ped and generates heat map going from white to dark red
	"""

	pedestrian_map=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	for j in xrange(n_ped): #iterates through each pedestrian
		try:
			pedestrian_map[int(math.floor(ped[j].y*mf))][int(math.floor(ped[j].x*mf))]-=1 #decrements if pedestrian is present in the cell
		except IndexError: #doesnt consider the pedestrian if he haa gone out of the map
			continue
	im1.set_data(pedestrian_map) #sets data for the heat map
	im1.axes.figure.canvas.draw() #draws the canvas
	
def animate(ped):
	"""
	This method takes the pedestian details ped and animates their motion
	"""
	x=[]
	y=[]
	ax2.cla()
	for j in range(n_ped):
		if ped[j].x<x_dim:
			x.append(ped[j].x)
			y.append(ped[j].y)
	ax2.set_xlim(0,x_dim)
	ax2.set_ylim(0,y_dim)
	ax2.scatter(x,y,s=50,lw=0,facecolor='0.5')


def generate_pedestrians(ped):
	"""
	This function generates random pedestrians across the map.
	It takes an empty list as an arguement, and appends it with instances of pedestrian class.
	Each pedestrian has a random initial position and random x velocity(from 0 to 1 m/s towards left or right)
	Their y velocity is zero, and their desired velocity is 1.3 m/s towards left or right
	"""

	for i in xrange(n_ped):
		#the list is appended with instances of the pedestrian class initialised using a constructor 
		ped.append(Pedestrian(random.uniform(0,x_dim),random.uniform(0,y_dim),0,0))

def save_as_excel(positionX,positionY,velocityX,velocityY):
	"""
	This method saves the pandas dataframe into excel.
	"""
	writer=pd.ExcelWriter("Pedestrian_details.xlsx") #creates an excel writes
	positionX.to_excel(writer,sheet_name="X positions") #writes the dataframe into the excel file in the given sheet
	positionY.to_excel(writer,sheet_name="Y positions") #writes the dataframe into the excel file in the given sheet
	velocityX.to_excel(writer,sheet_name="X velocities") #writes the dataframe into the excel file in the given sheet
	velocityY.to_excel(writer,sheet_name="Y velocities") #writes the dataframe into the excel file in the given sheet
	writer.save() #saves the excel file in the same diretory as this script


if __name__=="__main__":
	ped=[]  #list of pedestrian instances

	generate_pedestrians(ped) 

	positionX_col=[]  #column names for the pandas dataframe containing pedestrian positions
	positionY_col=[]  #column names for the pandas dataframe containing pedestrian positions
	velocityX_col=[]  #column names for the pandas dataframe containing pedestrian velocities
	velocityY_col=[]  #column names for the pandas dataframe containing pedestrian velocities
 

 
	for i in xrange(n_ped): #iterates for n_ped number of times 
		positionX_col.append("Pedestrian "+str(i+1)+" (X)") #adds the column label for x position
		positionY_col.append("Pedestrian "+str(i+1)+" (Y)") #adds the column label for y position
		velocityX_col.append("Pedestrian "+str(i+1)+" (X)") #adds the column label for x velocity
		velocityY_col.append("Pedestrian "+str(i+1)+" (Y)") #adds the column label for y velocity

	positionX=pd.DataFrame(columns=positionX_col) #contructs empty dataframe with the required columns 
	positionY=pd.DataFrame(columns=positionY_col) #contructs empty dataframe with the required columns
	velocityX=pd.DataFrame(columns=velocityX_col) #contructs empty dataframe with the required columns
	velocityY=pd.DataFrame(columns=velocityY_col) #contructs empty dataframe with the required columns

	loops=int(t_total/t_int) #calculates the number of loops to be performed

	fig=plt.figure(figsize=(10,5)) #gets the figure of the plot
	plt.suptitle("Animation graphs")

	ax1=fig.add_subplot(1,2,1) #adds a subplot to the graph
	ax1.set_title("Heat map animation") #sets title to the subplot
	ax1.set_xlim(0,x_dim)
	ax1.set_ylim(0,y_dim)

	ax2=fig.add_subplot(1,2,2)
	ax2.set_title("Pedestrian locus animation")
	

	pedestrian_map=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	#This statement now creates a heat map corresponding to the pedestrian_map
	#Here, hot varies from black to red to white. So, we want black if the pedestrian density is 5 per m2, and white if it is 0 per m2
	im1=ax1.imshow(pedestrian_map,cmap='hot', interpolation='nearest',vmin=-5,vmax=0)
	#plt.gca().invert_yaxis() #invert y axis
	fig.show() #displays the plot

	for i in xrange (loops): #iterates through the loops
		sfm(ped) #updates the pedestrian positions and velocity by calculating the forces 
		positionX.loc[i]=[a.x for a in ped] #adds the x position of all the pedestrians to the data frame
		positionY.loc[i]=[a.y for a in ped] #adds the y position of all the pedestrians to the data frame
		velocityX.loc[i]=[a.vx for a in ped] #adds the x velocity of all the pedestrians to the data frame
		velocityY.loc[i]=[a.vy for a in ped] #adds the y velocity of all the pedestrians to the data frame
		ax1.set_title(str(i))
		heat_maps(ped) #generate heat map with the current pedestrian state
		animate(ped)

	#save_as_excel(positionX,positionY,velocityX,velocityY)  #Comment this line if you dont want to save as excel 
	plot_locus(positionX,positionX_col,positionY,positionY_col) #Comment this if you dont want to see the locus plot