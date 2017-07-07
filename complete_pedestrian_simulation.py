"""
The following commands are for importing the required packages to run the script.
You will need to install the following modules in python : numpy, openpyxl, xlrd, matplotlib, pandas
"""
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import pandas as pd
import numpy as np
import random


""" GLOBAL VARIABLES"""

t_int=0.05     #interval time (in seconds)
t_total=20     #total running time (in seconds)
loops=int(t_total/t_int) #calculates the number of loops to be performed
grid_size=1  #size of grid in the heat map (in meters)
n_ped=50   #number of pedestrians
m_ped=1    #mass of one pedestrian
r_p=0.3   #radius of one pedestrain (in meters)
v_desired_max=1.4 #desired speed of the pedestrian (in m/s)
border_threshold=5 #distance beyond which walls dont influence (in meters)

"""The weightage of various factors on the Crowd risk Index (CRI)"""
imp_weight=2 #impatience factor
density_weight=1 #pedestrian density project

"""The threshold values of various factors influencing CRI, beyond which, it is dangerous"""
imp_threshold=1     
density_threshold=5

max_cri=imp_threshold*imp_weight+density_threshold*density_weight #The maximum value the CRI can have without danger

points=pd.ExcelFile("points.xlsx") #a dataframe loaded from the excel file containing the points 
walls=pd.ExcelFile("walls.xlsx").parse("Map1")  #a dataframe loaded from the excel file containing the coordinates of the wall
rooms=4 #number of rooms in the map

#Here, the arena is a rectangle of dimension (0,x_dim) and (0,y_dim)
#So, x_dim and y_dim are taken as the largest coordinate of the wall
#in order to accomdate all the walls within the arena
x_dim=max(walls["X1"].max(),walls["X2"].max())  
y_dim=max(walls["Y1"].max(),walls["Y2"].max()) 

mf=1/float(grid_size) #multiplication factor since the number of grids in the heat map increases with decrease in grip size




class Pedestrian:
	"""
	This class is the pedestrian class, and it contains the kinematic details of a pedestrian
	"""
	

	def __init__(self):
		"""
		This is a constructor method.
		It is called when a new object is created
		"""
		self.set_path()
		self.vx=0    #x velocity
		self.vy=0    #y velocity
		self.ax=0    #x acceleration
		self.ay=0    #y acceleration

		self.init_constants() #invokes the function init_constants 
		self.side=self.calc_side() #invokes the function calc side and assign it to the variable side
		self.calc_desired_velocity() #invokes the function to calculate desired velocity
		
		
	def set_path(self):
		room=random.randint(1,rooms)
		if room==1:
			self.x=random.uniform(2,12)
			self.y=random.uniform(15,25)
			self.path=points.parse("Path1")
			self.checkpoints=len(self.path.index)   #number of points the pedestrian have to cross

		elif room==2:
			self.x=random.uniform(25,35)
			self.y=random.uniform(30,40)
			self.path=points.parse("Path2")
			self.checkpoints=len(self.path.index)   #number of points the pedestrian have to cross

		elif room==3:
			self.x=random.uniform(25,35)
			self.y=random.uniform(15,25)
			self.path=points.parse("Path3")
			self.checkpoints=len(self.path.index)   #number of points the pedestrian have to cross

		else:
			self.x=random.uniform(40,50)
			self.y=random.uniform(5,15)
			self.path=points.parse("Path4")
			self.checkpoints=len(self.path.index)   #number of points the pedestrian have to cross



	def calc_desired_velocity(self):

		self.imp=1-(math.sqrt(self.vx**2+self.vy**2))/self.vd_net
		self.vd_net= (1-self.imp)*self.vd_net + self.imp*v_desired_max
		
		newside=self.calc_side()
		if self.side*newside<0:
			if self.point<self.checkpoints-1:
				self.point+=1
				self.side=self.calc_side()
			else:
				self.end=True
		x1=self.path.loc[self.point][0]
		y1=self.path.loc[self.point][1]
		x2=self.path.loc[self.point][2]
		y2=self.path.loc[self.point][3]

		A=y1-y2
		B=x2-x1
		C=y2*x1-x2*y1	

		m=self.x
		n=self.y
		
		dist=abs((A*m+B*n+C)/math.sqrt(A*A+B*B))

		t_x= m - A*(A*m+B*n+C)/(A*A+B*B)  #coordinates of the foot of the perpendicular from the pedestrian location (m,n)	
		t_y= n - B*(A*m+B*n+C)/(A*A+B*B)  

		if (t_x-x1)*(t_x-x2)>=0 and (t_y-y1)*(t_y-y2)>=0:
			t_x=(x1+x2)/2
			t_y=(y1+y2)/2

		u_x=(t_x-m)/dist   #x component of unit vector from the pedestrian to the desired line
		u_y=(t_y-n)/dist   #y component of unit vector from the pedestrian to the desired line

		self.vdx=self.vd_net*u_x
		self.vdy=self.vd_net*u_y		

	def init_constants(self):
		self.vd_net=random.uniform(1,1.4)

		self.point=0
		self.end=False
		self.start=False
		self.starting_frame=random.randint(0,loops/2)

		self.t_relax=random.uniform(0.9,1.1)
		self.rad=random.uniform(0.5,0.6)

		self.a_b=random.uniform(0.9,1.1)
		self.b_b=random.uniform(0.27,0.33)

		self.a1=random.uniform(0.027,0.033)
		self.b1=random.uniform(0.18,0.22)
		self.a2=random.uniform(0.045,0.055)
		self.b2=random.uniform(0.18,0.22)

	def calc_side(self):
		x1=self.path.loc[self.point][0]
		x2=self.path.loc[self.point][2]
		y1=self.path.loc[self.point][1]
		y2=self.path.loc[self.point][3]
		return (y1-y2)*(self.x-x1)+(self.y-y1)*(x2-x1)





'''
Functions for determining the position and velocity of the pedestrians
by calculating the forces acting on them 
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

	f_x=0
	f_y=0


	for row in walls.itertuples():
		A=row[2]-row[4]                #we are finding the perpendicular distance between the wall and the pedestrian
		B=row[3]-row[1]                #we know (x1,y1) and (x2,y2) for the line. We are converting it into Ax + By + C = 0 format
		C=row[1]*row[4]-row[3]*row[2]  # A= y1 - y2 , B = x2 - x1 , C = x1*y2 - x2*y1

		m=p.x  #pedestrian coordinates
		n=p.y

		dist=abs((A*m+B*n+C)/math.sqrt(A*A+B*B)) # distance=mod(Am+Bn+C)/root(A^2+B^2)

		t_x= m - A*(A*m+B*n+C)/(A*A+B*B)  #coordinates of the foot of the perpendicular from the pedestrian location (m,n)	
		t_y= n - B*(A*m+B*n+C)/(A*A+B*B)  


		u_x=(m - t_x)/dist   #x component of unit vector from the pedestrian to the perpendicular of the wall
		u_y=(n - t_y)/dist   #y component of unit vector from the pedestrian to the perpendicular of the wall

		if dist<border_threshold: #walls outside this threshold wont influence
			if (t_x-row[1])*(t_x-row[3]) <= 0 and (t_y-row[2])*(t_y-row[4]) <= 0: #the foot must lie on the wall
				f_x+=p.a_b*math.exp((p.rad-dist)/p.b_b)*u_x
				f_y+=p.a_b*math.exp((p.rad-dist)/p.b_b)*u_y
	
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
		dist=math.sqrt((ped[i].x-p.x)**2+(ped[i].y-p.y)**2) #distance of the i th pedestrian from pedestrian p 
		
		if dist==0: #distance is zero indicates that i th pedestrian is p 
			continue #since the pedestrian does not exert any force on themselves
		else:
			if ped[i].start==True and ped[i].end==False:
				nx=(p.x-ped[i].x)/dist  #x component of unit vector between the two pedestrians
				ny=(p.y-ped[i].y)/dist  #y component of unit vector between the two pedestrians
				cos=abs(p.x*nx+p.y*ny)  # cos of angle between the line joining two pedestrian, and the path of p
			
				#x component of force on p due to the i th pedestrian
				f_x+=p.a1*math.exp((r-dist)/p.b1)*nx*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*nx
				#y component of force on p due to the i th pedestrian 
				f_y+=p.a1*math.exp((r-dist)/p.b1)*ny*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*ny
			
	
	return f_x,f_y

def sfm(ped,frame):
	"""
	This is the social force model function.
	It takes the list containing all the pedestrian instances as arguement, 
	and updates the pedestrian kinematic values with the ones after a time of 't_int'   
	"""
	
	for i in xrange(n_ped):  #this loop iterates through all the pedestrians and calculates the force on the pedestrians
		#this statement calls the three force functions, and obtains the net force of each pedestrian in the x and y directions
		#It then updates the positions and velocities of the pedestrians using kinematic equations
		
		if ped[i].start==True and ped[i].end==False:
			ped[i].calc_desired_velocity()
			f_total=[sum(x) for x in zip( driving_force(ped[i]) , border_repulsion(ped[i]) , pedestrian_repulsion(ped,ped[i]))]
			ped[i].ax=f_total[0]/m_ped
			ped[i].ay=f_total[1]/m_ped

			ped[i].x+=ped[i].vx*t_int+0.5*ped[i].ax*t_int*t_int  # s = ut + 0.5 at^2 in the x direction
			ped[i].y+=ped[i].vy*t_int+0.5*ped[i].ay*t_int*t_int  # s = ut + 0.5 at^2 in the y direction
			ped[i].vx+=ped[i].ax*t_int  # v = u + at in the x direction
			ped[i].vy+=ped[i].ay*t_int  # v = u + at in the y direction
		else:
			if frame==ped[i].starting_frame:
				ped[i].start=True

		
		

'''
Functions for plotting the data
'''
def update_heat_map(i,ax1,im):
	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	density=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))
	impatience=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))

	ped_x=0
	ped_y=0
	ped_vx=0
	ped_vy=0
	imp=1

	for j in xrange(n_ped):
		ped_x=positionX.loc[i][j]
		ped_y=positionY.loc[i][j]
		ped_vx=velocityX.loc[i][j]
		ped_vy=velocityY.loc[i][j]
		imp=1-(math.sqrt(ped_vx**2 + ped_vy**2))/v_desired_max
		density[int(math.floor(ped_y*mf))][int(math.floor(ped_x*mf))]-=1 #decrements if pedestrian is present in the cell
		impatience[int(math.floor(ped_y*mf))][int(math.floor(ped_x*mf))]-=imp

	cri=density_weight*density + imp_weight*impatience
	im.set_data(cri) #sets data for the heat map
	return im,


def update_scatter_plot(i,ax2,scat):
	ped_xy=[]
	for j in xrange(n_ped):
		if velocityX.loc[i][j]!=0 and velocityY.loc[i][j]!=0:
			ped_xy.append([positionX.loc[i][j] , positionY.loc[i][j]])
	scat.set_offsets(ped_xy)
	return scat,

'''
Functions to generate and delete pedestrians
'''
def generate_pedestrians(ped):
	"""
	This function generates pedestrians within a given position and velocity.
	"""

	for i in xrange(n_ped):
		#the list is appended with instances of the pedestrian class initialised using a constructor 
		ped.append(Pedestrian())


def create_dataframe():
	
	positionX_col=[]  #column names for the pandas dataframe containing pedestrian x positions
	positionY_col=[]  #column names for the pandas dataframe containing pedestrian y positions
	velocityX_col=[]  #column names for the pandas dataframe containing pedestrian x velocities  
	velocityY_col=[]  #column names for the pandas dataframe containing pedestrian y velocities
	accelerationX_col=[]  #column names for the pandas dataframe containing pedestrian x accelerations
	accelerationY_col=[]  #column names for the pandas dataframe containing pedestrian y accelerations

	for i in xrange(n_ped): #iterates for n_ped number of times 
		positionX_col.append("Pedestrian "+str(i+1)) #adds the column label for x position
		positionY_col.append("Pedestrian "+str(i+1)) #adds the column label for y position
		velocityX_col.append("Pedestrian "+str(i+1)) #adds the column label for y position
		velocityY_col.append("Pedestrian "+str(i+1)) #adds the column label for y position
		accelerationX_col.append("Pedestrian "+str(i+1)) #adds the column label for y position
		accelerationY_col.append("Pedestrian "+str(i+1)) #adds the column label for y position
		

	positionX=pd.DataFrame(columns=positionX_col) #contructs empty dataframe with the required columns 
	positionY=pd.DataFrame(columns=positionY_col) #contructs empty dataframe with the required columns
	velocityX=pd.DataFrame(columns=velocityX_col) #contructs empty dataframe with the required columns
	velocityY=pd.DataFrame(columns=velocityY_col) #contructs empty dataframe with the required columns
	accelerationX=pd.DataFrame(columns=accelerationX_col) #contructs empty dataframe with the required columns
	accelerationY=pd.DataFrame(columns=accelerationY_col) #contructs empty dataframe with the required columns

	return positionX,positionY,velocityX,velocityY,accelerationX,accelerationY

def add_dataframe(positionX,positionY,velocityX,velocityY,accelerationX,accelerationY):

	positionX.loc[i]=[a.x for a in ped] #adds the x position of all the pedestrians to the data frame
	positionY.loc[i]=[a.y for a in ped] #adds the y position of all the pedestrians to the data frame
	velocityX.loc[i]=[a.vx for a in ped] #adds the x velocity of all the pedestrians to the data frame
	velocityY.loc[i]=[a.vy for a in ped] #adds the y velocity of all the pedestrians to the data frame
	accelerationX.loc[i]=[a.ax for a in ped] #adds the x acceleration of all the pedestrians to the data frame
	accelerationY.loc[i]=[a.ay for a in ped] #adds the y acceleration of all the pedestrians to the data frame
	


def save_as_excel(positionX,positionY,velocityX,velocityY,accelerationX,accelerationY):
	"""
	This method saves the pandas dataframe into excel.
	"""
	writer=pd.ExcelWriter("Pedestrian_details.xlsx") #creates an excel writes

	positionX.to_excel(writer,sheet_name="X positions") #writes the dataframe into the excel file in the given sheet
	positionY.to_excel(writer,sheet_name="Y positions") #writes the dataframe into the excel file in the given sheet
	velocityX.to_excel(writer,sheet_name="X velocity") #writes the dataframe into the excel file in the given sheet
	velocityY.to_excel(writer,sheet_name="Y velocity") #writes the dataframe into the excel file in the given sheet
	accelerationX.to_excel(writer,sheet_name="X acceleration") #writes the dataframe into the excel file in the given sheet
	accelerationY.to_excel(writer,sheet_name="Y acceleration") #writes the dataframe into the excel file in the given sheet

	writer.save() #saves the excel file in the same diretory as this script

def read_from_excel():
	positionX=pd.ExcelFile("Pedestrian_details.xlsx").parse("X positions")
	positionY=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y positions")
	velocityX=pd.ExcelFile("Pedestrian_details.xlsx").parse("X velocity")
	velocityY=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y velocity")
	accelerationX=pd.ExcelFile("Pedestrian_details.xlsx").parse("X acceleration")
	accelerationY=pd.ExcelFile("Pedestrian_details.xlsx").parse("Y acceleration")
	return positionX,positionY,velocityX,velocityY,accelerationX,accelerationY


	


if __name__=="__main__":

	############## CALCULATION ########################
	'''
	ped=[]  #list of pedestrian instances

	generate_pedestrians(ped)  #generate some pedestrians initially
	positionX,positionY,velocityX,velocityY,accelerationX,accelerationY=create_dataframe() #create dataframes in which the pedestrian data at each instance is stored

	for i in xrange (loops): #iterates through the loops
		sfm(ped,i) #updates the pedestrian positions and velocity by calculating the forces 
		add_dataframe(positionX,positionY,velocityX,velocityY,accelerationX,accelerationY)
	
	save_as_excel(positionX,positionY,velocityX,velocityY,accelerationX,accelerationY)

	'''

	############# ANIMATION ############################
	positionX,positionY,velocityX,velocityY,accelerationX,accelerationY=read_from_excel()

	fig=plt.figure(figsize=(10,5)) #sets the size of the figure of the plot
	plt.suptitle("Animation graphs") #set title for graph

	ax1=fig.add_subplot(1,2,1) #we create a 1 x 2 subplot and assigning ax1 to the first subplot
	ax1.set_title("Heat map animation") #sets title to the subplot
	ax1.set_xlim(0,x_dim*mf) #set the x scale of the heat map (first subplot)
	ax1.set_ylim(0,y_dim*mf) #set the y scale of the heat map (first subplot)
	ax1.get_xaxis().set_visible(False) #removes the scale display of the heat map
	ax1.get_yaxis().set_visible(False) #removes the scale display of the heat map
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax1.plot([row[1]*mf,row[3]*mf],[row[2]*mf,row[4]*mf])  #this will plot a line using the x and y coordinates of the wall entered in the excel

	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	#This statement now creates a heat map corresponding to the pedestrian_map
	#Here, hot varies from black to red to white. So, we want black if the pedestrian cri is max and white if it is minimum
	im=ax1.imshow(cri,cmap='hot', interpolation='nearest',vmin=-(max_cri),vmax=0) #we are using negative value for cri since maximum is white and minimum is black when cmap is set to hot
	

	ax2=fig.add_subplot(1,2,2) #we create a 1 x 2 subplot and assigning ax2 to the second subplot
	ax2.set_title("Pedestrian locus animation") #sets title to the subplot
	ax2.set_xlim(0,x_dim)
	ax2.set_ylim(0,y_dim)
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax2.plot([row[1],row[3]],[row[2],row[4]]) #this will plot a line using the x and y coordinates of the wall entered in the excel
	scat=ax2.scatter([0],[0],s=50,lw=0,facecolor='0.5')	

	anim=animation.FuncAnimation(fig,update_heat_map,fargs=(ax1,im),frames=400,interval=500)
	anim2=animation.FuncAnimation(fig,update_scatter_plot,fargs=(ax2,scat),frames=400,interval=500)

	fig.show() #displays the plot

