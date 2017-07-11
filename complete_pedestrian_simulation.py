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

t_int=0.1     #interval time (in seconds)
t_total=20     #total running time (in seconds)
loops=int(t_total/t_int) #calculates the number of loops to be performed

n_ped=200   #number of pedestrians
m_ped=1     #mass of one pedestrian
r_p=0.3     #radius of one pedestrain (in meters)
v_desired_max=1.4 #desired speed of the pedestrian (in m/s)

border_threshold=5 #distance beyond which walls dont influence (in meters)
ped_threshold=3 #distance beyond which pedestrians dont influence (in meters)


points=pd.ExcelFile("points.xlsx") #a dataframe loaded from the excel file containing the points 
walls=pd.ExcelFile("walls.xlsx").parse("Map1")  #a dataframe loaded from the excel file containing the coordinates of the wall
walls_m=walls.as_matrix() #load the dataframe into a numpy array for faster computation
rooms=4 #number of rooms in the map




class Pedestrian:
	"""
	This class is the pedestrian class, and it contains the kinematic details of a pedestrian
	"""
	

	def __init__(self):
		"""
		This is a constructor method.
		It is called when a new object is created
		"""
		self.set_path() #decides the path the pedestrians should take
		self.vx=0    #x velocity
		self.vy=0    #y velocity
		self.ax=0    #x acceleration
		self.ay=0    #y acceleration

		self.init_constants() #invokes the function init_constants 
		self.set_initial_conditions() #function to set initial conditions of the pedestrian
		self.side=self.calc_side() #invokes the function calc side and assign it to the variable side
				
		
	def set_path(self):
		self.room=random.randint(1,rooms) #randomly chooses a room in which the pedestrian starts
		if self.room==1:
			self.x=random.uniform(2,12) 
			self.y=random.uniform(15,25)
			self.path=points.parse("Path1").as_matrix()  #chooses the first path for 
			self.checkpoints=self.path.shape[0]   #number of points the pedestrian have to cross

		elif self.room==2:
			self.x=random.uniform(25,35)
			self.y=random.uniform(30,40)
			self.path=points.parse("Path2").as_matrix()
			self.checkpoints=self.path.shape[0]   #number of points the pedestrian have to cross

		elif self.room==3:
			self.x=random.uniform(25,35)
			self.y=random.uniform(15,25)
			self.path=points.parse("Path3").as_matrix()
			self.checkpoints=self.path.shape[0]   #number of points the pedestrian have to cross

		else:
			self.x=random.uniform(40,50)
			self.y=random.uniform(5,15)
			self.path=points.parse("Path4").as_matrix()
			self.checkpoints=self.path.shape[0]   #number of points the pedestrian have to cross



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
		x1=self.path[self.point][0]
		y1=self.path[self.point][1]
		x2=self.path[self.point][2]
		y2=self.path[self.point][3]

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
		
		self.starting_frame=random.randint(0,loops/2)

		self.category=random.randint(1,4)
		if self.category==1:
			self.vd_net=random.uniform(1,1.1)
		elif self.category==2:
			self.vd_net=random.uniform(1.1,1.2)
		elif self.category==3:
			self.vd_net=random.uniform(1.2,1.3)
		elif self.category==4:
			self.vd_net=random.uniform(1.3,1.4)

		self.t_relax=random.uniform(0.9,1.1)
		self.rad=random.uniform(0.5,0.6)

		self.a_b=random.uniform(0.9,1.1)
		self.b_b=random.uniform(0.27,0.33)

		self.a1=random.uniform(0.027,0.033)
		self.b1=random.uniform(0.18,0.22)
		self.a2=random.uniform(0.045,0.055)
		self.b2=random.uniform(0.18,0.22)

	def set_initial_conditions(self):
		self.point=0
		self.end=False
		self.start=False

		self.b_net=0
		self.p_net=0


	def calc_side(self):
		x1=self.path[self.point][0]
		x2=self.path[self.point][2]
		y1=self.path[self.point][1]
		y2=self.path[self.point][3]
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

	#return 0.0,0.0

	f_x=0
	f_y=0


	for i in range(walls_m.shape[0]):
		A=walls_m[i][1]-walls_m[i][3]                #we are finding the perpendicular distance between the wall and the pedestrian
		B=walls_m[i][2]-walls_m[i][0]                #we know (x1,y1) and (x2,y2) for the line. We are converting it into Ax + By + C = 0 format
		C=walls_m[i][0]*walls_m[i][3]-walls_m[i][2]*walls_m[i][1]  # A= y1 - y2 , B = x2 - x1 , C = x1*y2 - x2*y1

		m=p.x  #pedestrian coordinates
		n=p.y

		dist=abs((A*m+B*n+C)/math.sqrt(A*A+B*B)) # distance=mod(Am+Bn+C)/root(A^2+B^2)

		t_x= m - A*(A*m+B*n+C)/(A*A+B*B)  #coordinates of the foot of the perpendicular from the pedestrian location (m,n)	
		t_y= n - B*(A*m+B*n+C)/(A*A+B*B)  


		u_x=(m - t_x)/dist   #x component of unit vector from the pedestrian to the perpendicular of the wall
		u_y=(n - t_y)/dist   #y component of unit vector from the pedestrian to the perpendicular of the wall

		if dist<border_threshold: #walls outside this threshold wont influence
			if (t_x-walls_m[i][0])*(t_x-walls_m[i][2]) <= 0 and (t_y-walls_m[i][1])*(t_y-walls_m[i][3]) <= 0: #the foot must lie on the wall
				f_x+=p.a_b*math.exp((p.rad-dist)/p.b_b)*u_x
				f_y+=p.a_b*math.exp((p.rad-dist)/p.b_b)*u_y

				p.b_net+=math.sqrt(f_x**2 + f_y**2)

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
	
	#return 0.0,0.0

	r=0.6
	lam=0.75

	f_x=0 # net force in the x direction
	f_y=0 # net force in the y direction


	for i in xrange(n_ped): #iterates through all the pedestrians from the list
		if ped[i].start==True and ped[i].end==False: 
			
			dist=math.sqrt((ped[i].x-p.x)**2+(ped[i].y-p.y)**2) #distance of the i th pedestrian from pedestrian p 
		
			if dist!=0 and dist<ped_threshold: #distance is zero indicates that i th pedestrian is p, and a pedestrian does not exert force on themselves
			
				nx=(p.x-ped[i].x)/dist  #x component of unit vector between the two pedestrians
				ny=(p.y-ped[i].y)/dist  #y component of unit vector between the two pedestrians
				cos=abs(p.x*nx+p.y*ny)  # cos of angle between the line joining two pedestrian, and the path of p
			
				#x component of force on p due to the i th pedestrian
				f_x+=p.a1*math.exp((r-dist)/p.b1)*nx*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*nx
				#y component of force on p due to the i th pedestrian 
				f_y+=p.a1*math.exp((r-dist)/p.b1)*ny*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*ny

				p.p_net+=math.sqrt(f_x**2 + f_y**2)

		
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
Functions to generate and delete pedestrians
'''
def generate_pedestrians(ped):
	"""
	This function generates pedestrians within a given position and velocity.
	"""

	for i in xrange(n_ped):
		#the list is appended with instances of the pedestrian class initialised using a constructor 
		ped.append(Pedestrian())

def save_property_matrix(ped):

	columns=[]
	for i in xrange(n_ped):
		columns.append("pedestrian "+str(i+1))

	indices=[]
	indices.append("Desired velocity")
	indices.append("Starting frame")
	indices.append("Category")
	indices.append("Room")
	indices.append("t relax")
	indices.append("radius")
	indices.append("a_b")
	indices.append("b_b")
	indices.append("a1")
	indices.append("a2")
	indices.append("b1")
	indices.append("b2")

	property_list=[]
	property_list.append([a.vd_net for a in ped])
	property_list.append([a.starting_frame for a in ped])
	property_list.append([a.category for a in ped])
	property_list.append([a.room for a in ped])
	property_list.append([a.t_relax for a in ped])
	property_list.append([a.rad for a in ped])
	property_list.append([a.a_b for a in ped])
	property_list.append([a.b_b for a in ped])
	property_list.append([a.a1 for a in ped])
	property_list.append([a.a2 for a in ped])
	property_list.append([a.b1 for a in ped])
	property_list.append([a.b2 for a in ped])

	property_matrix=pd.DataFrame(property_list,columns=columns,index=indices)
	property_matrix.to_excel("Pedestrian_properties.xlsx",sheet_name="Sheet1")

def add_dataframe(ped):

	positionX.append([a.x for a in ped]) #adds the x position of all the pedestrians to the data frame
	positionY.append([a.y for a in ped]) #adds the y position of all the pedestrians to the data frame
	velocityX.append([a.vx for a in ped]) #adds the x velocity of all the pedestrians to the data frame
	velocityY.append([a.vy for a in ped]) #adds the y velocity of all the pedestrians to the data frame
	accelerationX.append([a.ax for a in ped]) #adds the x acceleration of all the pedestrians to the data frame
	accelerationY.append([a.ay for a in ped]) #adds the y acceleration of all the pedestrians to the data frame
	border_net.append([a.b_net for a in ped])
	ped_net.append([a.p_net for a in ped])
	
	for a in ped:
		a.p_net=a.b_net=0
	


def save_as_excel():
	"""
	This method saves the pandas dataframe into excel.
	"""
	columns=[]
	for i in xrange(n_ped):
		columns.append("pedestrian "+str(i+1))

	px=pd.DataFrame(np.array(positionX),columns=columns)
	py=pd.DataFrame(np.array(positionY),columns=columns)
	vx=pd.DataFrame(np.array(velocityX),columns=columns)
	vy=pd.DataFrame(np.array(velocityY),columns=columns)
	ax=pd.DataFrame(np.array(accelerationX),columns=columns)
	ay=pd.DataFrame(np.array(accelerationY),columns=columns)
	b_net=pd.DataFrame(np.array(border_net),columns=columns)
	p_net=pd.DataFrame(np.array(ped_net),columns=columns)

	writer=pd.ExcelWriter("Pedestrian_details.xlsx") #creates an excel writes

	px.to_excel(writer,sheet_name="X positions") #writes the dataframe into the excel file in the given sheet
	py.to_excel(writer,sheet_name="Y positions") #writes the dataframe into the excel file in the given sheet
	vx.to_excel(writer,sheet_name="X velocity") #writes the dataframe into the excel file in the given sheet
	vy.to_excel(writer,sheet_name="Y velocity") #writes the dataframe into the excel file in the given sheet
	ax.to_excel(writer,sheet_name="X acceleration") #writes the dataframe into the excel file in the given sheet
	ay.to_excel(writer,sheet_name="Y acceleration") #writes the dataframe into the excel file in the given sheet
	b_net.to_excel(writer,sheet_name="border force")
	p_net.to_excel(writer,sheet_name="pedestrian force")

	writer.save() #saves the excel file in the same diretory as this script


	


if __name__=="__main__":

	############## CALCULATION ########################
	
	ped=[]  #list of pedestrian instances

	generate_pedestrians(ped)  #generate some pedestrians initially
	save_property_matrix(ped)

	positionX=[]
	positionY=[]
	velocityX=[]
	velocityY=[]
	accelerationX=[]
	accelerationY=[]
	border_net=[]
	ped_net=[]
	
	for frame in xrange (loops): #iterates through the loops
		sfm(ped,frame) #updates the pedestrian positions and velocity by calculating the forces 
		add_dataframe(ped)
	
	save_as_excel()