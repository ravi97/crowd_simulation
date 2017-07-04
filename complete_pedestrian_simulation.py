"""
The following commands are for importing the required packages to run the script.
You will need to install the following modules in python : numpy, openpyxl,matplotlib,pandas
"""
import matplotlib.pyplot as plt
import math
import pandas as pd
import numpy as np
import random


""" GLOBAL VARIABLES"""

t_int=0.05     #interval time (in seconds)
t_total=20     #total running time (in seconds)
loops=int(t_total/t_int) #calculates the number of loops to be performed
grid_size=0.5  #size of grid in the heat map (in meters)
n_ped=20   #number of pedestrians
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

points=pd.ExcelFile("points.xlsx").parse("Sheet2") #a dataframe loaded from the excel file containing the points 
walls=pd.ExcelFile("walls.xlsx").parse("Sheet2")  #a dataframe loaded from the excel file containing the coordinates of the wall
checkpoints=len(points.index)   #number of points the pedestrian have to cross

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
	

	def __init__(self,x,y,vx,vy):
		"""
		This is a constructor method.
		It is called when a new object is created
		"""
		

		self.x=x      #x position
		self.y=y      #y position
		self.vx=vx    #x velocity
		self.vy=vy    #y velocity

		self.init_constants() #invokes the function init_constants 
		self.side=self.calc_side() #invokes the function calc side and assign it to the variable side
		self.calc_desired_velocity() #invokes the function to calculate desired velocity
		
		
		

	def calc_desired_velocity(self):

		self.imp=1-(math.sqrt(self.vx**2+self.vy**2))/self.vd_net
		self.vd_net= (1-self.imp)*self.vd_net + self.imp*v_desired_max
		
		newside=self.calc_side()
		if self.side*newside<0:
			if self.point<checkpoints-1:
				self.point+=1
				self.side=self.calc_side()
			else:
				self.end=True
		x1=points.loc[self.point][0]
		y1=points.loc[self.point][1]
		x2=points.loc[self.point][2]
		y2=points.loc[self.point][3]

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
		self.starting_frame=random.randint(0,loops)

		self.t_relax=random.uniform(0.9,1.1)
		self.rad=random.uniform(0.5,0.6)

		self.a_b=random.uniform(0.9,1.1)
		self.b_b=random.uniform(0.27,0.33)

		self.a1=random.uniform(0.027,0.033)
		self.b1=random.uniform(0.18,0.22)
		self.a2=random.uniform(0.045,0.055)
		self.b2=random.uniform(0.18,0.22)

	def calc_side(self):
		x1=points.loc[self.point][0]
		x2=points.loc[self.point][2]
		y1=points.loc[self.point][1]
		y2=points.loc[self.point][3]
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

	if p.start==False or p.end==True:
		return 0.0,0.0

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

	if p.start==False or p.end==True:
		return f_x,f_y

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

	if p.start==False or p.end==True:
		return f_x,f_y

	for i in xrange(n_ped): #iterates through all the pedestrians from the list 
		dist=math.sqrt((ped[i].x-p.x)**2+(ped[i].y-p.y)**2) #distance of the i th pedestrian from pedestrian p 
		
		if dist==0: #distance is zero indicates that i th pedestrian is p 
			continue #since the pedestrian does not exert any force on themselves
		else:
			if ped[i].start==True:
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
	fx_total=[]   #stores the forces experienced by all the pedestrians in the x direction
	fy_total=[]   #stores the forces experienced by all the pedestrians in the y direction
	for i in xrange(n_ped):  #this loop iterates through all the pedestrians and calculates the force on the pedestrians
		#this statement calls the three force functions, and obtains the net force of each pedestrian in the x and y directions
		
		ped[i].calc_desired_velocity()
		f_total=[sum(x) for x in zip( driving_force(ped[i]) , border_repulsion(ped[i]) , pedestrian_repulsion(ped,ped[i]))]
		fx_total.append(f_total[0])    #net force of all the pedestrians in the x direction
		fy_total.append(f_total[1])    #net force of all the pedestrians in the y direction


	for i in xrange(n_ped):    #this loop updates the position and velocity of each pedestrian using the forces obtained
		
		if ped[i].start==True and ped[i].end==False: 
			ped[i].x+=ped[i].vx*t_int+0.5*(fx_total[i]/m_ped)*t_int*t_int  # s = ut + 0.5 at^2 in the x direction
			ped[i].y+=ped[i].vy*t_int+0.5*(fy_total[i]/m_ped)*t_int*t_int  # s = ut + 0.5 at^2 in the y direction
			ped[i].vx+=(fx_total[i]/m_ped)*t_int  # v = u + at in the x direction
			ped[i].vy+=(fy_total[i]/m_ped)*t_int  # v = u + at in the y direction
		else:
			if frame==ped[i].starting_frame:
				ped[i].start=True

'''
Functions for plotting the data
'''
def heat_maps(ped):
	"""
	This method takes the pedestrian details ped and generates heat map going from white to dark red
	"""

	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	density=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))
	impatience=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf)))


	for j in xrange(n_ped): #iterates through each pedestrian
		if ped[j].x>=0 and ped[j].x<=x_dim and ped[j].y>=0 and ped[j].y<=y_dim:
			if ped[j].start==True and ped[j].end==False:
				density[int(math.floor(ped[j].y*mf))][int(math.floor(ped[j].x*mf))]-=1 #decrements if pedestrian is present in the cell
				impatience[int(math.floor(ped[j].y*mf))][int(math.floor(ped[j].x*mf))]=-ped[j].imp

	cri=density_weight*density + imp_weight*impatience
	im1.set_data(cri) #sets data for the heat map
	im1.axes.figure.canvas.draw() #draws the canvas

def animate(ped):
	"""
	This method takes the pedestian details ped and animates their motion
	"""
	x=[]
	y=[]
	ax2.cla()

	for j in range(n_ped):
		if ped[j].start==True and ped[j].end==False:
			if ped[j].x<x_dim:
				x.append(ped[j].x)
				y.append(ped[j].y)

	ax2.set_xlim(0,x_dim)
	ax2.set_ylim(0,y_dim)
	ax2.set_title("Pedestrian locus animation")
	ax2.scatter(x,y,s=50,lw=0,facecolor='0.5')
	for row in walls.itertuples():
		ax2.plot([row[1],row[3]],[row[2],row[4]])

'''
Functions to generate and delete pedestrians
'''
def generate_pedestrians(ped):
	"""
	This function generates pedestrians within a given position and velocity.
	"""

	for i in xrange(n_ped):
		#the list is appended with instances of the pedestrian class initialised using a constructor 
		ped.append(Pedestrian(random.uniform(4,6),random.uniform(0,2),0,0))


def create_dataframe():
	
	positionX_col=[]  #column names for the pandas dataframe containing pedestrian y positions
	positionY_col=[]  #column names for the pandas dataframe containing pedestrian x positions
	impatience_col=[] #column names for the pandas dataframe containing pedestrian impatience indexes

	for i in xrange(n_ped): #iterates for n_ped number of times 
		positionX_col.append("Pedestrian "+str(i+1)+" (X)") #adds the column label for x position
		positionY_col.append("Pedestrian "+str(i+1)+" (Y)") #adds the column label for y position
		impatience_col.append("Pedestrian "+str(i+1)+" (imp)") #adds the column label for impatience

	positionX=pd.DataFrame(columns=positionX_col) #contructs empty dataframe with the required columns 
	positionY=pd.DataFrame(columns=positionY_col) #contructs empty dataframe with the required columns
	impatience=pd.DataFrame(columns=impatience_col) #contructs empty dataframe with the required columns

	return positionX,positionY,impatience

def add_dataframe(positionX,positionY,impatience):

	positionX.loc[i]=[a.x for a in ped] #adds the x position of all the pedestrians to the data frame
	positionY.loc[i]=[a.y for a in ped] #adds the y position of all the pedestrians to the data frame
	impatience.loc[i]=[a.imp for a in ped] #adds the impatience of all the pedestrians to the data frame


def save_as_excel(positionX,positionY,impatience):
	"""
	This method saves the pandas dataframe into excel.
	"""
	writer=pd.ExcelWriter("Pedestrian_details.xlsx") #creates an excel writes
	positionX.to_excel(writer,sheet_name="X positions") #writes the dataframe into the excel file in the given sheet
	positionY.to_excel(writer,sheet_name="Y positions") #writes the dataframe into the excel file in the given sheet
	impatience.to_excel(writer,sheet_name="impatience") #writes the dataframe into the excel file in the given sheet
	writer.save() #saves the excel file in the same diretory as this script





if __name__=="__main__":
	ped=[]  #list of pedestrian instances

	generate_pedestrians(ped)  #generate some pedestrians initially
	positionX,positionY,impatience=create_dataframe() #create dataframes in which the pedestrian data at each instance is stored


	fig=plt.figure(figsize=(10,5)) #sets the size of the figure of the plot
	plt.suptitle("Animation graphs") #set title for graph

	ax1=fig.add_subplot(1,2,1) #we create a 1 x 2 subplot and assigning ax1 to the first subplot
	ax1.set_title("Heat map animation") #sets title to the subplot
	ax1.invert_yaxis()  #inverts the y axis
	ax1.set_xlim(0,x_dim*mf) #set the x scale of the heat map (first subplot)
	ax1.set_ylim(0,y_dim*mf) #set the y scale of the heat map (first subplot)
	ax1.get_xaxis().set_visible(False) #removes the scale display of the heat map
	ax1.get_yaxis().set_visible(False) #removes the scale display of the heat map
 
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax1.plot([row[1]*mf,row[3]*mf],[row[2]*mf,row[4]*mf])  #this will plot a line using the x and y coordinates of the wall entered in the excel

	ax2=fig.add_subplot(1,2,2) #we create a 1 x 2 subplot and assigning ax2 to the second subplot
	ax2.set_title("Pedestrian locus animation") #sets title to the subplot
	for row in walls.itertuples(): #for loop to iterate through the dataframe containing the wall coordinates
		ax2.plot([row[1],row[3]],[row[2],row[4]]) #this will plot a line using the x and y coordinates of the wall entered in the excel
	

	cri=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	#This statement now creates a heat map corresponding to the pedestrian_map
	#Here, hot varies from black to red to white. So, we want black if the pedestrian cri is max and white if it is minimum
	im1=ax1.imshow(cri,cmap='hot', interpolation='nearest',vmin=-(max_cri),vmax=0) #we are using negativee value for cri since maximum is white and minimum is black when cmap is set to hot
	fig.show() #displays the plot

	for i in xrange (loops): #iterates through the loops
		sfm(ped,i) #updates the pedestrian positions and velocity by calculating the forces 
		heat_maps(ped) #generate heat map with the current pedestrian state
		animate(ped)
		add_dataframe(positionX,positionY,impatience)
		
	save_as_excel(positionX,positionY,impatience)