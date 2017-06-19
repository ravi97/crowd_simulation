import matplotlib.pyplot as plt
import matplotlib.lines as line
import math
import pandas as pd
import numpy as np
import random


""" GLOBAL VARIABLES"""
t_int=0.05     #interval time (in seconds)
t_total=15      #total running time (in seconds)
grid_size=1  #size of grid in the heat map (in meters)
n_ped=10   #number of pedestrians
m_ped=1    #mass of one pedestrian
r_p=0.3   #radius of one pedestrain
v_desired=2 #desired speed of the pedestrian 
border_threshold=5 #distance beyond which walls dont influence

points=pd.ExcelFile("points.xlsx").parse("Sheet1")
walls=pd.ExcelFile("walls.xlsx").parse("Sheet1")

x_dim=max(walls["X1"].max(),walls["X2"].max())
y_dim=max(walls["Y1"].max(),walls["Y2"].max())

mf=1/grid_size #multiplication factor since the number of grids in the heat map increases with decrease in grip size




class Pedestrian:
	"""
	This class is the pedestrian class, and it contains the kinematic details of a pedestrian
	"""
	

	def __init__(self,x,y,vx,vy,points):
		"""
		This is a constructor method that creates a new instance 
		and initialises it with the values given in the arguement
		"""
		
		self.points=points

		self.x=x      #x position
		self.y=y      #y position
		self.vx=vx    #x velocity
		self.vy=vy    #y velocity
		self.point=0
		self.side=self.calc_side()
		self.calc_desired_velocity()
		self.init_constants()
		

	def calc_desired_velocity(self):

		
		newside=self.calc_side()
		if self.side*newside<0:
			self.point=1
			self.side=newside

		A=self.points.loc[self.point][1]-self.points.loc[self.point][3]
		B=self.points.loc[self.point][2]-self.points.loc[self.point][0]
		C=self.points.loc[self.point][3]*self.points.loc[self.point][0]-self.points.loc[self.point][2]*self.points.loc[self.point][1]	

		m=self.x
		n=self.y
		
		dist=abs((A*m+B*n+C)/math.sqrt(A*A+B*B))

		t_x= m - A*(A*m+B*n+C)/(A*A+B*B)  #coordinates of the foot of the perpendicular from the pedestrian location (m,n)	
		t_y= n - B*(A*m+B*n+C)/(A*A+B*B)  

		u_x=(t_x-m)/dist   #x component of unit vector from the pedestrian to the desired line
		u_y=(t_y-n)/dist   #y component of unit vector from the pedestrian to the desired line

		self.vdx=v_desired*u_x
		self.vdy=v_desired*u_y




	def init_constants(self):
		self.t_relax=random.uniform(0.9,1.1)
		self.rad=random.uniform(0.5,0.6)
		self.a_b=random.uniform(0.4,0.5)
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
		A=row[2]-row[4]              #we are finding the perpendicular distance between the wall and the pedestrian
		B=row[3]-row[1]               #we know (x1,y1) and (x2,y2) for the line. We are converting it into Ax + By + C = 0 format
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
			nx=(p.x-ped[i].x)/dist  #x component of unit vector between the two pedestrians
			ny=(p.y-ped[i].y)/dist  #y component of unit vector between the two pedestrians
			cos=abs(p.x*nx+p.y*ny)  # cos of angle between the line joining two pedestrian, and the path of p
			
			#x component of force on p due to the i th pedestrian
			f_x+=p.a1*math.exp((r-dist)/p.b1)*nx*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*nx
			#y component of force on p due to the i th pedestrian 
			f_y+=p.a1*math.exp((r-dist)/p.b1)*ny*(lam+(1-lam)*(1+cos)/2)+p.a2*math.exp((r-dist)/p.b2)*ny
			
	#print f_x,f_y
	return f_x,f_y


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


def heat_maps(ped):
	"""
	This method takes the pedestrian details ped and generates heat map going from white to dark red
	"""

	pedestrian_map=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	for j in xrange(n_ped): #iterates through each pedestrian
		try:
			pedestrian_map[int(math.floor(ped[j].y*mf))][int(math.floor(ped[j].x*mf))]-=1 #decrements if pedestrian is present in the cell
		except IndexError: #doesnt consider the pedestrian if he has gone out of the map
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
	for row in walls.itertuples():
		ax2.plot([row[1],row[3]],[row[2],row[4]])



def generate_pedestrians(ped):
	"""
	This function generates random pedestrians across the map.
	It takes an empty list as an arguement, and appends it with instances of pedestrian class.
	Each pedestrian has a random initial position and random x velocity(from 0 to 1 m/s towards left or right)
	Their y velocity is zero, and their desired velocity is 1.3 m/s towards left or right
	"""

	for i in xrange(n_ped):
		#the list is appended with instances of the pedestrian class initialised using a constructor 
		ped.append(Pedestrian(random.uniform(0,2),random.uniform(8,10),0,0,points))


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
	loops=int(t_total/t_int) #calculates the number of loops to be performed


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


	fig=plt.figure(figsize=(10,5)) #gets the figure of the plot
	plt.suptitle("Animation graphs")

	ax1=fig.add_subplot(1,2,1) #adds a subplot to the graph
	ax1.set_title("Heat map animation") #sets title to the subplot
	ax1.invert_yaxis()
	#ax1.set_xlim(0,x_dim)
	#ax1.set_ylim(0,y_dim)

	for row in walls.itertuples():
		ax1.plot([row[1],row[3]],[row[2],row[4]])

	ax2=fig.add_subplot(1,2,2)
	ax2.set_title("Pedestrian locus animation")
	for row in walls.itertuples():
		ax2.plot([row[1],row[3]],[row[2],row[4]])
	

	pedestrian_map=np.zeros(shape=(int(y_dim*mf),int(x_dim*mf))) #creates a numpy array for each cell of the heat map, and element value 0
	#This statement now creates a heat map corresponding to the pedestrian_map
	#Here, hot varies from black to red to white. So, we want black if the pedestrian density is 5 per m2, and white if it is 0 per m2
	im1=ax1.imshow(pedestrian_map,cmap='hot', interpolation='nearest',vmin=-5,vmax=0)
	#plt.gca().invert_yaxis() #invert y axis
	fig.show() #displays the plot

	for i in xrange (loops): #iterates through the loops
		sfm(ped) #updates the pedestrian positions and velocity by calculating the forces 
		ax1.set_title(str(i))
		heat_maps(ped) #generate heat map with the current pedestrian state
		animate(ped)
		positionX.loc[i]=[a.x for a in ped] #adds the x position of all the pedestrians to the data frame
		positionY.loc[i]=[a.y for a in ped] #adds the y position of all the pedestrians to the data frame
		velocityX.loc[i]=[a.vx for a in ped] #adds the x velocity of all the pedestrians to the data frame
		velocityY.loc[i]=[a.vy for a in ped] #adds the y velocity of all the pedestrians to the data frame

	 
	plot_locus(positionX,positionX_col,positionY,positionY_col) #Comment this if you dont want to see the locus plot
	#save_as_excel(positionX,positionY,velocityX,velocityY)  #Comment this line if you dont want to save as excel 