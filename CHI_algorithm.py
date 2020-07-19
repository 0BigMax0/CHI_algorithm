# -*- coding: utf-8 -*-
"""
Created on Sat Jul 18 09:46:49 2020

@author: JianZhong

"""

import time
import os
import re
import matplotlib.pyplot as plt

''' ===========================================================================
#                ***********  Import data   ************
# ============================================================================='''

def distance(A,B):
    "calculate the distance between the point A and point B"
    return abs(A-B)

def tour_length(tour):
    "The total of distances between each pair of consecutive cities in the tour."
    return sum( distance(tour[i], tour[i-1] ) 
               for i in range(len(tour)))

def DateFilePath(file_dir,path1): 
    "generate the visiting path of the data."
    data_file_size = []
    data_file_list = os.listdir(file_dir) 
    for i in data_file_list:#逐个提取数据文件中节点的数量
        data_file_size.append( int(re.findall(r'\d+',i)[0]))
    temp = data_file_size.copy()#排序：对数据文件中的节点数量从小到大排序
    temp1 = temp.copy()
    temp1.sort()   
    file_name = [] #对文件名进行排序
    for i in temp1:
        file_name.append(data_file_list.pop(temp.index(i)))
        temp.remove(i)
    data = open(path1,'r').readlines()#导入tsplib的最优值
    data_set_name = []
    data_solution = []
    for i in range(len(data)):
        a = data[i].strip('\n').split(' ')
        data_set_name.append(a[0])
        data_solution.append(int(a[1]))
    optimul_solution = []#数据集的最优值
    for i in range(len(file_name)):
        x1 = file_name[i].split('.')
        x2 = data_set_name.index(x1[0])
        optimul_solution.append(data_solution[x2])
    file_name_path = [ file_dir + '\\' + file_name[i] #列表结构，存储获取访数据文件的访问路径
           for i in range(len(file_name)) ]  
    return file_name_path,file_name,optimul_solution
 
def ClearData(file_name_path):
    "import and clean the data."
    new_data = []
    data = open(file_name_path,'r').readlines()
    for i in range(len(data)):   #清洗:数据中的空格符，换行符
        data[i] = data[i].strip('\n').split(' ')
        data[i] = [ x for x in data[i] if x !='' ]      
        if data[i] ==[]:
            continue
        else:
            new_data.append(data[i])                    
    new_data.pop()       #清洗:数据集中的尾部数据
    for i in range(20):    #定位:数据的起始行 
        if '1'  == new_data[i][0] and '2' == new_data[i+1][0]:
            data_start = i
            break       
    cities = {(complex(float(new_data[i][1]) , float(new_data[i][2])))
               for i in range(data_start,len(new_data))}
    list_cities = [(float(new_data[i][1]) , float(new_data[i][2]))
               for i in range(data_start,len(new_data))]
    return cities,list_cities

''' ===========================================================================
#                ***********  Initialisition Process  ************
# ============================================================================='''
# employed the monotone chain algorithm to generate the convex hull
def ConvexHull(points):
	return MakeHullPresorted(sorted(points))

def MakeHullPresorted(points):
	if len(points) <= 1:
		return list(points)
	upperhull = []
	lowerhull = []
	for hull in (upperhull, lowerhull):
		for p in (points if (hull is upperhull) else reversed(points)):
			while len(hull) >= 2:
				qx, qy = hull[-1]
				rx, ry = hull[-2]
				if (qx - rx) * (p[1] - ry) >= (qy - ry) * (p[0] - rx):
					del hull[-1]
				else:
					break
			hull.append(p)
		del hull[-1]
	
	if not (len(upperhull) == 1 and upperhull == lowerhull):
		upperhull.extend(lowerhull)
	return upperhull

def GenerateUnvisitedPoint(convex_tour,list_data):
    "obtained vertexes that are not visited"
    temp = list_data.copy()
    for i in convex_tour:
        if i in convex_tour:
            temp.remove(i)
    
    visited_tour = [complex(i[0],i[1]) for i in convex_tour]
    unvisited_tour = [complex(i[0],i[1]) for i in temp] 
    return visited_tour,unvisited_tour

 

''' ===========================================================================
#                ***********  Construction Process   ************
# ============================================================================='''

###########################  Selection Criterion #############################
def calculateCostMatrix(unvisited_tour,visited_tour):
    "calculate the distance between the vertex i and vertex j"
    cost_matrix = []
    min_row_cost = []
    for i in unvisited_tour:
        temp = []
        for j in range(-1,len(visited_tour)-1):
            temp.append(distance(i,visited_tour[j])+distance(i,visited_tour[j+1])-distance(visited_tour[j],visited_tour[j+1]))
        cost_matrix.append(temp)
        min_row_cost.append(min(temp))
    return cost_matrix,min_row_cost

def selectionCriterion(cost_matrix,min_row_cost,unvisited_tour):
    "SelectionCriterion:find the vertex with the cheapest cost"
    divide_insert_cost = min(min_row_cost) #查找插入成本的最小值
    divide_insert_position = cost_matrix[(min_row_cost.index(divide_insert_cost))].index(divide_insert_cost)#计算插入位置
    visit_point = unvisited_tour[(min_row_cost.index(divide_insert_cost))]#查找插入成本最小的点
    return visit_point,divide_insert_cost,divide_insert_position  


###########################  Insertion criterion #############################
######## Individual insertion strategy ############
def individualCost(list_unvisited,neighbour_point,save_matrix,value):
    "calculate the cost of the individual insertion strategy "
    row = list_unvisited.index(neighbour_point)
    value2 = min(save_matrix[row])
    divid_insert_value = value + value2
    return divid_insert_value

######## The first conbined insertion strategy ############
def findClosetPoint(list_unvisited,visit_point):
    "selection_criterion:find the closet vertex to the vertex selected "
    temp1=[]
    for k in list_unvisited:
        if k == visit_point:
            temp1.append(9999999)
        else:
            temp1.append(distance(visit_point,k))
    link_value = min(temp1)
    neighbour_point = list_unvisited[temp1.index(link_value)]
    return neighbour_point,link_value 

def firstCombinedCost(list_tour,visit_point,neighbour_point,link_value): 
    "calculate the cost of the first combined method "
    temp2 = []
    combine_value = []
    for l in range(-1,len(list_tour)-1):
        a = (link_value + distance(visit_point,list_tour[l])+distance(neighbour_point,list_tour[l+1])-distance(list_tour[l],list_tour[l+1]))
        b = (link_value + distance(neighbour_point,list_tour[l])+distance(visit_point,list_tour[l+1])-distance(list_tour[l],list_tour[l+1]))
        if a>b:
            combine_value.append(b)
            temp2.append([neighbour_point,visit_point])
        else:
            combine_value.append(a)
            temp2.append([visit_point,neighbour_point])
    combine_cost = min(combine_value)
    combine_insert_position = combine_value.index(combine_cost)
    return combine_cost,combine_insert_position

def firstCombinedStragety(list_tour,list_unvisited,visit_point,save_matrix,value,insert_position):
    neighbour_point,link_value = findClosetPoint(list_unvisited,visit_point)
    divid_insert_value = individualCost(list_unvisited,neighbour_point,save_matrix,value)
    #判断哪种插入方式更优
    max_neighbour_point_saved = 0
    neighbour_point_insert_position = 0
    if divid_insert_value<link_value:
        neighbour_point_insert_position = insert_position
    else:
        combine_cost,combine_insert_position = firstCombinedCost(list_tour,visit_point,neighbour_point,link_value) 
        if divid_insert_value<combine_cost:
            neighbour_point_insert_position = insert_position
        else:
            max_neighbour_point_saved = divid_insert_value - combine_cost
            neighbour_point_insert_position = combine_insert_position
    return max_neighbour_point_saved,neighbour_point_insert_position

######## The second conbined insertion strategy #########
def findColumnCost(cost_matrix,visit_point,unvisited_tour):  
    "find the set of vertex c that closet to the edge"
    combine_point = []
    min_column_cost = []
    if len(unvisited_tour) == 1:
        combine_point = []
        min_column_cost = []
    else:
        for i in range(len(cost_matrix[0])):
            temp = []
            for j in range(len(unvisited_tour)):
                if unvisited_tour[j] != visit_point:
                    temp.append(cost_matrix[j][i])
                else:
                    temp.append(999999)
            min_column_cost.append(min(temp))
            combine_point.append(unvisited_tour[temp.index(min(temp))])
    return min_column_cost,combine_point

def secondCombinedStragety(cost_matrix,visit_point,visited_tour,unvisited_tour,divide_insert_cost,divide_insert_position):   
    "calculate the cost of the second combined strategy"
    min_column_cost,combine_point = findColumnCost(cost_matrix,visit_point,unvisited_tour)    
    way_insert_saved = []
    way_insert_position = []    
    for i in range(len(combine_point)):
        temp_cost = []
        temp_position = []
        combine_divide_insert_cost = divide_insert_cost + min(cost_matrix[unvisited_tour.index(combine_point[i])])
        link_cost = distance(visit_point,combine_point[i])
        temp_cost.append(combine_divide_insert_cost)#记录单独插入的成本及位置
        temp_position.append(divide_insert_position)
        a = (link_cost + distance(combine_point[i],visited_tour[i-1])+distance(visit_point,visited_tour[i])-distance(visited_tour[i],visited_tour[i-1]))#选择组合插入的两种成本值较低的一种
        b = (link_cost + distance(visit_point,visited_tour[i-1])+distance(combine_point[i],visited_tour[i])-distance(visited_tour[i],visited_tour[i-1]))
        if a>b:
            temp_cost.append(b)
            temp_position.append(i)
        else:
            temp_cost.append(a)
            temp_position.append(i)
        way_insert_saved.append(combine_divide_insert_cost - min(temp_cost))  #保存该neighbour_point组合插入的节约值 
        if temp_cost.index(min(temp_cost))==0:
            way_insert_position.append(divide_insert_position)
        else:
            way_insert_position.append(i) 
    max_neighbour_edge_saved = max(way_insert_saved)#保存最临边插入法则获取的最大节约值及位置
    neighbour_edge_insert_position = way_insert_position[way_insert_saved.index(max_neighbour_edge_saved)]
    return max_neighbour_edge_saved, neighbour_edge_insert_position

''' ===========================================================================
#             ***********  The framwork of the CHI algorithm   ************
# ============================================================================='''

def ConvexHullInsert(data):#傻瓜式准则
    list_data = [(i.real,i.imag) for i in data]
    
    #Initialization process
    convex_tour = ConvexHull(list_data) #generate the sub tour
    visited_tour,unvisited_tour = GenerateUnvisitedPoint(convex_tour,list_data)
    
    # Construction process
    while len(unvisited_tour) >= 2:
        #selection criterion
        cost_matrix,min_row_cost = calculateCostMatrix(unvisited_tour,visited_tour)#计算成本矩阵   
        visit_point,divide_insert_cost,divide_insert_position = selectionCriterion(cost_matrix,min_row_cost,unvisited_tour)
        #Insertion criterion
        neighbour_edge_saved, neighbour_edge_insert_position = secondCombinedStragety(cost_matrix,visit_point,visited_tour,unvisited_tour,divide_insert_cost,divide_insert_position) 
        neighbour_point_saved,neighbour_point_insert_position = firstCombinedStragety(visited_tour,unvisited_tour,visit_point,cost_matrix,divide_insert_cost,divide_insert_position)
        #choose the location with minimum insertion cost
        if neighbour_edge_saved > neighbour_point_saved:
            visited_tour.insert(neighbour_edge_insert_position,visit_point)
        else:
            visited_tour.insert(neighbour_point_insert_position,visit_point)
        unvisited_tour.remove(visit_point)      
    cost_matrix,min_row_cost = calculateCostMatrix(unvisited_tour,visited_tour)#计算成本矩阵
    divide_insert_cost = min(min_row_cost) #查找插入成本的最小值
    visit_point = unvisited_tour[0]#查找插入成本最小的点
    divide_insert_position = cost_matrix[(min_row_cost.index(divide_insert_cost))].index(divide_insert_cost)#计算插入位置
    visited_tour.insert(divide_insert_position,visit_point) 
    return visited_tour


''' ===========================================================================
#             ***********  plot the calculation results   ************
# ============================================================================='''
def plot_tsp(algorithm, cities):
    "Apply a TSP algorithm to cities, plot the resulting tour, and print information."
    # Find the solution and time how long it takes
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示
    plt.rcParams['axes.unicode_minus'] = False #解决符号无法显示
    t0 = time.process_time()
    tour = algorithm(cities)
    t1 = time.process_time()
    assert valid_tour(tour, cities)
    plt.scatter([p.real for p in cities], [p.imag for p in cities],color="red")
    plt.title(algorithm.__name__)
    plot_tour(tour); plt.show()
    print("{} city tour with length {:.1f} in {:.3f} secs for {}"
          .format(len(tour), tour_length(tour), t1 - t0, algorithm.__name__))
    return tour
# TO DO: functions: algorithm,valid_tour,plot_tour  

def plot_tour(tour): 
    "Plot the cities as circles and the tour as lines between them."
    points = list(tour) 
    plot_lines( points + [points[0]] )
    start = tour[0]
    plot_lines([start], 'rs') # Mark the start city with a red square
# TO DO: functions: plot_lines      

def valid_tour(tour, cities):
    "Is tour a valid tour for these cities?"
    return set(tour) == set(cities) and len(tour) == len(cities)    
    
def plot_lines(points,style='bo-'):
    "Plot lines to connect a series of points."
    plt.plot([p.real for p in points], [p.imag for p in points],color="blue")
    plt.axis('scaled'); plt.axis('off')
    

'''============================================================================
#                           Run program
# ============================================================================='''
path = r"C:\Users\54074\Desktop\1.TSP构造式算法\test_data_tsplib"
path1 = r"D:\3_BigMaxMaterials\1_Code Depot\2_TSP\Benchmark\TSPLIB_Bench\tsplib_solution.txt"
file_name_path_list,file_name,optimul_solution = DateFilePath(path,path1)

length = []
gap = []
for i in range(12):
    data,list_data = ClearData(file_name_path_list[i])
    tour = plot_tsp(ConvexHullInsert, data)
    length.append(tour_length(tour))
    gap.append((length[i]-optimul_solution[i])/optimul_solution[i])
    print(file_name[i],":  length:", length[i],"  gap:",gap[i],'\n')

