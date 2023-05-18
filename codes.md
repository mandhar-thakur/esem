#Ford FULKERSON------------------------------------------------------------------------------
class Graph:
 
    def __init__(self, graph):
        self.graph = graph  
        self. ROW = len(graph)
 
    def BFS(self, s, t, parent):
 
        visited = [False]*(self.ROW)
        queue = []
        queue.append(s)
        visited[s] = True
        while queue:
            u = queue.pop(0)
            for ind, val in enumerate(self.graph[u]):
                if visited[ind] == False and val > 0:
                    queue.append(ind)
                    visited[ind] = True
                    parent[ind] = u
                    if ind == t:
                        return True
        return False
             
    def FordFulkerson(self, source, sink):
        parent = [-1]*(self.ROW)
        max_flow = 0
        while self.BFS(source, sink, parent) :
            path_flow = float("Inf")
            s = sink
            while(s !=  source):
                path_flow = min (path_flow, self.graph[parent[s]][s])
                s = parent[s]
 
            max_flow +=  path_flow
            v = sink
            while(v !=  source):
                u = parent[v]
                self.graph[u][v] -= path_flow
                self.graph[v][u] += path_flow
                v = parent[v]
 
        return max_flow
 
graph = [[0, 7, 0, 10, 0, 0, 0],
        [0, 0, 5, 1, 3, 0, 0],
        [0, 0, 0, 0, 0, 9, 6],
        [0, 0, 0, 0, 2, 8, 0],
        [0, 0, 4, 0, 0, 12, 0],
        [0, 0, 0, 0, 0, 0, 11],
        [0, 0, 0, 0, 0, 0, 0]]
 
g = Graph(graph)
 
source = 0; sink = 5
  
print ("The maximum possible flow is %d " % g.FordFulkerson(source, sink))


import networkx as nx

# Create a directed graph
G = nx.DiGraph()

# Add edges and capacities to the graph
G.add_edge(0, 1, capacity=7)
G.add_edge(0, 2, capacity=10)
G.add_edge(1, 2, capacity=1)
G.add_edge(1, 3, capacity=3)
G.add_edge(1, 4, capacity=5)
G.add_edge(2, 3, capacity=2)
G.add_edge(2, 5, capacity=7)
G.add_edge(3, 4, capacity=3)
G.add_edge(3, 5, capacity=2)
G.add_edge(4, 5, capacity=2)
G.add_edge(4, 6, capacity=10)
G.add_edge(5, 6, capacity=4)

# Find the maximum flow
source = 0
sink = 5
max_flow_value = nx.maximum_flow_value(G, source, sink)

print("The maximum possible flow is", max_flow_value)


Northwest------------------------------------------------------------------------------------

grid = [[11,13,17,14],
        [16,18,19,10],
        [21,24,13,10]]
supply = [250,300,400]
demand = [200,225,275,250]

ans= 0 
startr=0
startc = 0

while(startr != len(grid) and startc!= len(grid[0])):
  if(supply[startr] <= demand[startc]):
    ans += supply[startr]*grid[startr][startc]
    demand[startc] -= supply[startr]
    startr += 1
  else:
    ans += demand[startc]*grid[startr][startc]
    supply[startr] -= demand[startc]
    startc += 1
print('total cost of transportation is',ans)


----------------------------------------Group Replacement---------------------------------------------------------------------------

n0=1000
week=5
cp = [10,25,50,80,100]
non_cp=[cp[0]/100]
for i in range(1,week):
  non_cp.append((cp[i]-cp[i-1])/100)
for i in range(week):
  print(f"non cummulative percentages are {non_cp[i]}")
print(f"------------------------------------------------------------")
n1 = n0*non_cp[0]
n2 = n0*non_cp[1]+n1*non_cp[0]
n3 = n0*non_cp[2]+n1*non_cp[1]+n2*non_cp[0]
n4 = n0*non_cp[3]+n1*non_cp[2]+n2*non_cp[1]+n3*non_cp[0]
n5 = n0*non_cp[4]+n1*non_cp[3]+n2*non_cp[2]+n3*non_cp[1]+n4*non_cp[0]
n = [n1,n2,n3,n4,n5]
# for number of replacement

#n=[n0*non_cp[0]]
#for i in range(1,week):
 # ninew = n0*non_cp[i]
 # for j in range(i):
  #  ninew += n[j]*non_cp[i-j-1]
 # n.append(ninew)
print(f"number of replacement per week are{n}")
expected_life = 0
for i in range(1,week):
  expected_life += round((i+1)*(non_cp[i]),1)
print(f"expected life = {expected_life}")
avg_fail = round(n0/expected_life,0)
print(f"average failure is {avg_fail}")
individualreplacementcost = avg_fail*2
print(f"individual replacement cost is {individualreplacementcost}")
print(f"------------------------------------------------------------")
ind_repl=0
tc=0
ac=0
min_avg_c=float('inf')
for i in range(week):
  ind_repl += round(n[i],0)
  tc = (ind_repl*2)+1000*.5
  ac = tc/(i+1)
  if (ac < min_avg_c):
    min_avg_c = ac
print(f"min_avg_cost is {min_avg_c}")

if (min_avg_c < individualreplacementcost):
  print("will be group")
else:
  print("individual replacement will be preferred")

-----------------------------------QUEUE THEORY------------------------------------------------------------------

#21CBS1057
arrival_rate=int(input("enter the arrival rate - "))
service_rate=int(input("enter the service rate - "))
arrival_rate= arrival_rate/(60*24)
service_rate=1/service_rate
traffic_intensity=arrival_rate/service_rate
print("traffic intenity is ",traffic_intensity)
queue_size=traffic_intensity/(1-traffic_intensity)
print("queue size is ",queue_size)
probability=round(pow(traffic_intensity,10),2)
print("probability that queue size exceeds 10 is",probability)

print("input increases to 33 per day")

arrival_rate= 33/(60*24)
traffic_intensity=arrival_rate/service_rate
print("traffic intenity is ", traffic_intensity)
queue_size=traffic_intensity/(1-traffic_intensity)
print("queue size is ",queue_size)
probability=round(pow(traffic_intensity,10),2)
print("probability that queue size exceeds 10 is",probability)

------------------------------------------EOQ----------------------------------------------------------------

#21CBS10
demand=float(input("enter the demand "))
cost = float(input("enter the cost "))
cost_per_order=float(input("enter the oredering cost "))
cost_of_holding=float(input("enter the holding cost "))#1.40 for 5 percent interest1

EOQ=round(((2*demand*cost_per_order)/(cost_of_holding))**0.5,2)

print("the EOQ for this is",EOQ)

print("Total Number of orders are ",round(demand/EOQ,2))

Total_annual_inventory_expense = round((demand/EOQ)*cost_per_order + EOQ/2 * cost_of_holding,2)

print("Total annual inventory expenses are",Total_annual_inventory_expense)

Total_Inventory_cost = demand*cost+Total_annual_inventory_expense

print(f"totaal inventory cost is {Total_Inventory_cost}")

--------------------------------------------------------------

initial_cost = 12200
running_costs = [200, 500, 800, 1200, 1800, 2500, 3200, 4000]
scrap_value = 200
depreciation_cost = 0
cumulative_cost = 0
total_cost = 0
average_cost = 0
min_avg_cost = float('inf')
optimal_replacement_time = 0

n = len(running_costs)
i = 1
for i in range (n):
    cumulative_cost += running_costs[i]
    print(f"The cumulative running cost is {cumulative_cost}.")
    depreciation_cost = initial_cost - scrap_value
    print(f"Depreciation cost: {depreciation_cost}")
    total_cost = cumulative_cost + depreciation_cost
    print(f"total cost: {total_cost}")
    average_cost = total_cost / (i+1)
    print(f"Average cost for year {i+1}: {average_cost}")
    if average_cost < min_avg_cost:
        min_avg_cost = average_cost
        optimal_replacement_time = i+1
print(f"The optimal replacement time is year {optimal_replacement_time} with an average annual cost of {min_avg_cost}.")

---------------------------------------------------------------------------

ini_cost = 12200
run_cost = [200,500,800,1200,1800,2500,3200,4000]
scrap = 200
depri_cost = 0
crc=0
tc=0
rc=0
min_avg_cost=float('inf')
opt_replac_time=0
n = len(run_cost) 
for i in range (n):
  crc += run_cost[i]
  print("crc is ",crc)
  depri_cost = ini_cost-scrap
  print("depriciation cost is",depri_cost)
  tc = depri_cost+crc
  print("total cost is ",tc)
  ac = tc/(i+1)
  print("average cost is ", ac)
  if ac < min_avg_cost:
    min_avg_cost = ac
    opt_replac_time=i+1
print(f"The optimal replacement time is year {opt_replac_time} with an average annual cost of {min_avg_cost}.")

---------------------------------------------------------------------------------------------
ic = 60000
running_costs = [0]*5
maintain_cost = [4000, 4270, 4880, 5700, 6800]
labour_cost = [14000,16000, 18000, 21000, 25000]
sv = [42000, 30000, 20400, 14400, 9650]
dc = 0 
crc=0
tc=0
rc=0
min_avg_cost=float('inf')
opt_replac_time=0
n=len(scrap_value)
i = 1
for i in range (n):
  running_costs= maintain_cost[i]+labour_cost[i]
  print("running cost = ",running_costs)
  crc += running_costs
  print("crc is ",crc)
  dc = ic-sv[i]
  print("depriciation cost is ",dc)
  tc = crc+dc
  print("total cost is ",tc)
  ac = tc/(i+1)
  print("verage cost is ",ac)
  if ac < min_avg_cost:
    min_avg_cost = ac
    opt_replac_time= i+1
  print("----------------------------")
print(f"The optimal replacement time is year {opt_replac_time} with an average annual cost of {min_avg_cost}.")

----------------------------------------------------------------------------------------

import math
demand = 1500
ordering_cost = 500
holding_cost = 0.15
shortage_cost= 20
production = 3000

EOQ=math.sqrt(((2 * demand * ordering_cost) / holding_cost)*(production/(production-demand))*((holding_cost+shortage_cost)/shortage_cost))
print(f"the optimal order quantity is:{EOQ}")

Q = EOQ*(1-(demand/production))*holding_cost/(holding_cost+shortage_cost)
print(f"the optimal supply during shortage is:{Q}")

production_time = EOQ/production
print(f"the production timeis:{production_time}")

production_cycle_time = EOQ/(production-demand)
print(f"the production cycle time is:{production_cycle_time}")

----------------------------------------------------------------------------------------

import math
demand = 200
ordering_cost = 50
holding_cost = 2
shortage_cost= 10

EOQ=math.sqrt(((2 * demand * ordering_cost) / holding_cost)*((holding_cost+shortage_cost)/shortage_cost))
print(f"the optimal order quantity is:{EOQ}")

reorder_cycle = EOQ/demand
print(f"the optimal reorder cycle is:{reorder_cycle}")

TVC = EOQ*holding_cost
print(f"the total variable inventory cost is:{TVC}")

------------------------------------------------LEAST COST--------------------------------------------

import numpy as np

def least_cost_method(supply, demand, costs):
    # Create a copy of the original cost matrix
    modified_costs = np.copy(costs)
    
    # Initialize the allocation matrix with zeros
    allocation = np.zeros((len(supply), len(demand)))
    
    # Iterate until all supply and demand are fulfilled
    while np.sum(supply) > 0 and np.sum(demand) > 0:
        # Find the indices of the minimum cost cell
        min_cost_indices = np.unravel_index(np.argmin(modified_costs), modified_costs.shape)
        i, j = min_cost_indices
        
        # Calculate the quantity to be allocated
        quantity = min(supply[i], demand[j])
        
        # Update the allocation matrix and deduct the supply and demand
        allocation[i, j] = quantity
        supply[i] -= quantity
        demand[j] -= quantity
        
        # Replace the allocated cell with a large value
        modified_costs[i, j] = np.max(costs) + 1
        
    return allocation

# Example usage
supply = [250, 300, 400]
demand = [200, 225, 275, 250]
costs = np.array([[11, 13, 17, 14],
                  [16, 18, 14, 10],
                  [21, 24, 13, 10]])

allocation = least_cost_method(supply, demand, costs)

# Calculate the total cost
total_cost = np.sum(allocation * costs)
print("Total Cost:", total_cost)

-------------------------------------------hung----------------------------------------
import numpy as np

from scipy.optimize import linear_sum_assignment

#Define the distance matrix

distances = np.array([[0,10,20,30,40], 
                      [10,0,15,25,35],
                      [20,15,0,14,24],
                      [30,25,14,0,18],
                      [40,35,24,18,0]])

#Find the minimum cost using the Hungarian method

row_ind, col_ind = linear_sum_assignment (distances)

#Print the optimal tour

tour = []

for i , j in zip(row_ind, col_ind):
  tour.append(i)
tour.append(0) 
#Return to the starting point

print("Optimal tour:",tour)

----------------------------------simplex------------------------------------

from scipy.optimize import linprog
c = [3,2]
A=[[-1,2],[3,2],[1,-1]]
b=[4,14,3]
bnd = [(0,float('inf')),(0,float('inf'))]
res = linprog(c, A_ub=A, b_ub=b, bounds=bnd, method='simplex')
print(res)
