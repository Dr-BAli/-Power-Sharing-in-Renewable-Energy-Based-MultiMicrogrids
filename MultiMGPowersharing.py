import numpy as np
import cvxpy as cp
import pandas as pd

# Number of microgrids and hours
nMicrogrids = 5
nHours = 24
dt = 1  # Time step in hours

# Data for 5 Microgrids
Bprice = np.array([5000, 6000, 6000, 4000, 3000])  # Battery price in Euros
Bcycle = np.array([3000, 3000, 3000, 3000, 3000])  # Battery cycle life in cycles
NOMb = np.array([35000, 20000, 25000, 30000, 25000])  # Nominal Battery Capacity in Wh
NOMbInit = np.array([27000, 15000, 20000, 25000, 18000])  # Initial Nominal Battery Capacity in Wh
SOCminB = np.array([0.15, 0.2, 0.2, 0.1, 0.2])  # Minimum State of Charge (SOC)
SOCmaxB = np.array([0.95, 0.85, 0.9, 0.9, 0.9])  # Maximum State of Charge (SOC)
ec = np.array([0.95, 0.97, 0.95, 0.97, 0.95])  # Battery Charging Efficiency
ed = np.array([0.95, 0.97, 0.95, 0.97, 0.95])  # Battery Discharging Efficiency
PbChmax = np.array([25000, 14000, 18000, 23000, 16000])  # Max Battery Charging Power in W
PbDischmax = np.array([25000, 14000, 18000, 23000, 16000])  # Max Battery Discharging Power in W

# Example forecast data (replace with actual forecast data)
Ppv = np.random.rand(nHours, nMicrogrids) * 10000
Pwind = np.random.rand(nHours, nMicrogrids) * 5000
LoadDemand = np.random.rand(nHours, nMicrogrids) * 20000
TOU = np.random.rand(nHours, nMicrogrids) * 0.2

# Base Case: Self-ESS Scheduling for Each Microgrid
TotalCostBase = np.zeros(nMicrogrids)
Pnet_flow_base = np.zeros((nHours, nMicrogrids))

for i in range(nMicrogrids):
    # Variables
    PbCh = cp.Variable(nHours, nonneg=True)
    PbDisch = cp.Variable(nHours, nonneg=True)
    SOCb = cp.Variable(nHours)
    u = cp.Variable(nHours, boolean=True)
    
    # Constraints
    constraints = [
        PbCh <= PbChmax[i],
        PbDisch <= PbDischmax[i],
        SOCb[0] == NOMbInit[i] / NOMb[i],
        SOCb <= SOCmaxB[i],
        SOCb >= SOCminB[i],
        SOCb[-1] == SOCb[0]
    ]
    
    for t in range(nHours - 1):
        constraints.append(SOCb[t + 1] == SOCb[t] + (PbCh[t] * ec[i] - PbDisch[t] / ed[i]) / NOMb[i] * dt)
    
    for t in range(nHours):
        constraints.append(PbCh[t] <= u[t] * PbChmax[i])
        constraints.append(PbDisch[t] <= (1 - u[t]) * PbDischmax[i])
    
    Pnet_flow = LoadDemand[:, i] - Ppv[:, i] - Pwind[:, i] + PbCh - PbDisch
    
    # Objective
    batteryCost = cp.sum((PbDisch / ed[i] + PbCh * ec[i]) / (2 * SOCmaxB[i] - SOCminB[i]*NOMb[i]) * (Bprice[i] / Bcycle[i]))
    gridCost = cp.sum(TOU[:, i] * Pnet_flow * dt)
    
    objective = cp.Minimize(gridCost + batteryCost)
    
    # Problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.CPLEX)
    
    if prob.status not in ["infeasible", "unbounded"]:
        Time = np.arange(1, nHours + 1)
        Pbatcharge = PbCh.value
        Pbatdischarge = PbDisch.value
        BatterySOC = SOCb.value
        Pgrid = LoadDemand[:, i] - Ppv[:, i] - Pwind[:, i] + Pbatcharge - Pbatdischarge
        
        resultsTable = pd.DataFrame({
            "Time": Time,
            "Ppv": Ppv[:, i],
            "Pwind": Pwind[:, i],
            "LoadDemand": LoadDemand[:, i],
            "Pgrid": Pgrid,
            "Pbatcharge": Pbatcharge,
            "Pbatdischarge": Pbatdischarge,
            "BatterySOC": BatterySOC
        })
        
        print(f'Results for Microgrid {i+1}')
        print(resultsTable)
        
        Pnet_flow_base[:, i] = Pgrid
        TotalCostBase[i] = prob.value
    else:
        print(f'No feasible solution found for microgrid {i+1}.')

# Optimization of Power Exchange Between Microgrids
Pexport = cp.Variable((nHours, nMicrogrids, nMicrogrids), nonneg=True)
Pimport = cp.Variable((nHours, nMicrogrids, nMicrogrids), nonneg=True)
PbCh = cp.Variable((nHours, nMicrogrids), nonneg=True)
PbDisch = cp.Variable((nHours, nMicrogrids), nonneg=True)
SOCb = cp.Variable((nHours, nMicrogrids))
u = cp.Variable((nHours, nMicrogrids), boolean=True)
isExporting = cp.Variable((nHours, nMicrogrids, nMicrogrids), boolean=True)
isImporting = cp.Variable((nHours, nMicrogrids, nMicrogrids), boolean=True)

Pnet_flow = LoadDemand - Ppv - Pwind + PbCh - PbDisch + cp.sum(Pimport, axis=2) - cp.sum(Pexport, axis=2)
TotalCost = cp.sum(cp.sum(TOU * Pnet_flow * dt))

constraints = [
    SOCb <= SOCmaxB,
    SOCb >= SOCminB
]

for i in range(nMicrogrids):
    constraints.append(SOCb[0, i] == NOMbInit[i] / NOMb[i])
    constraints.append(SOCb[-1, i] == SOCb[0, i])
    
    for t in range(nHours - 1):
        constraints.append(SOCb[t + 1, i] == SOCb[t, i] + (PbCh[t, i] * ec[i] - PbDisch[t, i] / ed[i]) / NOMb[i] * dt)
    
    for t in range(nHours):
        constraints.append(PbCh[t, i] <= u[t, i] * PbChmax[i])
        constraints.append(PbDisch[t, i] <= (1 - u[t, i]) * PbDischmax[i])
    
    for j in range(nMicrogrids):
        if i != j:
            for t in range(nHours):
                constraints.append(isExporting[t, i, j] + isImporting[t, j, i] <= 1)
                constraints.append(Pexport[t, i, j] <= isExporting[t, i, j] * max(PbDischmax))
                constraints.append(Pimport[t, j, i] <= isImporting[t, j, i] * max(PbDischmax))

for t in range(nHours):
    constraints.append(cp.sum(cp.sum(Pexport[t, :, :])) == cp.sum(cp.sum(Pimport[t, :, :])))

objective = cp.Minimize(TotalCost)
prob = cp.Problem(objective, constraints)
prob.solve(solver=cp.CPLEX)

if prob.status not in ["infeasible", "unbounded"]:
    for i in range(nMicrogrids):
        print(f'Microgrid {i+1}')
        print('Time\tPexport\tPimport')
        for t in range(nHours):
            print(f'{t+1}\t{np.sum(Pexport.value[t, i, :]):.2f}\t{np.sum(Pimport.value[t, :, i]):.2f}')
        print()

    print('Total Cost:')
    print(prob.value)
else:
    print('No feasible solution found for power exchange optimization.')
