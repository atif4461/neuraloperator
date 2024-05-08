import torch
from mpi4py import MPI
import numpy as np

import csv
import h5py

#N = 1220
#nx = 256
#ny = 256
#timesteps = 320
#
##num_processes = MPI.COMM_WORLD.size
##print("working with ", num_processes, " MPI processes")
#rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
#
#with h5py.File('dpsl_data_N1220_alpha1.h5', 'a', driver='mpio', libver='latest', comm=MPI.COMM_WORLD) as f:
#    # create empty data set
#    data = f.create_dataset('vel2d', shape=(N,timesteps,nx,ny,2), dtype='f', chunks=(1,1,nx,ny,1))#, compression='gzip', compression_opts=9)
#    #dset = f["vel2d"]
#    
#    #with dset.collective:
#    
#    for simindx in range(0,N,1):
#        if simindx % num_processes == rank:
#            # add chunk of rows
#            print("reading data for simulation ", simindx, " in rank ", rank, flush=True)
#        
#            for j in range(0,timesteps,1):
#    
#                #print ("time ", j*20, flush=True)
#                i = 0
#                flat_ux = torch.zeros(nx*ny) 
#                flat_uy = torch.zeros(nx*ny) 
#    
#                with open("/work/atif/fourier_neural_operator/data_generation/D2Q9/results"+str(simindx)+"/velocity"+str(j*20)+".dat") as fl:
#                    reader = csv.reader(fl, delimiter=" ")
#                    for line in reader:
#                        flat_ux[i] = float(line[0]) #column number of ux
#                        flat_uy[i] = float(line[1]) #column number of uy
#                        i=i+1
#    
#                data[simindx,j,:,:,0] = flat_ux.reshape(nx,ny)  #ux      
#                data[simindx,j,:,:,1] = flat_uy.reshape(nx,ny)  #uy      
#                del flat_ux, flat_uy
#
#f.close() 
#MPI.Finalize()
##time ~/.conda/envs/dat_to_h5/bin/mpiexec -n 10 python3 dat_to_h5_mpi.py


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def get_rank_coord(r):
    rank_coord = ([0,0,0])
    if r == 1:
         rank_coord = ([1,0,0])
    if r == 2:
         rank_coord = ([2,0,0])
    if r == 3:
         rank_coord = ([3,0,0])
    if r == 4:
         rank_coord = ([0,1,0])
    if r == 5:
         rank_coord = ([1,1,0])
    if r == 6:
         rank_coord = ([2,1,0])
    if r == 7:
         rank_coord = ([3,1,0])
    if r == 8:
         rank_coord = ([0,0,1])
    if r == 9:
         rank_coord = ([1,0,1])
    if r == 10:
         rank_coord = ([2,0,1])
    if r == 11:
         rank_coord = ([3,0,1])
    if r == 12:
         rank_coord = ([0,1,1])
    if r == 13:
         rank_coord = ([1,1,1])
    if r == 14:
         rank_coord = ([2,1,1])
    if r == 15:
         rank_coord = ([3,1,1])
    
    return rank_coord


timesteps = np.array([1047,
1052,
1056,
1061,
1066,
1071,
1076,
1080,
1085,
1090,
1094,
1099,
1103,
1108,
1112,
1117,
1121,
1126,
1131,
1135,
1140,
1145,
1150,
1154,
1159,
1164,
1169,
1173,
1178,
1183,
1187,
1192,
1196,
1200,
1205,
1209,
1213,
1218,
1222,
1226,
1231,
1235,
1239,
1244,
1248,
1253,
1257,
1262,
1266,
1271,
1275,
1280,
1285,
1289,
1294,
1298,
1303,
1307,
1312,
1316,
1321,
1325,
1330,
1334,
1339,
1343,
1348,
1352,
1357,
1361,
1366,
1370,
1375,
1379,
1384,
1388,
1393,
1398,
1402,
1407,
1411,
1415,
1420,
1424,
1429,
1433,
1437,
1442,
1446,
1451,
1455,
1459,
1464,
1468,
1472,
1477,
1481,
1486,
1490,
1495,
1499,
1504,
1508,
1512,
1517,
1521,
1526,
1530,
1534,
1539,
1543,
1547,
1552,
1556,
1560,
1565,
1569,
1573,
1578,
1582,
1586,
1591,
1595,
1600,
1605,
1610,
1615,
1619,
1624,
1629,
1634,
1639,
1644,
1648,
1653,
1657,
1662,
1666,
1670,
1675,
1679,
1683,
1688,
1692,
1696,
1701,
1705,
1710,
1714,
1719,
1724,
1728,
1733,
1738,
1743,
1747,
1752,
1757,
1762,
1766,
1771,
1776,
1781,
1785,
1790,
1795,
1800,
1804,
1809,
1814,
1818,
1823,
1827,
1832,
1836,
1841,
1846,
1850,
1855,
1860,
1864,
1869,
1873,
1878,
1883,
1888,
1892,
1897,
1902,
1907,
1912,
1917,
1922,
1927,
1932,
1937,
1941,
1946,
1951,
1956,
1961,
1965,
1970,
1974,
1979,
1984,
1988,
1992,
1997,
2001,
2006,
2010,
2015,
2019,
2024,
2028,
2032,
2037,
2041,
2046,
2050,
2055,
2059,
2064,
2068,
2073,
2077,
2082,
2087,
2091,
2096,
2101,
2105,
2110,
2115,
2119,
2124,
2129,
2133,
2138,
2143,
2147,
2152,
2157,
2161,
2166,
2171,
2176,
2181,
2185,
2190,
2195,
2200
])

Nx = 384
Ny = 384
Nz = 384
nx = 96
ny = 192
nz = 192

def index_to_ijk(index):

    nx_ = nx + 8
    ny_ = ny + 8
    k = np.floor(index/(nx_*ny_)).astype(int)
    yindex = index % (nx_*ny_)

    j = np.floor(yindex/nx_).astype(int)
    i = yindex % nx_

    return i,j,k

num_processes = MPI.COMM_WORLD.size
print("working with ", num_processes, " MPI processes")
rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)

s = 0
with h5py.File('ico_turb_3d_384.h5', 'a', driver='mpio', libver='latest', comm=MPI.COMM_WORLD) as f:
    # create empty data set
    data = f.create_dataset('vel3d', shape=(timesteps.shape[0],3,Nx,Ny,Nz), dtype='f', chunks=(1,1,1,Ny,Nz))#, compression='gzip', compression_opts=9)

    for t in timesteps:
        for rxank in range(0,1,1):
            rank_coord = get_rank_coord(rank)
            print("reading data in rank ", rank, " at time step ", t, flush=True)
    
            flat_ux = torch.zeros(nx*ny*nz) 
            flat_uy = torch.zeros(nx*ny*nz) 
            flat_uz = torch.zeros(nx*ny*nz) 
            
            with open("/work/atif/PR_DNS_base/DNS/output-dataset1-384-16/vtk/vtk.ts000"+f'{t:04d}'+"-nd00"+f'{rank:02d}'+"/VELOCITY.vtk") as fl:
                reader = csv.reader(fl, delimiter=" ")
               
                skip = 0
                index = 0
                ind = 0
                for line in reader:
                    
                    if skip < 9 :
                        skip = skip + 1
                        continue
                    
                    i,j,k = index_to_ijk(index)
 
                    if (i >= 4 and i <= 99 and j >= 4 and j <= 195 and  k >= 4 and k <= 195):
                        flat_ux[ind] = float(line[0]) #column number of ux
                        flat_uy[ind] = float(line[1]) #column number of uy
                        flat_uz[ind] = float(line[2]) #column number of uy
                        ind = ind + 1
                        #print(index, "    ", i, j, k, "     ", I, J, K, "    ", line[0], line[1], line[2])
                    
                    index = index + 1
                    if index % 1000000 == 0:
                        print(rank, "rank and index ", index, "passed", flush=True)
            
            Ib = rank_coord[0]*nx 
            Jb = rank_coord[1]*ny 
            Kb = rank_coord[2]*nz 
            Ie = Ib + nx
            Je = Jb + ny
            Ke = Kb + nz
            data[s,0,Ib:Ie,Jb:Je,Kb:Ke] = flat_ux.reshape(nx,ny,nz)  #ux      
            data[s,1,Ib:Ie,Jb:Je,Kb:Ke] = flat_uy.reshape(nx,ny,nz)  #uy      
            data[s,2,Ib:Ie,Jb:Je,Kb:Ke] = flat_uz.reshape(nx,ny,nz)  #uz      
            del flat_ux, flat_uy, flat_uz
            print(rank, "4rank ", fl, " passed", flush=True)
        s = s + 1


f.close() 
MPI.Finalize()
##time ~/.conda/envs/dat_to_h5/bin/mpiexec -n 10 python3 dat_to_h5_mpi.py


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # #



#timesteps = np.array([0])
#
#Nx = 64
#Ny = 64
#Nz = 64
#nx = np.floor(Nx/4).astype(int)
#ny = np.floor(Ny/2).astype(int)
#nz = np.floor(Nz/2).astype(int)
#
#def index_to_ijk(index):
#
#    nx_ = nx + 8
#    ny_ = ny + 8
#    k = np.floor(index/(nx_*ny_)).astype(int)
#    yindex = index % (nx_*ny_)
#
#    j = np.floor(yindex/nx_).astype(int)
#    i = yindex % nx_
#
#    return i,j,k
#
#num_processes = MPI.COMM_WORLD.size
#print("working with ", num_processes, " MPI processes")
#rank = MPI.COMM_WORLD.rank  # The process ID (integer 0-3 for 4-process run)
#
#s = 0
#with h5py.File('ico_turb_3d_64.h5', 'a', driver='mpio', libver='latest', comm=MPI.COMM_WORLD) as f:
#    # create empty data set
#    data = f.create_dataset('vel3d', shape=(timesteps.shape[0],3,Nx,Ny,Nz), dtype='f', chunks=(1,1,1,Ny,Nz))#, compression='gzip', compression_opts=9)
#
#    for t in timesteps:
#        for rxank in range(0,1,1):
#            rank_coord = get_rank_coord(rank)
#            print("reading data in rank ", rank, " at time step ", t, flush=True)
#    
#            flat_ux = torch.zeros(nx*ny*nz) 
#            flat_uy = torch.zeros(nx*ny*nz) 
#            flat_uz = torch.zeros(nx*ny*nz) 
#            
#            with open("/work/atif/PR_DNS_base/DNS/output-dataset-64-16/vtk/vtk.ts000"+f'{t:04d}'+"-nd00"+f'{rank:02d}'+"/VELOCITY.vtk") as fl:
#                reader = csv.reader(fl, delimiter=" ")
#               
#                skip = 0
#                index = 0
#                ind = 0
#                for line in reader:
#                    
#                    if skip < 9 :
#                        skip = skip + 1
#                        continue
#                    else : 
#                        i,j,k = index_to_ijk(index)
# 
#                        if (i >= 4 and i <= 19 and j >= 4 and j <= 35 and  k >= 4 and k <= 35):
#                            flat_ux[ind] = float(line[0]) #column number of ux
#                            flat_uy[ind] = float(line[1]) #column number of uy
#                            flat_uz[ind] = float(line[2]) #column number of uy
#                            ind = ind + 1
#                            #print(index, "    ", i, j, k, "     ", I, J, K, "    ", line[0], line[1], line[2])
#                        
#                        index = index + 1
#                        #if index % 1000 == 0:
#                        #    print(rank, "rank and index ", index, "passed", flush=True)
#            
#            Ib = rank_coord[0]*nx 
#            Jb = rank_coord[1]*ny 
#            Kb = rank_coord[2]*nz 
#            Ie = Ib + nx
#            Je = Jb + ny
#            Ke = Kb + nz
#            print(rank, "0rank ", fl, " passed", flush=True)
#            data[s,0,Ib:Ie,Jb:Je,Kb:Ke] = flat_ux.reshape(nx,ny,nz)  #ux      
#            print(rank, "1rank ", fl, " passed", flush=True)
#            data[s,1,Ib:Ie,Jb:Je,Kb:Ke] = flat_uy.reshape(nx,ny,nz)  #uy      
#            print(rank, "2rank ", fl, " passed", flush=True)
#            data[s,2,Ib:Ie,Jb:Je,Kb:Ke] = flat_uz.reshape(nx,ny,nz)  #uz      
#            print(rank, "3rank ", fl, " passed", flush=True)
#            del flat_ux, flat_uy, flat_uz
#            print(rank, "4rank ", fl, " passed", flush=True)
#        s = s + 1
#
#
#f.close() 
#MPI.Finalize()
##time ~/.conda/envs/dat_to_h5/bin/mpiexec -n 10 python3 dat_to_h5_mpi.py


