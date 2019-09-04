from Kitsune import Kitsune
import numpy as np
import time

##############################################################################
# Kitsune a lightweight online network intrusion detection system based on an ensemble of autoencoders (kitNET).
# For more information and citation, please see our NDSS'18 paper: Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection

# This script demonstrates Kitsune's ability to incrementally learn, and detect anomalies in recorded a pcap of the Mirai Malware.
# The demo involves an m-by-n dataset with n=115 dimensions (features), and m=100,000 observations.
# Each observation is a snapshot of the network's state in terms of incremental damped statistics (see the NDSS paper for more details)

#The runtimes presented in the paper, are based on the C++ implimentation (roughly 100x faster than the python implimentation)
###################  Last Tested with Anaconda 3.6.3   #######################

# Load Mirai pcap (a recording of the Mirai botnet malware being activated)
# The first 70,000 observations are clean...
# print("Unzipping Sample Capture...")
# import zipfile
# with zipfile.ZipFile("mirai.zip","r") as zip_ref:
#     zip_ref.extractall() #解压缩


# File location
path = "100000_packets_mirai.tsv" #the pcap, pcapng, or tsv file to process.
packet_limit = np.Inf #the number of packets to process
labels_path = "100000_labels_mirai.tsv"

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)
packet_num = 100000
benign_num = 70000
mal_num = 30000

# Build Kitsune
K = Kitsune(path,packet_limit,maxAE,FMgrace,ADgrace) #读了文件第一行字段名称

print("Running Kitsune:")
RMSEs = []
RMSEs_ensembleLayer = []
i = 0
execute_cnt = 0
autoencoders_num = 0
start = time.time()
# Here we process (train/execute) each individual packet.
# In this way, each observation is discarded after performing process() method.
while True:
    i+=1
    if i % 1000 == 0:
        print(i)
    rmse, rmses_ensembleLayer = K.proc_next_packet()
    #print(rmses_ensembleLayer)
    if rmse == -1:
        break
    RMSEs.append(rmse)
    if i > FMgrace + ADgrace+1: #可能需要 +1 因为不知道为啥执行阶段第一个样本rmses_ensembleLayer是没有的
        RMSEs_ensembleLayer.append(rmses_ensembleLayer)
RMSEs_ensembleLayer = np.array(RMSEs_ensembleLayer).T
autoencoders_num = RMSEs_ensembleLayer.shape[0]
execute_cnt = RMSEs_ensembleLayer.shape[1]
stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))
threshold = max(RMSEs[0: benign_num])
labels = []
for line in open(labels_path): 
    if line.find('0') != -1:
        labels.append(0)
    elif line.find('1') != -1:
        labels.append(1)
positive_sum = mal_num
positive_num = 0
negative_sum = benign_num - (FMgrace + ADgrace+1)
negative_num = 0
for i in range(FMgrace + ADgrace+1, packet_num):
    if RMSEs[i] > threshold and labels[i] == 1:
        positive_num += 1
    if RMSEs[i] > threshold and labels[i] == 0:
        negative_num += 1
Test_TPR = positive_num / positive_sum
Test_FPR = negative_num / negative_sum
print('threshold: {0}'.format(threshold))
print('Test_TPR: {0}, Test_FPR: {1}'.format(Test_TPR, Test_FPR)) #训练样本全是良性样本
        
# Here we demonstrate how one can fit the RMSE scores to a log-normal distribution (useful for finding/setting a cutoff threshold \phi)
from scipy.stats import norm
benignSample = np.log(RMSEs[FMgrace+ADgrace+1:packet_num]) #10万个数据包，benignSample并不是字面上的意思，而是全部样本
logProbs = norm.logsf(np.log(RMSEs), np.mean(benignSample), np.std(benignSample)) #返回值: log of (1 - log(RMSEs)概率密度函数的积分)

# plot the RMSE anomaly scores
print("Plotting results")
from matplotlib import pyplot as plt
from matplotlib import cm
plt.figure(figsize=(10,5))
fig = plt.scatter(range(FMgrace+ADgrace+1,len(RMSEs)),RMSEs[FMgrace+ADgrace+1:],s=0.1,c=logProbs[FMgrace+ADgrace+1:],cmap='RdYlGn')
plt.yscale("log") #改变坐标轴的刻度为log
plt.title("Anomaly Scores from Kitsune's Execution Phase")
plt.ylabel("RMSE (log scaled)")
plt.xlabel("Time elapsed [min]")
#plt.annotate('Mirai C&C channel opened [Telnet]', xy=(121662,RMSEs[121662]), xytext=(151662,1),arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.annotate('Mirai Bot Activated\nMirai scans network\nfor vulnerable devices', xy=(122662,10), xytext=(122662,150),arrowprops=dict(facecolor='black', shrink=0.05),)
#plt.annotate('Mirai Bot launches DoS attack', xy=(370000,100), xytext=(390000,1000),arrowprops=dict(facecolor='black', shrink=0.05),)
figbar=plt.colorbar()
figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
#plt.savefig("./figures/mirai_kitsune.png")
plt.savefig("./figures/mirai_kitsune.pdf")
plt.savefig("./figures/mirai_kitsune.eps")
plt.show()

# plot the RMSEs_ensembleLayer anomaly scores
for i in range(autoencoders_num):
    allSample = np.log(RMSEs_ensembleLayer[i])
    logProb = norm.logsf(np.log(RMSEs_ensembleLayer[i]), np.mean(allSample), np.std(allSample))
    plt.figure(figsize=(10,5))
    fig = plt.scatter(range(0,len(RMSEs_ensembleLayer[i])),RMSEs_ensembleLayer[i],s=0.1,c=logProb,cmap='RdYlGn')
    plt.yscale("log") #改变坐标轴的刻度为log
    plt.title("Anomaly Scores from Ensemble Layer's Autoencoder"+ str(i))
    plt.ylabel("RMSE (log scaled)")
    plt.xlabel("Time elapsed [min]")
    figbar=plt.colorbar()
    figbar.ax.set_ylabel('Log Probability\n ', rotation=270)
    print(i)
    #figname = "./figures/mirai_" + str(i) + ".png"
    figname = "./figures/mirai_" + str(i) + ".pdf"
    plt.savefig(figname)
    figname = "./figures/mirai_" + str(i) + ".eps"
    plt.savefig(figname)
    plt.show()