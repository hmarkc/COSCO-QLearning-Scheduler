from .Scheduler import *
from .BaGTI.train import *
import numpy as np 

class QLearningScheduler(Scheduler):
    def __init__(self):
        super().__init__()
        self.model = None
        self.data_type = None
        self.hosts = None
        self.max_container_ips = None
    
    # select container to be sheduled or migrated
    def selection(self):
        selectedContainerIDs = []
        for hostId, host in enumerate(self.env.hostlist):
            if host.getCPU() > 70:
                containerIds = self.env.getContainersOfHost(hostId)
                if containerIds:
                    containerIPS = [self.env.containerlist[id].getBaseIPS() for id in containerIds]
                    selectedContainerIDs.append(containerIds[np.argmax(containerIPS)])
        return selectedContainerIDs
    
    # place selected container to a host
    def placement(self, containerIDs):
        decision = []
        for id in containerIDs:
            scores = [self.env.stats.runSimpleSimulation([(id, hostId)])[0] for hostId, _ in enumerate(self.env.hostlist)]
            decision.append((id, np.argmin(scores)))
        return decision