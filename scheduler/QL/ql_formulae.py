# network delay of container
def network_delay_of_container(env, container_id):
    container = env.getContainerByID(container_id)
    host = env.getHostByID(container.hostid)
    return host.latency

# computation delay of container
def computation_delay_of_container(env, container_id):
    container = env.getContainerByID(container_id)
    return (container.ipsmodel.totalInstructions - container.ipsmodel.completedInstructions) / container.getApparentIPS() if container.getApparentIPS() else 0

# total delay of container
def total_delay_of_container(env, container_id, k=1):
    return network_delay_of_container(env, container_id) + k * computation_delay_of_container(env, container_id)

# total delay 
def total_delay(env, k=1):
    total = 0
    for container in env.containerlist:
        if container is None:
            continue
        total += total_delay_of_container(env, container.id, k)
    return total

# migration cost of container
def migration_cost_of_container(env, container_id, new_host_id, allocBw):
    migrationTime = 0
    container = env.getContainerByID(container_id)
    if container.hostid != new_host_id:
        migrationTime += container.getContainerSize() / allocBw
        if new_host_id != -1:
            migrationTime += abs(env.hostlist[container.hostid].latency - env.hostlist[new_host_id].latency)
    return migrationTime

# total migration cost
def total_migration_cost(env, decisions):
    total = 0
    routerBwToEach = env.totalbw / len(decisions) if decisions else env.totalbw
    for container_id, new_host_id in decisions:
        numberAllocToHost = len(env.scheduler.getMigrationToHost(new_host_id, decisions))
        allocBw = min(env.getHostByID(new_host_id).bwCap.downlink / numberAllocToHost, routerBwToEach)
        total += migration_cost_of_container(env, container_id, new_host_id, allocBw)
    return total

# total power consumption of the system
def total_power_consumption(env):
    total = 0
    for host in env.hostlist:
        total += host.getPower()
    return total

# calculates the delay matrix for the system
def system_delay_matrix(env, X_scale = 1, k_comp = 1):
    containers = env.containerlist
    hosts = env.hostlist
    matrix = [[0 for _ in containers] for _ in hosts]
    for i, host in enumerate(hosts):
        for j, container in enumerate(containers):
            if container is None:
                continue 
            if container.hostid == host.id:
                matrix[i][j] = total_delay_of_container(env, container.id, k_comp) / X_scale 
    return matrix

# get the list of container locations
def container_locations(env):
    locations = []
    for container in env.containerlist:
        if container is None:
            locations.append(-1)
        else:
            locations.append(container.hostid)
    return locations

# get the list of container resource allocations
def container_resource_allocations(env):
    allocations = []
    for container in env.containerlist:
        if container is None:
            allocations.append(0)
        else:
            allocations.append(container.getApparentIPS())
    return allocations

# return the state represntation of the system
def state_vector(env):
    state = []
    for row in system_delay_matrix(env):
        state.extend(row)
    state.extend(container_locations(env))
    state.extend(container_resource_allocations(env))
    return state 

def reward(env, decision, delay_weight = 1, power_weight = 1):
    return -(delay_weight * total_delay(env) + power_weight * total_power_consumption(env) + total_migration_cost(env, decision))