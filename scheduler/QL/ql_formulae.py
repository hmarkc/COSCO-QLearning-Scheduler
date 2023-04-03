# total delay
def total_delay_of_(env, host_id):
    containers = env.getContainersOfHost(host_id)
    delay = 0
    for container_id in containers:
        container = env.getContainerByID(container_id)
        delay += container.getDelay()
    return delay 

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
                matrix[i][j] = host.latency / X_scale 
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
            allocations.append((0, 0, 0))
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

def reward(env, delay_weight = 1, power_weight = 1):
    return 0 