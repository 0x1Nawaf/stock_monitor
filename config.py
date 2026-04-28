env_config ={}
def load():
    global env_config 
    config_file = open('.conf','r')
    configs = config_file.readlines()
    for config_row in configs:
        x = config_row.split("=")
        if(len(x) > 1):
            env_config[x[0]] = x[1].strip()
load()