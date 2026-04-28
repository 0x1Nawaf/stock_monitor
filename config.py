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


#the  command (image) = ffmpeg -i <input> -vf scale="iw/<the scale from 1 to 2>:-1" <output>
#the  command (video) = ffmpeg -i <input> -vf "scale=-2:'min(<900-320>,ih)'" <output>
# scales: 1 = is mid-high, 1.5 = medium, 2 = low, 2.5 = very low

# poster creator ffmpeg -i <input> -vf "select=eq(n\,0)" -vframes 1 poster.png