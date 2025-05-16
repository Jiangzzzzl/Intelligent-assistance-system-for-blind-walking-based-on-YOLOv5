foc = 1700.0        # 镜头焦距
#行人用高度表示，根据公式 D = (F*W)/P，知道相机焦距F、行人的高度66.9（单位英寸→170cm/2.54）、像素点距离 h，即可求出相机到物体距离D。
# 这里用到h-2是因为框的上下边界像素点不接触物体

# roadblocks = 70cm
real_hight_roadblocks = 170/2.54
#4.7
# 105
real_hight_fence = 105/2.54
# 30
real_hight_box = 30/2.54
# 300
real_hight_guideboard = 300/2.54
# 250
real_hight_trafficlight = 250/2.54
# 30
real_hight_stone = 30/2.54
# 150
real_hight_tree = 150/2.54
# chair = 60
real_hight_chair = 23.62
# 30
real_hight_dog = 11.81
# 30
real_hight_cat = 11.81
# 170
real_hight_people = 170/2.54
# 150
real_hight_car = 150/2.54
# 100
real_hight_bicycle = 100/2.54
# 120
real_hight_plant = 120/2.54
# 80
real_hight_rubbishbin = 80/2.54
# 300
real_hight_pole = 300/2.54
# 150
real_hight_distributorbox = 150/2.54
# 55
real_hight_cart = 55/2.54
# 120
real_hight_motorcycle = 120/2.54
# 300
real_hight_streetlight = 300/2.54
#70
real_hight_brand = 70/2.54

# 自定义函数，单目测距
def roadblocks_distance(h):
    dis_inch = (real_hight_roadblocks * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100/15*4.7
    return dis_m

def fence_distance(h):
    dis_inch = (real_hight_fence * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def box_distance(h):
    dis_inch = (real_hight_box * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm /10*4/100
    return dis_m

def guideboard_distance(h):
    dis_inch = (real_hight_guideboard * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm / 10*4/100
    return dis_m

def trafficlight_distance(h):
    dis_inch = (real_hight_trafficlight * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm / 100*4/100
    return dis_m

def stone_distance(h):
    dis_inch = (real_hight_stone * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm / 10*4/100
    return dis_m

def tree_distance(h):
    dis_inch = (real_hight_tree * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def chair_distance(h):
    dis_inch = (real_hight_chair * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def dog_distance(h):
    dis_inch = (real_hight_dog * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def cat_distance(h):
    dis_inch = (real_hight_cat * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def people_distance(h):
    dis_inch = (real_hight_people * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def car_distance(h):
    dis_inch = (real_hight_car * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def bicycle_distance(h):
    dis_inch = (real_hight_bicycle * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def plant_distance(h):
    dis_inch = (real_hight_plant * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4
    return dis_m

def rubbishbin_distance(h):
    dis_inch = (real_hight_rubbishbin * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def pole_distance(h):
    dis_inch = (real_hight_pole * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def distributorbox_distance(h):
    dis_inch = (real_hight_distributorbox * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def cart_distance(h):
    dis_inch = (real_hight_cart * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def motorcycle_distance(h):
    dis_inch = (real_hight_motorcycle * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def streetlight_distance(h):
    dis_inch = (real_hight_streetlight * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m

def brand_distance(h):
    dis_inch = (real_hight_brand * foc) / (h - 2)
    dis_cm = dis_inch * 2.54
    dis_cm = int(dis_cm)
    dis_m = dis_cm/10*4/100
    return dis_m