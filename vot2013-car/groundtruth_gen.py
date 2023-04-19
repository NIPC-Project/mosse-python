import csv

groundtruth = []
with open("info/groundtruth.txt") as f:
    for row in csv.reader(f):
        groundtruth.append([(int(float(i))) for i in row])
print("# [x, y, w, h]\nvot2013_car_groundtruth = ", groundtruth)
