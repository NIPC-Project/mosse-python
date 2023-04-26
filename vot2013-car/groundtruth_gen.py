import csv

groundtruth = []
with open("info/groundtruth.txt", "r") as f:
    for row in csv.reader(f):
        groundtruth.append([(int(float(i))) for i in row])
groundtruth_center = [[int(x + w / 2), int(y + h / 2)] for [x, y, w, h] in groundtruth]
with open("groundtruth.py", "w") as f:
    f.write(f"# [x, y, w, h]\nvot2013_car_groundtruth = {groundtruth}\n")
    f.write("\n")
    f.write(f"# [xc, yc]\nvot2013_car_groundtruth_center = {groundtruth_center}\n")
