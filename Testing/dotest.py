import os


folder1 = "dataset/brats2d/images"
folder2 = "dataset/brats2d/masks"

count1 = len(os.listdir(folder1))
count2 = len(os.listdir(folder2))

print("Total items in folder:", count1,count2)
