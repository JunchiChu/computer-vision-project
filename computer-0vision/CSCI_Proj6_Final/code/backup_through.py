import os
from skimage import io 

photo_cat=['bridge','car','flowers','greens','house and road','road view','sofa','TV','house1','trial_data1','panorama-data1','panorama-data2']
parainput=input("Pick one: bridge/ car/ greens/ house&road/ road view/ sofa / TV / house1 / trial_data1/ panorama-data 1&2: ")
while parainput not in photo_cat:
  parainput = input("Pick A VALID one: bridge/ car/ greens/ house&road/ road view/ sofa / TV / house1 / trial_data1/ panorama-data 1&2::")
print("You select [",parainput,"] as input")

directory = r"../data/"+parainput
for filename in os.listdir(directory):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        #print("sssss")
        print(os.path.join(directory, filename))
    else:
        continue