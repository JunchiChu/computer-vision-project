Two most important files: main.py and stitch.py

To run the program, run "python main.py"
When you see this output:
Enter a extrac function! Your choice: SIFT,SURF,ORB:
Choose one from the three. For example, "SIFT".

After seeing this output:
Pick one: car/ flowers/ greens/ house2/ HouseAndRoad/ RoadView/ sofa/ TV/ trial-data1/ panorama-data1/ panorama-data2: car
Choose one from our dataset. For example, "car".

The output figure is redirected to the corresponding data folder. For example, "data/car/result.PNG".

main.py includes the functions of the UI, feature extraction and image refinement.

stitch.py includes the process of image stitching and seam blending.

The supported versions for packages(pip): python==3.7, opencv-contrib-python==3.4.2.16
For anaconda: opencv==3.2.0, python==3.5.6
