@relation   breast
@attribute Age{10,20,30,40,50,60,70,80,90}
@attribute Menopause integer[0,2]
@attribute Tumor-size integer[0,11]
@attribute Inv-nodes integer[0,12]
@attribute Node-caps integer[0,1]
@attribute Deg-malig integer[1,3]
@attribute Breast{0,1}
@attribute Breast-quad integer[0,4]
@attribute Irradiated{0,1}
@attribute Class{0,1}
@inputs Age,Menopause,Tumor-size,Inv-nodes,Node-caps,Deg-malig,Breast,Breast-quad,Irradiated
@outputs Class
@data
0 1
0 0
0 1
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 0
0 1
0 0
0 0
1 0
1 0
1 0
1 0
1 1
1 0
1 0
1 0
