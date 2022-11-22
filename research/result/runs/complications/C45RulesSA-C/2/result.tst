@relation   complications
@attribute Age integer[29,73]
@attribute Height integer[150,176]
@attribute Weight integer[44,118]
@attribute BMI real[17.47,44.96]
@attribute OM integer[0,1]
@attribute RUM{3,2,0,1}
@attribute Lymph integer[0,3]
@attribute FIGO{0,1,2,3,4,5}
@attribute Complication{no,yes}
@inputs Age,Height,Weight,BMI,OM,RUM,Lymph,FIGO
@outputs Complication
@data
no yes
no no
no yes
no yes
no yes
no yes
yes yes
yes no
yes no
yes yes
yes yes
