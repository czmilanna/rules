@relation    heart
@attribute Age integer[29,77]
@attribute Sex integer[0,1]
@attribute ChestPainType integer[1,4]
@attribute RestBloodPressure integer[94,200]
@attribute SerumCholestoral integer[126,564]
@attribute FastingBloodSugar integer[0,1]
@attribute ResElectrocardiographic integer[0,2]
@attribute MaxHeartRate integer[71,202]
@attribute ExerciseInduced integer[0,1]
@attribute Oldpeak real[0.0,62.0]
@attribute Slope integer[1,3]
@attribute MajorVessels integer[0,3]
@attribute Thal integer[3,7]
@attribute Class{1,2}
@inputs Age,Sex,ChestPainType,RestBloodPressure,SerumCholestoral,FastingBloodSugar,ResElectrocardiographic,MaxHeartRate,ExerciseInduced,Oldpeak,Slope,MajorVessels,Thal
@outputs Class
@data
1 1
1 2
1 1
1 1
1 1
1 1
1 1
1 2
1 1
1 1
1 1
1 1
1 1
1 1
1 1
2 1
2 2
2 2
2 2
2 2
2 1
2 2
2 2
2 2
2 2
2 1
2 2