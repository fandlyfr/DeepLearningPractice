#data yang akan di train
celcius=c(1,20,59,5,7,290)
kelvin=c(274,293,332,277,280,563)

#combine data
df=data.frame(celcius,kelvin)

#using neuralnet
require(neuralnet)
nn<- neuralnet(kelvin~celcius,data = df, hidden = 0, act.fct = "logistic",
               linear.output = F)

#menampilkan plot nn
plot(nn)

#membuat data yang akan di test
TKS_test<- c(0,2,3,4)
test<- data.frame(TKS_test,CSS_test)

#memprediksi data
predict<- compute(nn,test)
predict$net.result
