
#Libraries for RFE
library("dplyr")
library("faux")
library("DataExplorer")
library("caret")
library("randomForest")

#control random forest RFE
cntrl=function(){
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 3, # number of repeats
                      number = 3,verbose = FALSE) # number of folds
return(control)
}

#filter
fltr=function(Time,NP){
data3 = na.omit(data2  %>% filter(Exposure.Time == Time,Cell.Nps == NP))
x = data3[,-c(1:3)]
y = data3[,2]
thedata=data.frame("y"=y,"X"=x)
return(thedata)
}

#Regression RFE
R_RFE=function(x,y,size,Size,control,NP){
  
  #set to numeic
  y = as.numeric(y)

  #run RFE
  memory.limit(size=56000)
  resultr <- rfe(x =x, y =y, sizes = size, rfeControl = control)

  #csv of predictors
  write.csv(predictors(resultr),file=paste(Time,NP,Size,
                                          "predictorsR.csv",sep="_"))
  #return results
  return(resultr)
}

#Classification RFE
C_RFE=function(x,y,size,Size,control,NP){
  
  #set to factor
  y = as.factor(y)
  
  #runRFE clasification and record Time
  memory.limit(size=56000)
 
  #run RFE
  resultc <- rfe(x =x, y =y, sizes = size, rfeControl = control)
  
  #csv of predictors
  write.csv(predictors(resultc),file=paste(Time,NP,Size,
                                           "RFE_TCGA_Rfeatures.csv",sep="_"))
  return(resultc)
}

#stack lists to create correlation variables as cormat-----------------------
#Preds tienen que ser predictors(resultr) predictors(resultc)
cormatrix=function(predsr,predsc,y,x){
  
corlist=NULL
if(length(predsr)<10){
  for(i in 1:length(predsr)){
    corlist=rbind(corlist,predsr[i])
  }
}else{
  for(i in 1:10){
    corlist=rbind(corlist,predsr[i])
  }
}

if(length(predsc)<10){
  for(i in 1:length(predsc)){
    corlist=rbind(corlist,predsc[i])
  }
}else{
  for(i in 1:10){
    corlist=rbind(corlist,predsc[i])
  }
}
corlist=as.character(corlist)
cormat=cbind(as.numeric(y),select(x,c(corlist)))
return(cormat)
}


