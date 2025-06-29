#Libraries for RFE 

#Import Datasets
TCGAfile = read.csv("/Users/nataliaa.rodriguez/ININ 4998 Pancreatic Cancer/TCGA_Preprocessed.csv")
ICGCfile = read.csv("/Users/nataliaa.rodriguez/ININ 4998 Pancreatic Cancer/ICGC_Preprocessed.csv")

#control random forest RFE
control <- rfeControl(functions = rfFuncs, # random forest
                      method = "repeatedcv", # repeated cv
                      repeats = 3, # number of repeats
                      number = 5,verbose = FALSE) # number of folds

#X and Y Factors
Ix=ICGCfile[,2:35601]
Iy=factor(ICGCfile[,'VS'])

Tx=TCGAfile[,2:20025]
Ty=factor(TCGAfile[,'OS'])

#RFE on TCGA
TCGAresult200 <- rfe(x =Tx, y =Ty, sizes = seq(from=1,to=200,by=1), rfeControl = control)
TCGAresult200
TCGAresult300 <- rfe(x =Tx, y =Ty, sizes = seq(from=1,to=300,by=1), rfeControl = control)
TCGAresult300
TCGAresult500 <- rfe(x =Tx, y =Ty, sizes = seq(from=1,to=500,by=1), rfeControl = control)
TCGAresult500
TCGAresult1000 <- rfe(x =Tx, y =Ty, sizes = seq(from=1,to=1000,by=1), rfeControl = control)
TCGAresult1000


#RFE on ICGC
ICGCresult200 <- rfe(x =Ix, y =Iy, sizes = seq(from=1,to=200,by=1), rfeControl = control)
ICGCresult200
ICGCresult300 <- rfe(x =Ix, y =Iy, sizes = seq(from=1,to=300,by=1), rfeControl = control)
ICGCresult300
ICGCresult500 <- rfe(x =Ix, y =Iy, sizes = seq(from=1,to=500,by=1), rfeControl = control)
ICGCresult500
ICGCresult1000 <- rfe(x =Ix, y =Iy, sizes = seq(from=1,to=1000,by=1), rfeControl = control)
ICGCresult1000

#Results

predictors(TCGAresult300)
predictors(TCGAresult500)
predictors(TCGAresult1000)

predictors(ICGCresult200) 
predictors(ICGCresult300) 
predictors(ICGCresult500) 
predictors(ICGCresult1000) 

#Export to csv files
TCGAexport = TCGAfile[,which(names(TCGAfile) %in% predictors(TCGAresult300))]
write.csv(TCGAexport,file="TCGA300_stp=1.csv",row.names=FALSE)
ICGCexport= ICGCfile[,which(names(ICGCfile) %in% predictors(ICGCresult200))]
write.csv(TCGAexport,file="ICGC200_stp=1.csv",row.names=FALSE)

# Heatmap CSV
TCGAcsv = TCGAexport["OS"] <- Ty
TCGAcsv
write.csv(ICGCcsv,file="ICGC300_com=1.csv",row.names=FALSE)

ICGCcsv = ICGCexport["VS"] <- Iy
ICGCcsv
write.csv(ICGCcsv,file="ICGC300_complete=1.csv",row.names=FALSE)

#Finding TCGA predictors in ICGC dataset
write.csv(ICGCfile[,which(names(ICGCfile) %in% TCGApredictors)],file="ICGC_TCGA_Rfeatures.csv",row.names = FALSE) 
#Finding ICGC predictors in TCGA dataset
write.csv(TCGAfile[,which(names(TCGAfile) %in% ICGCpredictors)],file="TCGA_ICGC_Rfeatures.csv",row.names = FALSE) 

