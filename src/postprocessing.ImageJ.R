## Jasper Lee
## 1/25/2021
## Processing for CODEX Machine Learning-based Cell Segmentation Data 

# This program is designed to process CODEX data from the machine learning-based platform, perform 
# area-based thresholding, and prepare the data for export in a MAV-compatiable format. This
# code can be customized for different experiments - the number and type of markers, threshold
# cutoff, and data sources can all be customized, and are commented accordingly.


# Setup --------------------------------------------------------------------------------------------

library(dplyr)
library(tidyverse)
library(devtools)

number.images <- 20 # change this if the number of processed images is different
setwd("D:\\Sascha\\New_Try") # the 
# working directory where input and output data are found


# Marker intensity ---------------------------------------------------------------------------------

color.intensity <- read.csv("colorresults.csv") # input data for color image marker intensity
raw.marker.data <- data.frame(1:(nrow(color.intensity)/number.images))

for(i in seq(1, nrow(color.intensity), nrow(color.intensity)/number.images)){
  data <- color.intensity[i:(i + (nrow(color.intensity)/number.images) - 1), 3]
  raw.marker.data <- cbind(raw.marker.data, data)
}


# Greyscale area calculation -----------------------------------------------------------------------

greyscale.area <- read.csv("greyscaleresults.csv") # input data for greyscale areas
area.data <- data.frame(1:(nrow(greyscale.area)/number.images))

for(i in seq(1, nrow(greyscale.area), nrow(greyscale.area)/number.images)){
  data <- greyscale.area[i:(i + (nrow(greyscale.area)/number.images) - 1), 7]
  area.data <- cbind(area.data, data)
}

cyto.area <- 100 - area.data[, 7]
cyto.area[which(cyto.area == 0)] <- 0.1 # this is done to prevent division by 0 - change as needed

for(i in seq(1, nrow(area.data))){
  area.data[i,] <- 100 * area.data[i,]/cyto.area[i]
}

area.data[,1] <- 1:nrow(area.data)


# Combined dataframe assembly ----------------------------------------------------------------------

position.data <- read.csv("XandY.csv") # data for the cell positions - see documentation
position.data <- position.data[,4:5]
thresholded.data <- position.data

for(i in seq(2, ncol(area.data), 1)){
  thresholded.data <- cbind(thresholded.data, raw.marker.data[, i], area.data[, i])
}


# Data thresholding, processing, and export --------------------------------------------------------

threshold <- 0 # this is percent (e.g. 5 is 5%) - change as needed. Area threshold

data <- thresholded.data
index <- nrow(data)
export <- data.frame(data$X)
export$XMax <- data$X
export$YMin <- data$Y
export$YMax <- data$Y


for(i in 3:ncol(data)){
  export[,i+2] <- data[,i]
}

export <- export %>% mutate_all(as.numeric)
export[is.na(export)] <- 0 # catches invalid values if they occur - change as needed

for(j in seq(5, ncol(export), 2)){
  for(i in 1:nrow(export)){
    if(export[i,j+1] < threshold){
      export[i,j] <- 0
    }
  }
}

final.data <- export[,c(1:4,seq(5, ncol(export),2))]
final.data$ID <- 1:nrow(final.data)

colnames(final.data) <- c("XMin", "XMax", "YMin", "YMax", # change column names as needed
                          "DAPI-1 Nucleus Intensity", "Blank-1 Nucleus Intensity", 
                          "Blank-2 Nucleus Intensity", "Blank-3 Nucleus Intensity", 
                          "DAPI-3 Nucleus Intensity", "Empty-1 Nucleus Intensity",
                          "CD3e Nucleus Intensity", "PD-L1 Nucleus Intensity", 
                          "DAPI-4 Nucleus Intensity", "E-Cadherin Nucleus Intensity", 
                          "CD68 Nucleus Intensity","TCF7 Nucleus Intensity",
                          "DAPI-5 Nucleus Intensity","Empty-2 Nucleus Intensity", 
                          "Empty-3 Nucleus Intensity","MHC-I Nucleus Intensity", 
                          "DAPI-6 Nucleus Intensity","Blank-4 Nucleus Intensity",
                          "Blank-5 Nucleus Intensity","Blank-6 Nucleus Intensity",
                          "Cell ID")

final.data$"Object ID" <- final.data$"Cell ID"
final.data.1 <- final.data[,c(5:(ncol(final.data) - 2), 1:4, ncol(final.data), ncol(final.data) - 1)]
final.data.1 <- final.data.1 %>%  mutate_all(as.character)
write.table(final.data.1, file = "New_Try.csv", # export file
            sep=",", col.names = NA, row.names = TRUE)

