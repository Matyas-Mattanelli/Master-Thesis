#Load necessary libraries
library(readr) #Reading csv files
library(readxl) #Reading excel files
library(stargazer) #Exporting summaries
library(rstatix) #Correlation matrices
library(fastDummies) #Dummy variable conversion

###########################
### Give me some credit ###
###########################

#Load training data
cs_training <- as.data.frame(read_csv("D:/My stuff/School/Master/Master Thesis/Data/Kaggle/Give me some credit/cs-training.csv"))
cs_training <- cs_training[, setdiff(colnames(cs_training), "...1")] #Drop the first column for clarity

#Print info about feature types
print(paste0("Give me some credit contains ", ncol(cs_training) - 1, " numerical features and 0 categorical features"), quote = F)

#Rearrange the variables in a more logical way
cs_training <- cs_training[, c(3, 11, 6, 5, 7, 9, 2, 4, 10, 8, 1)]

#Rename the age column
colnames(cs_training)[1] <- "Age"

#Generate summary
summary(cs_training)

#Export summary to latex
#stargazer(cs_training, summary = T, title = "Give me some credit", header = F, summary.stat = c("n", "mean", "sd", "min", "p25", "median", "p75", "max"), label = "tab:gimme")

#Get percentage of delinquents
print(paste0("Percentage of delinquents: ", round(mean(cs_training$SeriousDlqin2yrs) * 100, 2), "%"), quote = FALSE)

#Histogram of Age
hist(cs_training$Age, border = F, breaks = "FD")

#Number of observations with Age over 100
print(paste0(sum(cs_training$Age >= 100), " observations with Age over 100"), quote = F)

#Number of customers below the age 18
print(paste0(sum(cs_training$Age < 18), " observations with Age below 18"), quote = F)
cs_training[cs_training$Age<18,]

#Drop observation with zero Age
cs_training <- cs_training[cs_training$Age != 0, ]

#Check debt ratio
print(paste0(round(sum(cs_training$DebtRatio > 1) / nrow(cs_training) * 100, 2), "% observations with debt ratio over 1"), quote = F)
hist(sort(cs_training$DebtRatio, decreasing = T)[50000:nrow(cs_training)]) #Check trimmed histogram

#Check utilization
print(paste0(round(sum(cs_training$RevolvingUtilizationOfUnsecuredLines > 1) / nrow(cs_training) * 100, 2), "% observations with debt ratio over 1"), quote = F)
hist(sort(cs_training$DebtRatio, decreasing = T)[50000:nrow(cs_training)]) #Check trimmed histogram

#####

#Get correlation matrix
get_cor_mat <- function(x, get_sorted = F) { #Function calculating and formatting a correlation matrix
  correlation_matrix <- cor_mat(x) #Calculate the correlations
  if (get_sorted == F) {
    correlation_matrix_final <- cor_mark_significant(correlation_matrix, cutpoints = c(0, 0.05, 1), symbols = c("*", "")) #Add stars to significant correlations
    for (i in 1:nrow(correlation_matrix_final)){ #Add ones for self-correlation for clarity
      correlation_matrix_final[i, i + 1] <- 1 
    }
    colnames(correlation_matrix_final)[1] <- c("") #Clear the name of the first column
    correlation_matrix_final <- correlation_matrix_final[, 1:(ncol(correlation_matrix_final) - 1)] #Drop the last column
    return(correlation_matrix_final)  
  } else { #If sorted correlations are required
    correlation_matrix <- pull_lower_triangle(correlation_matrix) #Get only the lower part of the matrix
    cor_tab <- cor_gather(correlation_matrix) #Create a table of all correlations
    cor_tab <- cor_tab[order(abs(cor_tab$cor), decreasing = T), ] #Order the absolute correlations
    return(cor_tab)
  }
}
cor_mat_gmsc <- get_cor_mat(cs_training) #Apply the function
colnames(cor_mat_gmsc) <- c("", "Age", "NoD", "MI", "DR", "NoOCLaL", "NoRELoL", "RUoUL", "NoT30-59DPDNW", "NoT60-89DPDNW", "NoT90DL")
cor_mat_gmsc_sorted <- get_cor_mat(cs_training, T) #Get sorted correlations

#Export the correlation matrix
#stargazer(cor_mat_gmsc, summary = F, rownames = F, title = "Correlation matrix (Give me some credit)", label = "tab:corgmsc")

#Replace missing values in number of dependents with mode
print(paste0(length(unique(cs_training$NumberOfDependents)), " unique values for the number of dependents"), quote = F)
cs_training_new <- cs_training #Create a copy of the data set
get_mode <- function(x) { #Define a function for finding the mode of a vector
  val_counts <- table(x) #Get value counts
  return(as.numeric(names(val_counts[which.max(as.numeric(val_counts))]))) #Return the value with maximum cunt
}
cs_training_new$NumberOfDependents[is.na(cs_training$NumberOfDependents)] <- get_mode(cs_training$NumberOfDependents) #Replace missing values with mode

#Replace missing values in income with median
hist(sort(cs_training$MonthlyIncome, decreasing = T)[1000:nrow(cs_training)]) #Check trimmed histogram
cs_training_new$MonthlyIncome[is.na(cs_training$MonthlyIncome)]<- median(cs_training$MonthlyIncome, na.rm = T) #Replace values with the median

#####

#Export the adjusted data set
write_csv(cs_training, "D:/My stuff/School/Master/Master Thesis/Data/Adjusted data/GiveMeSomeCredit.csv")

###################
### Home credit ###
###################

#Load training data
application_train <- as.data.frame(read_csv("D:/My stuff/School/Master/Master Thesis/Data/Kaggle/Home credit/application_train.csv", na = c("", "NA", "XNA")))

#Drop first column 
application_train <- application_train[, setdiff(colnames(application_train), "SK_ID_CURR")]

#Load feature information
hc_info <- as.data.frame(read_excel("D:/My stuff/School/Master/Master Thesis/Data/Kaggle/Home credit/HomeCredit_columns_description.xlsx"))
hc_info <- hc_info[hc_info$Row != "SK_ID_CURR", ]

#Get class information for all columns
hc_info$Class <- sapply(application_train, class)

#Create a mask for numeric features
num_vars <- (hc_info$Type == "numerical") & (hc_info$Class == "numeric")

#Convert character columns to factors (to get a nice summary)
for (col in 1:ncol(application_train)) { #Loop through the columns
  if (is.character(application_train[, col])) { #Check if the column is a character
    application_train[, col] <- as.factor(application_train[, col]) #Convert to factor
  }
}

#Print info about feature types
print(paste0("Home credit contains ", sum(num_vars) - 1, " numeric features and ", ncol(application_train) - sum(num_vars), " categorical features"), quote = F)

#Get the correlation matrix
cor_mat_hc <- get_cor_mat(application_train[, num_vars])
cor_mat_hc_sorted <- get_cor_mat(application_train[, num_vars & (hc_info$Include == T)], T)

#Select only some variables for the summary
var_for_sum <- colnames(application_train)[num_vars & !(grepl('MEDI', colnames(application_train)) | grepl('MODE', colnames(application_train)))]

#Generate summary for numerical features
summary(application_train[, var_for_sum])

#Export summary to latex
stargazer(application_train[, var_for_sum], summary = T, title = "Home credit default risk (summary)", header = F, summary.stat = c("n", "mean", "sd", "min", "p25", "median", "p75", "max"), label = "tab:homecr")

#Check summary of age
summary(-application_train$DAYS_BIRTH / 365)

#Get percentage of missing values
sum(sort(sapply(application_train, function(x) {sum(is.na(x)) / nrow(application_train)}), decreasing = T) > 0.2)

#Mean of dependent variable
mean(application_train$TARGET)

#Export the adjusted data set
write_csv(application_train, "D:/My stuff/School/Master/Master Thesis/Data/Adjusted data/HomeCredit.csv")

################################
### Credit approval data set ###
################################

#Load data
credit_approv <- read.delim("D:/My stuff/School/Master/Master Thesis/Data/UCI Machine Learning Repository/Credit approval data set/crx.txt", sep = ",", header = F)

#Convert missing values to NAs for clarity
credit_approv[credit_approv == "?"] <- NA

#Store numerical columns
ca_num_cols <- c(2, 3, 8, 11, 14, 15)

#Convert numerical columns to numeric
for (col in ca_num_cols) {
  credit_approv[, col] <- as.numeric(credit_approv[, col])
}

#Convert categorical columns to factors
for (col in setdiff(1:(ncol(credit_approv) - 1), ca_num_cols)) {
  credit_approv[, col] <- as.factor(credit_approv[, col])
}

#Specify the target
colnames(credit_approv)[ncol(credit_approv)] <- "Target"
credit_approv$Target <- ifelse(credit_approv$Target == "+", 1, 0) #Convert to binary encoding

#Generate summary
summary(credit_approv)

#Print information about features
print(paste0("The credit approval data set contains ", length(ca_num_cols), " numerical variables and ", length(setdiff(2:ncol(credit_approv), ca_num_cols)), " categorical features"), quote = F)

#Export summary to latex
stargazer(credit_approv, summary = T, title = "Credit Approval (summary)", header = F, summary.stat = c("n", "mean", "sd", "min", "p25", "median", "p75", "max"), label = "tab:creditapp")

#Export the adjusted data set
write_csv(credit_approv, "D:/My stuff/School/Master/Master Thesis/Data/Adjusted data/CreditApproval.csv")

################################################
### Default of credit card clients in Taiwan ###
################################################

#Load data
default_of_credit_card_clients <- as.data.frame(read_excel("D:/My stuff/School/Master/Master Thesis/Data/UCI Machine Learning Repository/Default of credit card clients Taiwan/default of credit card clients.xls", 
                                             skip = 1))
default_of_credit_card_clients <- default_of_credit_card_clients[, setdiff(colnames(default_of_credit_card_clients), "ID")] #Drop the first column for clarity

#Rename the target variable for clarity
colnames(default_of_credit_card_clients)[ncol(default_of_credit_card_clients)] <- "Target"

#Rescale the gender variable
colnames(default_of_credit_card_clients)[2] <- "SEX_FEMALE"
default_of_credit_card_clients$SEX_FEMALE <- ifelse(default_of_credit_card_clients$SEX_FEMALE == 1, 0, 1)

#Transfrom the education variable
default_of_credit_card_clients$EDU_GRAD <- ifelse(default_of_credit_card_clients$EDUCATION == 1, 1, 0)
default_of_credit_card_clients$EDU_UNDERGRAD <- ifelse(default_of_credit_card_clients$EDUCATION == 2, 1, 0)
default_of_credit_card_clients$EDU_HIGH <- ifelse(default_of_credit_card_clients$EDUCATION == 3, 1, 0)
default_of_credit_card_clients <- default_of_credit_card_clients[, setdiff(colnames(default_of_credit_card_clients), "EDUCATION")] #Drop the education variable

#Transform the marriage variable
default_of_credit_card_clients$MARRIAGE <- ifelse(default_of_credit_card_clients$MARRIAGE == 1, 1, 0)

#Generate summary
summary(default_of_credit_card_clients)

#Export summary to latex
stargazer(default_of_credit_card_clients, summary = T, title = "Default of Credit Card Clients in Taiwan", header = F, summary.stat = c("n", "mean", "sd", "min", "p25", "median", "p75", "max"), label = "tab:taiw")

#Export the adjusted data set
write_csv(default_of_credit_card_clients, "D:/My stuff/School/Master/Master Thesis/Data/Adjusted data/CreditCardTaiwan.csv")

####################################
### South German Credit Data Set ###
####################################

#Load data
SouthGermanCredit <- as.data.frame(read_csv("D:/My stuff/School/Master/Master Thesis/Data/UCI Machine Learning Repository/South German Credit Data Set/SouthGermanCredit.csv"))

#Convert characters to factors
for (col in 1:ncol(SouthGermanCredit)) { #Loop through the columns
  if (is.character(SouthGermanCredit[, col])) { #Check if the column is a character
    SouthGermanCredit[, col] <- as.factor(SouthGermanCredit[, col]) #Convert to factor
  }
}

#Check the class of the columns
for (i in 1:ncol(SouthGermanCredit)) {
  print(class(SouthGermanCredit[, i]))
}

#Specify the target
colnames(SouthGermanCredit)[ncol(SouthGermanCredit)] <- "Target" #Change the name for clarity
SouthGermanCredit$Target <- ifelse(SouthGermanCredit$Target == "bad", 1, 0) #Binary encoding

#Generate summary
summary(SouthGermanCredit)

#Export summary to latex
stargazer(SouthGermanCredit, summary = T, title = "South German Credit", header = F, summary.stat = c("n", "mean", "sd", "min", "p25", "median", "p75", "max"), label = "tab:southger")

#Export the adjusted data set
write_csv(SouthGermanCredit, "D:/My stuff/School/Master/Master Thesis/Data/Adjusted data/SouthGermanCredit.csv")
