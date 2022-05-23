#This is the formula for calculating the fair price of a stock

EPS = 3.70 #yahoo finance value in summary
GrowthRate = 0.1140 #yahoo finance in 5 years in analysis
MinimumRateOfReturn = 0.15 #fixed value
PEratio = 24.02 #yahoo finance value in summary
MarginOfSafety = 0
for x in range(9):
    EPS = EPS + (EPS*GrowthRate)
NewEps = EPS * PEratio
for x in range(9):
    NewEps = NewEps / (1+MinimumRateOfReturn)
print("The fair price of a stock is £",round(NewEps,2)) ##rounded to 2 d.p

#taken from https://www.youtube.com/watch?v=nX2DcXOrtuo

#There are 2 other ways to calculate PEratio

#one is to go to msn money and calculate the average between
#5 year high and 5 year low P/E ratios

#second is to double the GrowthRate percentage
#In the video, it is recommended to use the lower value out of those options

#apple
#EPS = 6.01
#GrowthRate = 0.1485
#MinimumRateOfReturn = 0.15
#PEratio = 26.09
#FairPrice = 154.97
#-----------
#Tesla
#EPS = 4.90
#GrowthRate = 0.2170
#MinimumRate = 0.15
#PEratio = 167.55
#FairPrice = 1366.71
#------------------
#Mercedes
#EPS = 12.94
#GrowthRate = 0.4727
#MinimumRate = 0.15
#PEratio = 4.57
#FairPrice = 547.77
#--------------------------
#Coca-Cola
#EPS = 2.25
#Growth = 0.0724
#MinimumRate = 0.15
#PEratio = 26.82
#FairPrice = 32.18
#------------------------
#McDonalds
#EPS = 10.04
#Growth = 0.1297
#MinimumRate = 0.15
#PEratio = 23.64
#FairPrice = 202.19
#-----------------------
#Microsoft
#EPS = 9.39
#Growth = 0.1740
#MinimumRate = 0.15
#PEratio = 32.35
#FairPrice = 365.82
#-----------------------
#Walmart
#EPS = 4.87
#Growth = 0.0835
#MinimumRate = 0.15
#PEratio = 29.51
#FairPrice = 84.08
#------------------------
#Nike
#EPS = 3.79
#Growth = 0.1534
#MinimumRate = 0.15
#PEratio = 35.54
#FairPrice = 138.32
#--------------------------------
#Tesco
#EPS = 84.60
#Growth = 0.2964
#MinimumRate = 0.15
#PEratio = 3.30
#FairPrice = 286.70
#-----------------------------
#Starbucks
#EPS = 3.70
#Growth = 0.1140
#MinimumRate = 0.15
#PEratio = 24.02
#FairPrice = 66.75
