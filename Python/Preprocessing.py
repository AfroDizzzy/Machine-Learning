import numpy as np
import pandas as pd


#load the data in
data = pd.read_csv("US_Accidents_Dec20_Updated.csv")
data.head()

#the columns before any processing
print(data.dtypes)

#shows the number of NAN values in each column
data.isnull().sum().sort_values(ascending = False).head(25)

#shows the columns left in the normal

#Simplify the wind direction
data.loc[data['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
data.loc[(data['Wind_Direction']=='West')|(data['Wind_Direction']=='WSW')|(data['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
data.loc[(data['Wind_Direction']=='South')|(data['Wind_Direction']=='SSW')|(data['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
data.loc[(data['Wind_Direction']=='North')|(data['Wind_Direction']=='NNW')|(data['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
data.loc[(data['Wind_Direction']=='East')|(data['Wind_Direction']=='ESE')|(data['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
data.loc[(data['Wind_Direction']=='VAR')|(data['Wind_Direction']=='Variable'),'Wind_Direction'] = 'variable'
print("Wind Direction after simplification: ", data['Wind_Direction'].unique())
data['Calm'] = np.where(data['Wind_Direction'].str.contains('CALM', case=False, na = False), 1, 0)
data['West'] = np.where(data['Wind_Direction'].str.contains('W', case=False, na = False), 1, 0)
data['South'] = np.where(data['Wind_Direction'].str.contains('S', case=False, na = False), 1, 0)
data['North'] = np.where(data['Wind_Direction'].str.contains('N', case=False, na = False), 1, 0)
data['East'] = np.where(data['Wind_Direction'].str.contains('E', case=False, na = False), 1, 0)
data['Variable'] = np.where(data['Wind_Direction'].str.contains('variable', case=False, na = False), 1, 0)


#Simplify weather condition
data['Clear'] = np.where(data['Weather_Condition'].str.contains('Clear', case=False, na = False), 1, 0)
data['Cloud'] = np.where(data['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na = False), 1, 0)
data['Rain'] = np.where(data['Weather_Condition'].str.contains('Rain|storm', case=False, na = False), 1, 0)
data['Heavy_Rain'] = np.where(data['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), 1, 0)
data['Snow'] = np.where(data['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na = False), 1, 0)
data['Heavy_Snow'] = np.where(data['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), 1, 0)
data['Fog'] = np.where(data['Weather_Condition'].str.contains('Fog', case=False, na = False), 1, 0)

#Simplify the Severity
data.loc[(data['Severity']=='1')|(data['Severity']=='2')|(data['Severity']=='3'),'Wind_Direction'] = '0'
data.loc[data['Severity']=='4','Severity'] = '1'

#Convert all boolean variables to float's
data["Amenity"] = data["Amenity"].astype(float)
data["Bump"] = data["Bump"].astype(float)
data["Crossing"] = data["Crossing"].astype(float)
data["Give_Way"] = data["Give_Way"].astype(float)
data["Junction"] = data["Junction"].astype(float)
data["No_Exit"] = data["No_Exit"].astype(float)
data["Railway"] = data["Railway"].astype(float)
data["Roundabout"] = data["Roundabout"].astype(float)
data["Station"] = data["Station"].astype(float)
data["Stop"] = data["Stop"].astype(float)
data["Traffic_Calming"] = data["Traffic_Calming"].astype(float)
data["Traffic_Signal"] = data["Traffic_Signal"].astype(float)
data['Clear'] = data["Clear"].astype(float)
data['Cloud'] = data["Cloud"].astype(float)
data['Rain'] = data["Rain"].astype(float)
data['Heavy_Rain'] = data["Heavy_Rain"].astype(float)
data['Snow'] = data["Snow"].astype(float)
data['Heavy_Snow'] = data["Heavy_Snow"].astype(float)
data['Fog'] = data["Fog"].astype(float)

#Creating some time features
# data["Start_Time"] = pd.to_datetime(data["Start_Time"])
# data["Year"] = data["Start_Time"].dt.year
# data["Month"] = data["Start_Time"].dt.month
# data["Hour"] = data["Start_Time"].dt.hour

# #Creats day of the week feature
# days_each_month = np.cumsum(np.array([31,28,31,30,31,30,31,31,30,31,30,31]))
# nday1 = [days_each_month[arg-1] for arg in data["Month"].values]
# nday = nday1 + data["Start_Time"].dt.day.values
# data['Day'] = nday
print("final columns left")
print(data.dtypes.size)
#The columns to remove from the dataset
data = data.drop(["ID","Start_Lat", "Start_Lng","End_Lat", "End_Lng", "Description", "Number", "Precipitation(in)", 
                  "Wind_Chill(F)", "Wind_Speed(mph)", "Nautical_Twilight", "Timezone", 
                  "Astronomical_Twilight", "Sunrise_Sunset", "Civil_Twilight", "Weather_Timestamp", "Turning_Loop", "Country", "Weather_Condition", "Wind_Direction", "Start_Time", "End_Time", "Street", "Side","City","County", "State", "Zipcode", "Airport_Code"], axis = 1)


#shows the columns left in the dataset
print("final columns left")
print(data.dtypes.size)
print(data.dtypes)

#final size of the dataset
size = data.memory_usage().sum() / 1024 / 1024 
print("Data memory size: %.2f MB" % size)

#removes the rows where NaN is found in any of the values
data.dropna()

#Saves the preprocessed data
data.to_csv("preprocessed.csv")


##Credit to https://www.kaggle.com/jingzongwang/usa-car-accidents-severity-prediction#1-OVERVIEW-&-PREPROCESSING 
## provided alot of ideas and code on how to preprocessing the various categories