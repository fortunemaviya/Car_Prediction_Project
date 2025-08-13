import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder

#read the car price dataset
df=pd.read_csv('data/car_data.csv')

exchange_rate = 0.22  # Or fetch historical average
df['selling_price'] = df['selling_price'] * exchange_rate

# Display basic information about the dataset to get an insight into the data
df.info()

#discriptive statics are essential to understand the distribution of the data , range and key variations to guide EDA.
df.describe()

# 1. Data Preprocessing
# Check for missing values
df.isnull().sum()

# Create label encoders for categorical variables
le = LabelEncoder()


# Encode categorical variables
df['brand_encoded'] = le.fit_transform(df['brand'])
df['model_encoded'] = le.fit_transform(df['model'])

#checking Data distribution and posible outliers
numerical_cols = ['vehicle_age', 'km_driven', 'mileage', 'max_power', 'selling_price']

# Select features for modeling
features = ['brand_encoded', 'model_encoded', 'vehicle_age', 'km_driven', 'mileage', 'max_power', 'seats']
X = df[features]
y = df['selling_price']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

outlierplot=X_scaled.drop(['brand_encoded', 'model_encoded', 'seats'], axis =1)


def remove_outliers_iqr(df, cols):
    for col in cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"{col} cleaned: {df.shape[0]} rows remaining")
    return df

# Specify columns you want to clean
outlierplot = ['km_driven', 'mileage', 'max_power', 'selling_price']

# Clean your DataFrame
df = remove_outliers_iqr(df,outlierplot)

#checking for null values
df.isnull().sum()

# Drop rows with missing values in ANY critical column
df = df.dropna(subset=['km_driven', 'mileage', 'max_power', 'selling_price'])

#checking reseted shape
df.shape

# Reset indices to avoid alignment issues
df = df.reset_index(drop=True)

#confirming the shape again
df.shape

#correlation check
# 
#plt.figure(figsize=(12, 8))
#sns.heatmap(df[features + ['selling_price']].corr(), annot=True, cmap='coolwarm')
#plt.title('Correlation Matrix')
#plt.tight_layout()
#plt.show()

X=df.drop(['brand_encoded', 'model_encoded', 'seats','car_name', 'brand', 'model', 'selling_price'], axis=1)
y=df['selling_price']

#train test split
from sklearn.model_selection import train_test_split

Xtrain, Xtest, ytrain, ytest=train_test_split(X,y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

#  Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model= RandomForestRegressor(random_state=42)

# Train the model
model.fit(X_train, y_train)
    
# Make predictions
y_pred = model.predict(X_test)

#Save the model
joblib.dump(model,'car_price.pkl')
print("The model has been saved successfully")