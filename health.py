

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = "C:\\Users\\kiran\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Anaconda3 (64-bit)\\healthcare_dataset.csv"
health_data = pd.read_csv("C:\\Users\\kiran\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Anaconda3 (64-bit)\\healthcare_dataset.csv")

# Step 1: Normalize column names by making them lowercase and replacing spaces with underscores
health_data.columns = [col.lower().replace(' ', '_') for col in health_data.columns]

# Step 2: Trim and standardize text fields (e.g., gender, name)
health_data['gender'] = health_data['gender'].str.strip().str.capitalize()
health_data['name'] = health_data['name'].str.strip().str.title()

# Step 3: Convert date columns to datetime objects
health_data['date_of_admission'] = pd.to_datetime(health_data['date_of_admission'], errors='coerce')
health_data['discharge_date'] = pd.to_datetime(health_data['discharge_date'], errors='coerce')

# Step 4: Handle any missing values (optional - can fill or drop as needed)
health_data.fillna(method='ffill', inplace=True)  # Forward fill as an example

# Step 5: Display cleaned data sample
print(health_data.head())

# Save cleaned data to a new CSV (optional)
cleaned_file_path = 'D:\python yhills project\cleaned.csv'
health_data.to_csv(cleaned_file_path, index=False)
# Extract relevant features for disease prediction
relevant_features = health_data[[
    'age',          # Age of the patient
    'gender',       # Gender of the patient
    'blood_type',   # Blood type as a potential risk factor
    'medical_condition',  # Existing medical condition
    'admission_type',  # Admission type (e.g., urgent, elective)
    'medication',  # Medications currently used
    'test_results'  # Results of medical tests (e.g., Normal, Abnormal)
]]

# Display the first few rows of the relevant features
print(relevant_features.head())

# Save the relevant features to a new CSV (optional)
relevant_features_file_path = 'D:\\python yhills project\\features.csv'
relevant_features.to_csv(relevant_features_file_path, index=False)


# Load the relevant features
file_path = 'D:\\python yhills project\\features.csv'
data = pd.read_csv('D:\\python yhills project\\features.csv')

# Step 1: Data Preprocessing
# Convert categorical variables into numeric using Label Encoding
label_encoders = {}
for column in ['gender', 'blood_type', 'medical_condition', 'admission_type', 'medication', 'test_results']:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Step 2: Split data into features and target
X = data.drop('medical_condition', axis=1)  # Features (using all but the target)
y = data['medical_condition']  # Target variable

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Train a simple classifier (e.g., RandomForest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Feature importance (optional)
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nFeature Importances:\n", feature_importances.sort_values(ascending=False))


