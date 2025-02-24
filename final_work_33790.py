import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from scikeras.wrappers import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

class AnimalMovementApp:
    def __init__(self):
        #Global variables into the app
        self.df = None #Dataframe to hold the data
        self.preprocessor = None #Preprocessor object Pipeline
        self.full_pipeline = None #Full pipeline object (preprocessor + model)
        self.hp_params = None #Hyperparameters for the model
        self.n_classes = None #Number of classes in the model (polygons)
        self.polygon_to_idx = None #Mapping polygons to index
        self.idx_to_polygons = None #Inverse Mapping index to polygons
        #self.model = None #Model object 


    
    def load_data(self, csv_path):
        """Load the cvs file """
        self.df = pd.read_csv(csv_path)
        print(f"Data loaded from {csv_path}, with a Shape: {self.df.shape}")

    def report_missing_values(self):
        """Report missing values in the data"""
        missing = self.df.isnull().sum()
        print("Missing values in the data:")
        print(missing[missing > 0])

    def report_outliers(self):
        """
        Report outliers in the data
        outliers are identified as values which are above or belowe 3.5 times 
        interquantile range (IQR)
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        outlier_report = {}
        for col in numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 3.5 * iqr
            upper_bound = q3 + 3.5 * iqr
            outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            if not outliers.empty:
                outlier_report[col] = outliers
        print("Outliers in the data:")
        print(outlier_report)

    def treat_data(self, num_strategy='mean', cat_strategy='most_frequent'):
        """
        Treat the data by filling missing values
        for numerical values "mean" or "median" can be used
        for categorical values "most_frequent" can be used
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        num_imputer = SimpleImputer(strategy=num_strategy)
        cat_imputer = SimpleImputer(strategy=cat_strategy)
        self.df[numeric_cols] = num_imputer.fit_transform(self.df[numeric_cols])
        self.df[categorical_cols] = cat_imputer.fit_transform(self.df[categorical_cols])
        print("Data treated for missing values")

    def report_distribution(self, category=None):
        """
        Report the distribution of the data
        - Quantitative Variables: BoxPLot will be showed, optionally distributed by the "category" variable
        - Categorical Variables: Countplot will be showed
        """
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if category and category in self.df.columns:
            for col in numeric_cols:
                plt.figure(figsize=(12, 6))
                sns.boxplot(x=category, y=col, data=self.df)
                plt.title(f"{col} Distribution by {category}")
                plt.show()
        else:
            for col in numeric_cols:
                plt.figure(figsize=(12, 6))
                sns.boxplot(y=col, data=self.df)
                plt.title(f"{col} Distribution")
                plt.show()
        categorical_cols = self.df.select_dtypes(exclude=[np.number]).columns.tolist()
        for col in categorical_cols:
            plt.figure(figsize=(12, 6))
            sns.countplot(x=col, data=self.df)
            plt.title(f"{col} Distribution")
            plt.show()
    
    def report_correlation(self, hp_params=None):
        """
        Report the correlation between the quantitative varoables
        
        """
        nuneric_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        corr_matrix = self.df[numeric_cols].corr()
        plt.figure(figsize=(12, 6))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
        plt.title("Correlation between Variables")
        plt.show()
    
    def instantiate_model(self, hp_params=None):
        """
        Allows instantiating the model with default or user-provided hyperparameters.
        hp_params is a dictionary that may include:
        - 'first_neurons': neurons in the first layer (default 64)
        - 'second_neurons': neurons in the second layer (default 32)
        - 'dropout_rate': dropout rate (default 0.3)
        - 'learning_rate': learning rate (default 0.001)

        """
        if hp_params is None:
            hp_params = {'first_neurons': 64, 'second_neurons': 32, 'dropout_rate': 0.3, 'learning_rate': 0.001}
        self.hp_params = hp_params
        print('Hyperparameters of the model:', self.hp_params)
    
    def prepare_pipeline(self, selected_features, target_column, cat_threshold=5):
        """
        Prepare the pipeline for the model using ColumnTransformer
        - For numerical variables: imputations and standardization
        - For categorical variables: imputations and one-hot encoding (<= cat_threshol for the onehot otherwise ordinal_encode)
        - Create a mapping for the target variable
        - Return X (features) and y (target) variables
        """
        df_features = self.df[selected_features]
        numeric_cols = df_features.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df_features.select_dtypes(exclude=[np.number]).columns.tolist()
        transformers = []
        if numeric_cols:
            transformers.append(('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='mean')), 
                                                        ('scaler', StandardScaler())]), numeric_cols))
        if categorical_cols:
            onehot_cols = [col for col in categorical_cols if self.df[col].nunique() <= cat_threshold]
            ordinal_cols = [col for col in categorical_cols if self.df[col].nunique() > cat_threshold]
            if onehot_cols:
                transformers.append(('cat_onehot', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                                        ('onehot', OneHotEncoder(handle_unknown='ignore'))]), onehot_cols))
            if ordinal_cols:
                transformers.append(('cat_ord', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), 
                                                        ('ordinal', OrdinalEncoder())]), ordinal_cols))
        preprocessor = ColumnTransformer(transformers=transformers)
        self.preprocessor = preprocessor

        #Prepare the target variable with mapping
        unique_targets = self.df[target_column].unique()
        self.polygon_to_idx = {poly: idx for idx, poly in enumerate(unique_targets)}
        self.idx_to_polygons = {idx: poly for idx, poly in self.polygon_to_idx.items()}
        #mapping target to numeric values
        self.df['target_numeric'] = self.df[target_column].map(self.polygon_to_idx)
        self.n_classes = len(unique_targets)

        X = self.df[selected_features].values
        y = self.df['target_numeric'].values
        return X, y
    

    def build_model(self, imput_dim):
        """
        Constructs and compile the MLP model using the hyperparameters in self.hp_params
        The end layer has self.n_classes neurons and softmax activation
        """
        model = Sequential()
        model.add(Dense(self.hp_params.get('first_neurons', 64), input_dim=input_dim, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.hp_params.get('dropout_rate', 0.3)))
        model.add(Dense(self.hp_params.get('second_neurons', 32), activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(self.hp_params.get('dropout_rate', 0.3)))
        model.add(Dense(self.hp_params.get('output_neurons', self.n_classes), activation='softmax'))
        model.compile(optimizer=Adam(learning_rate=self.hp_params.get('learning_rate', 0.001)), 
                      loss='categorical_crossentropy', 
                      metrics=['accuracy'])
        return model

    def train_model(self, selected_features, target_column, grid_search=False):
        """
        Train the model
        - Prepare the pipeline fro preprocessing
        - Transform the target to one-hot encoding
        - Split the data into train and test (stratifed by the target)
        - Create ta complete pipeline (preprocessor + model Keras Classifier)
        - (optionally) perform grid search for hyperparameters cv=3
        - Fit the model
        - Evaluate the model
        - Save the model
        """
        X, y = self.prepare_pipeline(selected_features, target_column)
        y_cat = to_categorical(y, num_classes=self.n_classes)
        X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, stratify=y, random_state=42)  #stratify by the target

        #first lets adjust the input_dim for the model
        X_train_tranformed = self.preprocessor.fit_transform(X_train)
        imput_dim = X_train_tranformed.shape[1]

        #create a wrapper for the model in Keras
        model_wrapper = KerasClassifier(build_fn=lambda: self.build_model(input_dim), 
                                        epochs=50, batch_size=32,
                                        verbose=0)
        
        #Complete pipeline: preprocessor + model
        self.full_pipeline = Pipeline(
            steps=[('preprocessor', self.preprocessor), 
                   ('model', model_wrapper)])

        if grid_search:
            param_grid = {
                #'model__first_neurons': [64, 128],
                #'model__second_neurons': [32, 64],
                #'model__dropout_rate': [0.3, 0.5],
                #'model__learning_rate': [0.001, 0.01],
                'model__epochs': [50, 100],
                'model__batch_size': [32, 64]
            }
            grid_search = GridSearchCV(estimator=self.full_pipeline, param_grid=param_grid, cv=3, scoring='accuracy')
            grid_search.fit(X_train, y_train)   
            self.full_pipeline = grid_search.best_estimator_
            print("Best parameters found by grid search:", grid_search.best_params_)

        else:   
            self.full_pipeline.fit(X_train, y_train)
        
        #Evaluate the model
        y_pred_cat  = self.full_pipeline.predict(X_test)
        y_test_labels = np.argmax(y_test, axis=1)
        acc = accuracy_score(y_test_labels, y_pred_cat)
        f1 = f1_score(y_test_labels, y_pred_cat, average='weighted')
        print(f"Accuracy: {acc:.4f}")
        print(f"F1 Score: {f1:.4f}")    

        #Save the pipeline
        joblib.dump(self.full_pipeline, 'animal_movement_pipeline.pkl')
        print("Model saved as animal_movement_pipeline.pkl")
        return acc, f1
    
    def predict_from_file(self,  csv_path, selected_features):
        """
        Predict the target variable from a new file
        - The new data must have the same columns and data types as the original data
        - Returns the predicted polygon for each row and the probability of each polygon

        """
        if self.full_pipeline is None:
            print("Model not trained yet")
            return
        new_data = pd.read_csv(csv_path)
        X_new = new_data[selected_features].values
        y_pred = self.full_pipeline.predict(X_new)
        y_pred_polygons = [self.idx_to_polygons[idx] for idx in y_pred]
        new_data['target'] = y_pred_polygons
        probs = self.full_pipeline.predict_proba(X_new)
        #new_data.to_csv('predictions.csv', index=False)
        for i, probs in enumerate(probs):
            print(f'\nReg. {i}:')
            for idx, prob in enumerate(probs):
                print(f'{self.idx_to_polygons[idx]}: {prob:.4f}')   
            print(f'Predicted polygon: {y_pred_polygons[i]}')
        return y_pred_polygons, probs
    
    def predict_polygon(self, input_features):
        """
        Predict the most probable polygon from a conditions vector
        - input_features: must be an array with the features values, and same structure as the original data
        - Returns the predicted polygon and the probability of each polygon
        """
        if self.full_pipeline is None:
            print("Model not trained yet")
            return
        X = np.array([input_features]).reshape(1, -1)
        input_transformed = self.full_pipeline.named_steps['preprocessor'].transform(X)
        probabilities = self.full_pipeline.named_steps['model'].model.predict(input_transformed)[0]
        idx = np.argmax(probabilities)
        polygon = self.idx_to_polygons[idx]
        return polygon, probabilities
    

if __name__=='__main__':
    app = AnimalMovementApp()

    #1. Enter the data in a CSV format
    csv_path = input('Enter the name of the CSV file: ')
    app.load_data('csv_path')

    #2. Report missing values
    app.report_missing_values()
    app.report_outliers()

    #3. Treat the data
    treat_method = input("Enter the strategy for treating the numeric data (mean, median): ")
    app.treat_data(num_strategy=treat_method)

    #4. Report the distribution of the data
    cat_var = input("Enter the category variable for distribution analysis (if any): ")
    cat_var = cat_var if cat_var else None
    app.report_distribution(category=cat_var)

    #5. Report the correlation between the variables
    app.report_correlation()

    #6. Instantiate the model: enter the hyperparameters (or use default configuration)
    hp_params = {
        'first_neurons': int(input("Enter the number of neurons in the first layer (default 64): ") or 64), 
        'second_neurons': int(input("Enter the number of neurons in the second layer (default 32): ") or 32),
        'dropout_rate': float(input("Enter the dropout rate (default 0.3): ") or 0.3),
        'learning_rate': float(input("Enter the learning rate (default 0.001): ") or 0.001)
        }
    app.instantiate_model(hp_params)

    #7. Train the model: enter the selected features and target column
    features_input = input("Enter the selected features separated by commas: ").split(',')
    features_input = [f.strip() for f in features_input]
    target_column = input("Enter the target column: example polygon_8 ")
    grid_choise = input("Perform grid search for hyperparameters (yes/no): ")
    acc, f1 = app.train_model(selected_features=features_input, target_column=target_column, grid_search=grid_choise.lower() == 'yes')

    #8. Predict from a new file Module
    pred_file = input("Enter the path to the new file for prediction: ")
    app.predict_from_file(pred_file, features_input)

    #9. Predict the polygon from a conditions vector
    manual_input = input("Enter the values for the selected features separated by commas: ").split(',') 
    manual_input = [float(f.strip()) for f in manual_input]
    polygon, probs = app.predict_polygon(manual_input)
    print("\nManual Prediction:")
    print(f"Predicted polygon: {polygon}")
    print(f"Probabilities: {probs}")
