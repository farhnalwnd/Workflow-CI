import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def train_basic_model():
    # Mengatur tracking URI ke localhost sesuai kriteria
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Insurance_Charges_Basic")
    
    print("Memuat dataset preprocessing...")
    try:
        df = pd.read_csv('dataset_preprocessing.csv')
    except FileNotFoundError:
        print("Error: File dataset_preprocessing.csv tidak ditemukan!")
        return
    
    # Memisahkan fitur (X) dan target (y)
    X = df.drop('charges', axis=1)
    y = df['charges']
    
    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mengaktifkan fitur autolog MLflow sesuai syarat kriteria basic
    mlflow.autolog()
    
    # Memulai pencatatan MLflow
    with mlflow.start_run(run_name="RandomForest_Basic"):
        print("Melatih model Random Forest Regressor dasar...")
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi model
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Pelatihan selesai! MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        # Simpan model secara fisik untuk keperluan artifact upload pada CI/CD
        os.makedirs("model_output", exist_ok=True)
        mlflow.sklearn.save_model(model, "model_output", serialization_format="cloudpickle")
        print("Model berhasil disimpan ke folder model_output.")

if __name__ == "__main__":
    train_basic_model()
