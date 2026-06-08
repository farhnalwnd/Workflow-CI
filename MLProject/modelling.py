import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

def train_basic_model():
    # Cek jika berjalan di dalam mlflow run
    is_mlflow_run = "MLFLOW_RUN_ID" in os.environ
    
    if not is_mlflow_run:
        # Mengatur tracking URI ke localhost jika tidak di CI
        if not os.environ.get("GITHUB_ACTIONS"):
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
    
    # Mengaktifkan fitur autolog MLflow
    mlflow.autolog()
    
    if is_mlflow_run:
        print("Melatih model di bawah MLflow run...")
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        print(f"Pelatihan selesai! MSE: {mse:.2f}, MAE: {mae:.2f}, R2: {r2:.4f}")
        
        # Simpan model secara fisik untuk keperluan artifact upload pada CI/CD
        import shutil
        if os.path.exists("model_output"):
            shutil.rmtree("model_output")
        os.makedirs("model_output", exist_ok=True)
        mlflow.sklearn.save_model(model, "model_output", serialization_format="cloudpickle")
        print("Model berhasil disimpan ke folder model_output.")
    else:
        # Memulai pencatatan MLflow secara mandiri jika dijalankan langsung
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
            
            # Simpan model secara fisik
            import shutil
            if os.path.exists("model_output"):
                shutil.rmtree("model_output")
            os.makedirs("model_output", exist_ok=True)
            mlflow.sklearn.save_model(model, "model_output", serialization_format="cloudpickle")
            print("Model berhasil disimpan.")

if __name__ == "__main__":
    train_basic_model()
