import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train_basic_model():
    # Mengatur tracking URI ke localhost sesuai kriteria
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Telco_Churn_Basic")
    
    print("Memuat dataset preprocessing...")
    try:
        df = pd.read_csv('/home/kuliah/intern/dicoding/pembelajaran/ML-Ops/membangun_model/dataset_preprocessing.csv')
    except FileNotFoundError:
        print("Error: File telco_churn_preprocessing.csv tidak ditemukan!")
        return
    
    # Memisahkan fitur (X) dan target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    # Membagi data menjadi training dan testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Mengaktifkan fitur autolog MLflow sesuai syarat kriteria basic
    mlflow.autolog()
    
    # Memulai pencatatan MLflow
    with mlflow.start_run(run_name="RandomForest_Basic"):
        print("Melatih model Random Forest dasar...")
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluasi model (Autolog akan otomatis mencatat ini jika menggunakan sklearn)
        predictions = model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"Pelatihan selesai! Akurasi Model: {acc:.4f}")
        print("Data telah dicatat ke MLflow secara otomatis menggunakan autolog.")

if __name__ == "__main__":
    train_basic_model()