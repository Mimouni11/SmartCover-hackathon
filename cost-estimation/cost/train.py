"""
Script d'entraînement du modèle XGBoost pour l'estimation de coûts
"""
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import numpy as np
import os
import sys

def train_cost_model():
    """
    Entraîner le modèle XGBoost pour l'estimation de coûts
    """
    print("="*60)
    print("🚀 ENTRAÎNEMENT DU MODÈLE D'ESTIMATION DE COÛTS")
    print("="*60)
    
    # Obtenir le chemin du fichier actuel et remonter à la racine
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Chemins vers les fichiers
    data_path = os.path.join(project_root, "data", "insurance_claims.csv")
    models_dir = os.path.join(project_root, "models")
    
    # Vérifier si le dataset existe
    if not os.path.exists(data_path):
        print(f"❌ Dataset introuvable: {data_path}")
        print("📁 Placer le fichier insurance_claims.csv dans le dossier data/")
        sys.exit(1)
    
    # Charger le dataset
    print(f"\n📂 Chargement du dataset...")
    df = pd.read_csv(data_path)
    print(f"✅ Dataset chargé: {len(df)} lignes, {len(df.columns)} colonnes")
    
    # Afficher quelques statistiques
    print(f"\n📊 Statistiques du coût total:")
    print(f"   Moyenne: ${df['total_claim_amount'].mean():,.2f}")
    print(f"   Médiane: ${df['total_claim_amount'].median():,.2f}")
    print(f"   Min: ${df['total_claim_amount'].min():,.2f}")
    print(f"   Max: ${df['total_claim_amount'].max():,.2f}")
    
    print(f"\n⚙️ Feature Engineering...")
    # Feature engineering
    df['customer_loyalty_score'] = df['months_as_customer'] / 100
    df['injury_severity'] = df['bodily_injuries'] * 1.5
    df['complexity_score'] = (
        df['number_of_vehicles_involved'] * 0.3 + 
        df['bodily_injuries'] * 0.5
    )
    df['policy_coverage_ratio'] = df['umbrella_limit'] / (df['policy_annual_premium'] + 1)
    df['night_incident'] = ((df['incident_hour_of_the_day'] >= 0) & 
                             (df['incident_hour_of_the_day'] <= 6)).astype(int)
    
    # Features et target
    feature_cols = [
        'months_as_customer', 'age', 'policy_deductable', 
        'policy_annual_premium', 'umbrella_limit', 'capital-gains',
        'capital-loss', 'incident_hour_of_the_day', 'number_of_vehicles_involved',
        'bodily_injuries', 'witnesses', 'customer_loyalty_score',
        'injury_severity', 'complexity_score', 'policy_coverage_ratio',
        'night_incident'
    ]
    
    print(f"✅ {len(feature_cols)} features créées")
    
    X = df[feature_cols]
    y = df['total_claim_amount']
    
    # Split train/test
    print(f"\n🔀 Split des données (80% train, 20% test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"   Train: {len(X_train)} échantillons")
    print(f"   Test: {len(X_test)} échantillons")
    
    # Normalisation
    print(f"\n📐 Normalisation des features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost avec GridSearch (version rapide pour le hackathon)
    print(f"\n🤖 Configuration du modèle XGBoost...")
    
    # Paramètres optimisés pour un entraînement rapide
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 7],
        'learning_rate': [0.05, 0.1],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"🔍 Recherche des meilleurs hyperparamètres...")
    print(f"   {len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['learning_rate']) * len(param_grid['subsample']) * len(param_grid['colsample_bytree'])} combinaisons à tester")
    print(f"   Validation croisée: 5 folds")
    
    grid_search = GridSearchCV(
        xgb_model, 
        param_grid, 
        cv=5, 
        scoring='neg_mean_absolute_error',
        n_jobs=-1, 
        verbose=1
    )
    
    print(f"\n⏳ Entraînement en cours (cela peut prendre 2-3 minutes)...")
    grid_search.fit(X_train_scaled, y_train)
    
    # Meilleur modèle
    best_model = grid_search.best_estimator_
    print(f"\n✅ Meilleurs hyperparamètres trouvés:")
    for param, value in grid_search.best_params_.items():
        print(f"   {param}: {value}")
    
    # Évaluation sur le test set
    print(f"\n📊 Évaluation sur le test set...")
    y_pred = best_model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    print(f"\n{'='*60}")
    print(f"🎯 PERFORMANCE DU MODÈLE")
    print(f"{'='*60}")
    print(f"   MAE (Mean Absolute Error):  ${mae:,.2f}")
    print(f"   RMSE (Root Mean Squared Error): ${rmse:,.2f}")
    print(f"   R² Score: {r2:.4f} ({r2*100:.2f}%)")
    print(f"   MAPE (Mean Absolute % Error): {mape:.2f}%")
    print(f"{'='*60}")
    
    # Afficher quelques prédictions exemples
    print(f"\n📝 Exemples de prédictions:")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        error = abs(actual - predicted)
        error_pct = (error / actual) * 100
        print(f"   Réel: ${actual:>8,.2f} | Prédit: ${predicted:>8,.2f} | Erreur: ${error:>7,.2f} ({error_pct:.1f}%)")
    
    # Feature importance
    print(f"\n🔍 Top 10 des features les plus importantes:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        bar_length = int(row['importance'] * 50)
        bar = '█' * bar_length
        print(f"   {row['feature']:25s} {bar} {row['importance']:.4f}")
    
    # Créer le dossier models s'il n'existe pas
    os.makedirs(models_dir, exist_ok=True)
    
    # Sauvegarder les modèles
    model_path = os.path.join(models_dir, "cost_model.pkl")
    scaler_path = os.path.join(models_dir, "cost_scaler.pkl")
    
    print(f"\n💾 Sauvegarde des modèles...")
    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"   ✅ Modèle sauvegardé: {model_path}")
    print(f"   ✅ Scaler sauvegardé: {scaler_path}")
    
    print(f"\n{'='*60}")
    print(f"🎉 ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS!")
    print(f"{'='*60}\n")
    
    return best_model, scaler, {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape
    }


if __name__ == "__main__":
    try:
        train_cost_model()
    except KeyboardInterrupt:
        print("\n\n⚠️ Entraînement interrompu par l'utilisateur")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
