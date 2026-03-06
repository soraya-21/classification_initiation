import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(page_title="Classification Cardiovasculaire", layout="wide")
st.title("Analyse Cardiovasculaire - Classification Initiation")

# ============================================================================
# CHARGEMENT ET PRÉPARATION DES DONNÉES
# ============================================================================
@st.cache_data
def load_and_prepare_data():
    """Charge et prépare les données complètes"""
    path_local = "/home/scodo/.cache/kagglehub/datasets/sulianova/cardiovascular-disease-dataset/versions/1/cardio_train.csv"
    data = pd.read_csv(path_local, sep=";")
    
    # Nettoyage basique
    data = data.dropna()
    data = data.drop_duplicates(ignore_index=True)
    data = data.drop(columns=['id'], errors='ignore')
    
    # IMC
    data['imc'] = data['weight'] / ((data['height'] / 100) ** 2)
    
    # Pression plausible
    def pression_plausible(sys, dia):
        if sys <= dia:
            return False
        diff = sys - dia
        if diff < 20 or diff > 80:
            return False
        ratio = sys / dia
        return 1.2 <= ratio <= 1.8
    
    data['pression_plausible'] = data.apply(
        lambda row: pression_plausible(row['ap_hi'], row['ap_lo']), axis=1
    )
    
    # Corriger les pressions implausibles
    pression_implausible_mask = ~data['pression_plausible']
    if pression_implausible_mask.sum() > 0:
        mean_ap_hi = data[data['pression_plausible']]['ap_hi'].mean()
        mean_ap_lo = data[data['pression_plausible']]['ap_lo'].mean()
        data.loc[pression_implausible_mask, 'ap_hi'] = int(round(mean_ap_hi))
        data.loc[pression_implausible_mask, 'ap_lo'] = int(round(mean_ap_lo))
    
    # Features engineering
    data['pulse_pressure'] = data['ap_hi'] - data['ap_lo']
    data['tension_ratio'] = data['ap_hi'] / data['ap_lo']
    data['age_ap_hi_interaction'] = data['age'] * data['ap_hi'] / 100
    data['imc_age_interaction'] = data['imc'] * data['age'] / 100
    data['weight_age_ratio'] = data['weight'] / (data['age'] / 10)
    
    scaler_risk = MinMaxScaler()
    risk_features = ['ap_hi', 'cholesterol', 'age', 'imc']
    data_risk_scaled = scaler_risk.fit_transform(data[risk_features])
    data['cardio_risk_score'] = data_risk_scaled.mean(axis=1)
    
    data['hypertension_stage'] = 0
    data.loc[(data['ap_hi'] >= 130) | (data['ap_lo'] >= 80), 'hypertension_stage'] = 1
    data.loc[(data['ap_hi'] >= 140) | (data['ap_lo'] >= 90), 'hypertension_stage'] = 2
    
    data['high_risk_profile'] = ((data['age'] > 55*365) & (data['cholesterol'] > 1)).astype(int)
    data['height_deviation'] = data.groupby('gender')['height'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    
    return data

@st.cache_resource
def train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test):
    """Entraîne les modèles et retourne les résultats"""
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(random_state=42, max_iter=1000),
            'scaled': True
        },
        'Random Forest': {
            'model': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
            'scaled': False
        },
        'Gradient Boosting': {
            'model': GradientBoostingClassifier(n_estimators=100, random_state=42),
            'scaled': False
        }
    }
    
    results = {}
    trained_models = {}
    
    for model_name, config in models.items():
        X_train_data = X_train_scaled if config['scaled'] else X_train
        X_test_data = X_test_scaled if config['scaled'] else X_test
        
        # Cross-validation
        cv_scores = cross_validate(
            config['model'], X_train_data, y_train,
            cv=5,
            scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
            return_train_score=False
        )
        
        # Entraînement
        config['model'].fit(X_train_data, y_train)
        
        # Prédictions
        y_pred = config['model'].predict(X_test_data)
        y_pred_proba = config['model'].predict_proba(X_test_data)[:, 1]
        
        results[model_name] = {
            'cv_accuracy': cv_scores['test_accuracy'].mean(),
            'cv_accuracy_std': cv_scores['test_accuracy'].std(),
            'cv_f1': cv_scores['test_f1'].mean(),
            'cv_roc_auc': cv_scores['test_roc_auc'].mean(),
            'cv_roc_auc_std': cv_scores['test_roc_auc'].std(),
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_f1': f1_score(y_test, y_pred),
            'test_roc_auc': roc_auc_score(y_test, y_pred_proba),
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        trained_models[model_name] = {
            'model': config['model'],
            'scaled': config['scaled']
        }
    
    return results, trained_models

# Charger les données
data = load_and_prepare_data()

# Préparer X et y
TARGET = 'cardio'
X = data.drop(columns=[TARGET, 'pression_plausible', 'weight', 'height', 'ap_hi', 'ap_lo'])
y = data[TARGET]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Entraîner les modèles
results, trained_models = train_models(X_train, X_test, X_train_scaled, X_test_scaled, y_train, y_test)

# ============================================================================
# INTERFACE STREAMLIT
# ============================================================================

# Menu de navigation
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Donnees", 
    "Comparaison Modeles", 
    "Performances",
    "Matrices & ROC",
    "Seuil Classification",
    "Infos"
])

# ============================================================================
# TAB 1: DONNÉES
# ============================================================================
with tab1:
    st.header("Aperçu des Données")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Nombre d'échantillons", len(data))
    with col2:
        st.metric("Nombre de features", X.shape[1])
    with col3:
        st.metric("Ratio train/test", f"{len(X_train)}/{len(X_test)}")
    
    st.subheader("Distribution de la cible")
    class_dist = y.value_counts()
    fig = px.bar(x=['Sain (0)', 'Malade (1)'], y=class_dist.values, 
                 labels={'y': 'Nombre', 'x': 'Classe'},
                 color=['#2ecc71', '#e74c3c'])
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Statistiques des features")
    st.dataframe(X.describe(), use_container_width=True)
    
    st.subheader("Liste des 17 features")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**8 features originales:**")
        st.write(", ".join(X.columns[:8]))
    with col2:
        st.write("**9 features engineered:**")
        st.write(", ".join(X.columns[8:]))

# ============================================================================
# TAB 2: COMPARAISON MODÈLES
# ============================================================================
with tab2:
    st.header("Comparaison des Modèles")
    
    comparison_df = pd.DataFrame({
        'Modèle': list(results.keys()),
        'CV Accuracy': [results[m]['cv_accuracy'] for m in results.keys()],
        'CV F1': [results[m]['cv_f1'] for m in results.keys()],
        'CV ROC-AUC': [results[m]['cv_roc_auc'] for m in results.keys()],
        'Test Accuracy': [results[m]['test_accuracy'] for m in results.keys()],
        'Test F1': [results[m]['test_f1'] for m in results.keys()],
        'Test ROC-AUC': [results[m]['test_roc_auc'] for m in results.keys()]
    })
    
    comparison_df = comparison_df.sort_values('Test ROC-AUC', ascending=False)
    st.dataframe(comparison_df, use_container_width=True)
    
    best_model_name = comparison_df.iloc[0]['Modèle']
    st.success(f"Meilleur modele: **{best_model_name}** (ROC-AUC: {comparison_df.iloc[0]['Test ROC-AUC']:.4f})")

# ============================================================================
# TAB 3: PERFORMANCES
# ============================================================================
with tab3:
    st.header("Analyse des Performances")
    
    # Afficher les métriques en colonnes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Test Accuracy")
        fig = px.bar(comparison_df, x='Modèle', y='Test Accuracy',
                     color='Test Accuracy', color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Test F1-Score")
        fig = px.bar(comparison_df, x='Modèle', y='Test F1',
                     color='Test F1', color_continuous_scale='Plasma')
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        st.subheader("Test ROC-AUC")
        fig = px.bar(comparison_df, x='Modèle', y='Test ROC-AUC',
                     color='Test ROC-AUC', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 4: MATRICES & COURBES ROC
# ============================================================================
with tab4:
    st.header("Matrices de Confusion & Courbes ROC")
    
    selected_model = st.selectbox("Sélectionner un modèle", results.keys())
    best_model = trained_models[selected_model]['model']
    best_model_scaled = trained_models[selected_model]['scaled']
    best_model_data = results[selected_model]
    
    col1, col2 = st.columns(2)
    
    # Matrice de confusion
    with col1:
        st.subheader(f"Matrice de Confusion - {selected_model}")
        cm = confusion_matrix(y_test, best_model_data['y_pred'])
        
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, cmap='Blues', aspect='auto')
        ax.set_xlabel('Prédiction', fontsize=11)
        ax.set_ylabel('Réalité', fontsize=11)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Sain', 'Malade'])
        ax.set_yticklabels(['Sain', 'Malade'])
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", 
                       color="black", fontsize=14, fontweight='bold')
        
        plt.colorbar(im, ax=ax)
        st.pyplot(fig, use_container_width=True)
    
    # Rapport de classification
    with col2:
        st.subheader(f"Rapport de Classification")
        report = classification_report(y_test, best_model_data['y_pred'], 
                                      target_names=['Sain', 'Malade'],
                                      output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df, use_container_width=True)
    
    # Courbes ROC pour tous les modèles
    st.subheader("Courbes ROC - Comparaison")
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    for i, (model_name, result) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', 
               linewidth=2.5, color=colors[i])
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
    ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=11)
    ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=11)
    ax.set_title('Courbes ROC - Comparaison des Modèles', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    st.pyplot(fig, use_container_width=True)

# ============================================================================
# TAB 5: SEUIL DE CLASSIFICATION
# ============================================================================
with tab5:
    st.header("Optimisation du Seuil de Classification")
    
    # Sélectionner le meilleur modèle
    best_model_name = comparison_df.iloc[0]['Modèle']
    best_model = trained_models[best_model_name]['model']
    best_model_scaled = trained_models[best_model_name]['scaled']
    
    X_test_data = X_test_scaled if best_model_scaled else X_test
    y_probs = best_model.predict_proba(X_test_data)[:, 1]
    
    st.subheader(f"Modèle: {best_model_name}")
    
    # Slider pour ajuster le seuil
    threshold = st.slider("Ajuster le seuil de classification", 
                         min_value=0.0, max_value=1.0, value=0.5, step=0.05)
    
    col1, col2 = st.columns(2)
    
    # Prédictions avec le seuil personnalisé
    y_pred_custom = (y_probs >= threshold).astype(int)
    
    with col1:
        st.metric("Seuil actuel", f"{threshold:.2f}")
        st.metric("Accuracy", f"{accuracy_score(y_test, y_pred_custom):.4f}")
        st.metric("Precision", f"{precision_score(y_test, y_pred_custom, zero_division=0):.4f}")
    
    with col2:
        st.metric("Recall", f"{recall_score(y_test, y_pred_custom, zero_division=0):.4f}")
        st.metric("F1-Score", f"{f1_score(y_test, y_pred_custom, zero_division=0):.4f}")
        st.metric("ROC-AUC", f"{roc_auc_score(y_test, y_pred_custom):.4f}")
    
    # Matrice de confusion avec le seuil personnalisé
    st.subheader(f"Matrice de Confusion (Seuil: {threshold:.2f})")
    cm_custom = confusion_matrix(y_test, y_pred_custom)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_custom, cmap='Blues', aspect='auto')
    ax.set_xlabel('Prédiction', fontsize=11)
    ax.set_ylabel('Réalité', fontsize=11)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Sain', 'Malade'])
    ax.set_yticklabels(['Sain', 'Malade'])
    
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm_custom[i, j], ha="center", va="center", 
                   color="black", fontsize=14, fontweight='bold')
    
    plt.colorbar(im, ax=ax)
    st.pyplot(fig, use_container_width=True)
    
    # Distribution des probabilités
    st.subheader("Distribution des Probabilites")
    fig = px.histogram(
        pd.DataFrame({'Probabilite': y_probs, 'Classe': y_test.map({0: 'Sain', 1: 'Malade'})}),
        x='Probabilite', color='Classe', nbins=30,
        color_discrete_map={'Sain': '#2ecc71', 'Malade': '#e74c3c'},
        barmode='overlay'
    )
    
    # Ajouter la ligne du seuil
    fig.add_shape(
        type="line",
        x0=threshold, y0=0,
        x1=threshold, y1=max(len(y_test) // 10, 100),
        line=dict(color="orange", width=2, dash="dash"),
        name="Seuil"
    )
    
    fig.update_layout(
        title_text=f"Distribution des Probabilites (Ligne orange = Seuil: {threshold:.2f})",
        hovermode='x'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 6: INFOS
# ============================================================================
# ============================================================================
# TAB 6: INFOS
# ============================================================================
with tab6:
    st.header("Comprendre les Features Engineered")
    
    st.markdown("""
    ## Vue d'ensemble des Features Engineering
    
    Les 9 features créées ne sont pas aléatoires. Chacune répond à une logique médicale ou statistique précise.
    """)
    
    # Créer des sous-onglets pour chaque catégorie de features
    feature_tabs = st.tabs([
        "Pression Arterielle",
        "Interactions Multiplicatives",
        "Score Agrege",
        "Normalisation Genre",
        "Profil Haut Risque"
    ])
    
    # TAB 1: PRESSION ARTÉRIELLE
    with feature_tabs[0]:
        st.subheader("1. Features de Pression Arterielle")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### pulse_pressure
            **Formule**: ap_hi - ap_lo
            
            **Médecine**: La différence entre systolique et diastolique est un marqueur majeur de **la rigidité artérielle**.
            
            **Clinique**: Une pression pulsée élevée signifie que les grosses artères sont moins élastiques. C'est un prédicteur d'AVC **indépendant** de la tension moyenne.
            
            **Exemple**:
            - Tension normale (120/80) → PP = 40
            - Hypertension (160/90) → PP = 70
            - Rigidité (180/60) → PP = 120
            
            ### tension_ratio
            **Formule**: ap_hi / ap_lo
            
            **Médecine**: Ce ratio montre la **proportionnalité** entre les deux pressions.
            
            **Clinique**: Un déséquilibre peut signaler une hypertension spécifique (systolique isolée = mauvais contrôle).
            
            **Normale**: ~1.5
            **Alerte**: > 1.8
            """)
        
        with col2:
            # Visualisation des distributions
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            # pulse_pressure
            ax = axes[0, 0]
            ax.hist(data['pulse_pressure'], bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution: pulse_pressure', fontweight='bold')
            ax.set_xlabel('Pression Pulsée (mmHg)')
            ax.set_ylabel('Fréquence')
            ax.axvline(data['pulse_pressure'].mean(), color='darkred', linestyle='--', 
                      linewidth=2, label=f'Moyenne: {data["pulse_pressure"].mean():.1f}')
            ax.legend()
            
            # tension_ratio
            ax = axes[0, 1]
            ax.hist(data['tension_ratio'], bins=50, color='#3498db', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution: tension_ratio', fontweight='bold')
            ax.set_xlabel('Ratio (ap_hi / ap_lo)')
            ax.set_ylabel('Fréquence')
            ax.axvline(data['tension_ratio'].mean(), color='darkblue', linestyle='--', 
                      linewidth=2, label=f'Moyenne: {data["tension_ratio"].mean():.2f}')
            ax.legend()
            
            # hypertension_stage distribution
            ax = axes[1, 0]
            stage_counts = data['hypertension_stage'].value_counts().sort_index()
            colors_stages = ['#2ecc71', '#f39c12', '#e74c3c']
            ax.bar(stage_counts.index, stage_counts.values, color=colors_stages, edgecolor='black')
            ax.set_title('Distribution: hypertension_stage (OMS)', fontweight='bold')
            ax.set_xlabel('Stade d\'Hypertension')
            ax.set_ylabel('Nombre de Patients')
            ax.set_xticks([0, 1, 2])
            ax.set_xticklabels(['Normal', 'Élevée', 'Hypertension'])
            for i, v in enumerate(stage_counts.values):
                ax.text(i, v+50, str(v), ha='center', fontweight='bold')
            
            # Corrélation avec cardio
            ax = axes[1, 1]
            corr_features = ['pulse_pressure', 'tension_ratio', 'hypertension_stage']
            corr_values = [
                abs(data['pulse_pressure'].corr(y)),
                abs(data['tension_ratio'].corr(y)),
                abs(data['hypertension_stage'].corr(y))
            ]
            ax.barh(corr_features, corr_values, color=['#e74c3c', '#3498db', '#f39c12'], edgecolor='black')
            ax.set_title('Corrélation avec la Cible (cardio)', fontweight='bold')
            ax.set_xlabel('|Corrélation|')
            ax.set_xlim([0, 0.5])
            for i, v in enumerate(corr_values):
                ax.text(v+0.01, i, f'{v:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown("""
        ### hypertension_stage
        **Définition**: Catégories basées sur l'OMS
        - **Stade 0** (Normal): ap_hi < 130 ET ap_lo < 80
        - **Stade 1** (Élevée): ap_hi ≥ 130 OU ap_lo ≥ 80
        - **Stade 2** (Hypertension): ap_hi ≥ 140 OU ap_lo ≥ 90
        
        **Pourquoi**: Les modèles aiment les **paliers discrets**. En transformant une donnée continue "bruitée" en catégories, tu crées un signal clair :
        - "Ici, on franchit un seuil de danger clinique"
        - Les arbres de décision adorent ces paliers
        """)
    
    # TAB 2: INTERACTIONS
    with feature_tabs[1]:
        st.subheader("2. Features d'Interactions Multiplicatives")
        
        st.markdown("""
        ### Principe Médical
        **Le risque cardiovasculaire N'EST PAS additif, il est MULTIPLICATIF.**
        
        Exemple:
        - Tension 150 à 30 ans → Grave ⚠️
        - Tension 150 à 70 ans → **Critique** 🚨
        
        Le même symptôme n'a pas le même impact selon le contexte.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### age_ap_hi_interaction
            **Formule**: age × ap_hi / 100
            
            **Logique**: L'impact de la tension augmente avec l'âge. Cette variable **explose** uniquement quand:
            - L'âge est élevé ET
            - La tension est élevée
            
            **Exemple**:
            - 30 ans, 120 mmHg → 36
            - 70 ans, 120 mmHg → 84
            - 70 ans, 160 mmHg → 112
            """)
        
        with col2:
            st.markdown("""
            ### imc_age_interaction
            **Formule**: imc × age / 100
            
            **Logique**: Le surpoids est plus **dur à supporter** pour le système cardiovasculaire d'une personne âgée.
            
            **Médecine**: Cette variable capture la "fatigue accumulée du cœur" face à la masse corporelle au fil des années.
            
            **Exemple**:
            - 30 ans, IMC 25 → 7.5
            - 70 ans, IMC 25 → 17.5
            - 70 ans, IMC 35 → 24.5
            """)
        
        # Visualisations d'interactions
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # age_ap_hi_interaction
        ax = axes[0]
        scatter = ax.scatter(data['age'], data['ap_hi'], c=data['age_ap_hi_interaction'], 
                            cmap='RdYlGn_r', alpha=0.6, s=20)
        ax.set_xlabel('Âge (jours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Tension Systolique (mmHg)', fontsize=11, fontweight='bold')
        ax.set_title('age_ap_hi_interaction\n(Couleur = valeur de l\'interaction)', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Valeur d\'interaction', fontweight='bold')
        
        # imc_age_interaction
        ax = axes[1]
        scatter = ax.scatter(data['age'], data['imc'], c=data['imc_age_interaction'], 
                            cmap='YlOrRd', alpha=0.6, s=20)
        ax.set_xlabel('Âge (jours)', fontsize=11, fontweight='bold')
        ax.set_ylabel('IMC (kg/m²)', fontsize=11, fontweight='bold')
        ax.set_title('imc_age_interaction\n(Couleur = valeur de l\'interaction)', fontweight='bold')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Valeur d\'interaction', fontweight='bold')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.markdown("""
        ### weight_age_ratio
        **Formule**: weight / (age / 10)
        
        **Logique**: Rapport du poids à l'âge. Indicateur de charge physiologique relative.
        
        Une personne lourd jeune → moins grave
        Une personne lourd âgé → plus grave selon le ratio
        """)
    
    # TAB 3: SCORE AGRÉGÉ
    with feature_tabs[2]:
        st.subheader("3. Cardio Risk Score (Feature Agregee)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### cardio_risk_score
            **Formule**: 
            ```
            risk_features = [ap_hi, cholesterol, age, imc]
            score = mean(MinMaxScaler(risk_features))
            ```
            
            **Principe**: Agrégation multidimensionnelle
            
            **Avantage**: Crée une "super-feature" qui capture la **santé globale** du patient.
            
            **Valeur**: Entre 0 (très sain) et 1 (très à risque)
            
            **Si score élevé** → Forte probabilité cardio=1
            """)
        
        with col2:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            # Distribution
            ax = axes[0, 0]
            ax.hist(data['cardio_risk_score'], bins=50, color='#9b59b6', alpha=0.7, edgecolor='black')
            ax.set_title('Distribution: cardio_risk_score', fontweight='bold')
            ax.set_xlabel('Score (0-1)')
            ax.set_ylabel('Fréquence')
            ax.axvline(data['cardio_risk_score'].mean(), color='purple', linestyle='--', 
                      linewidth=2, label=f'Moyenne: {data["cardio_risk_score"].mean():.2f}')
            ax.legend()
            
            # Box plot par classe cible
            ax = axes[0, 1]
            data_temp = pd.DataFrame({
                'Score': data['cardio_risk_score'],
                'Classe': y.map({0: 'Sain', 1: 'Malade'})
            })
            data_temp.boxplot(column='Score', by='Classe', ax=ax)
            ax.set_title('cardio_risk_score par Classe', fontweight='bold')
            ax.set_xlabel('Classe')
            ax.set_ylabel('Score')
            plt.sca(ax)
            plt.xticks([1, 2], ['Sain', 'Malade'])
            
            # Corrélation avec cardio
            ax = axes[1, 0]
            corr = abs(data['cardio_risk_score'].corr(y))
            ax.barh(['cardio_risk_score'], [corr], color='#9b59b6', edgecolor='black', height=0.5)
            ax.set_xlim([0, 0.5])
            ax.set_title('Corrélation avec Cible', fontweight='bold')
            ax.set_xlabel('|Corrélation|')
            ax.text(corr+0.01, 0, f'{corr:.3f}', va='center', fontweight='bold')
            
            # Composantes du score
            ax = axes[1, 1]
            components = ['ap_hi', 'cholesterol', 'age', 'imc']
            corr_components = [
                abs(data['ap_hi'].corr(y)),
                abs(data['cholesterol'].corr(y)),
                abs(data['age'].corr(y)),
                abs(data['imc'].corr(y))
            ]
            ax.barh(components, corr_components, color=['#e74c3c', '#3498db', '#f39c12', '#2ecc71'], 
                   edgecolor='black')
            ax.set_title('Composantes: Corrélation avec cardio', fontweight='bold')
            ax.set_xlabel('|Corrélation|')
            for i, v in enumerate(corr_components):
                ax.text(v+0.005, i, f'{v:.3f}', va='center', fontweight='bold', fontsize=9)
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # TAB 4: NORMALISATION GENRE
    with feature_tabs[3]:
        st.subheader("4. Normalisation par Genre (height_deviation)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### height_deviation
            **Formule**:
            ```
            height_deviation = (height - mean_by_gender) / std_by_gender
            ```
            
            **Problème corrigé**: 
            - Les hommes sont **statistiquement plus grands** que les femmes
            - Si tu donnes la hauteur brute au modèle, il risque d'associer "grande taille" à "homme" plutôt qu'au risque
            
            **Solution**: Calculer l'écart à la moyenne **par genre**
            
            **Résultat**: Une variable qui répond à la question:
            - "Est-ce que cette personne est grande **par rapport aux gens de son sexe**?"
            
            **Avantage**: 
            ✅ Plus équitable
            ✅ Scientifiquement plus juste
            ✅ Réduit les biais de genre
            """)
        
        with col2:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            # Distribution brute
            ax = axes[0, 0]
            for gender in [1, 2]:
                mask = data['gender'] == gender
                ax.hist(data[mask]['height'], bins=30, alpha=0.6, 
                       label=f"Genre {gender}", edgecolor='black')
            ax.set_title('Distribution: height (brute)', fontweight='bold')
            ax.set_xlabel('Hauteur (cm)')
            ax.set_ylabel('Fréquence')
            ax.legend()
            
            # Distribution normalisée
            ax = axes[0, 1]
            for gender in [1, 2]:
                mask = data['gender'] == gender
                ax.hist(data[mask]['height_deviation'], bins=30, alpha=0.6, 
                       label=f"Genre {gender}", edgecolor='black')
            ax.set_title('Distribution: height_deviation (normalisée)', fontweight='bold')
            ax.set_xlabel('Height Deviation (Z-score)')
            ax.set_ylabel('Fréquence')
            ax.legend()
            
            # Statistiques par genre
            ax = axes[1, 0]
            stats_height = data.groupby('gender')['height'].agg(['mean', 'std'])
            stats_height.plot(kind='bar', ax=ax, color=['#3498db', '#e74c3c'], edgecolor='black')
            ax.set_title('Statistiques: height par genre', fontweight='bold')
            ax.set_ylabel('Valeur')
            ax.set_xticklabels(['Genre 1 (F)', 'Genre 2 (M)'], rotation=0)
            
            # Corrélation avec cardio
            ax = axes[1, 1]
            features_height = ['height', 'height_deviation']
            corr_values = [
                abs(data['height'].corr(y)),
                abs(data['height_deviation'].corr(y))
            ]
            ax.barh(features_height, corr_values, color=['#95a5a6', '#3498db'], edgecolor='black')
            ax.set_title('Height: brute vs normalisée', fontweight='bold')
            ax.set_xlabel('|Corrélation avec cardio|')
            for i, v in enumerate(corr_values):
                ax.text(v+0.002, i, f'{v:.3f}', va='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    # TAB 5: PROFIL HAUT RISQUE
    with feature_tabs[4]:
        st.subheader("5. Profil de Haut Risque (Regle Metier)")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("""
            ### high_risk_profile
            **Formule**:
            ```
            high_risk_profile = (age > 55 ans) AND (cholesterol > 1)
            ```
            
            **Type**: Règle métier (Rule-based Feature)
            
            **Logique Médicale**: 
            En cardiologie, on sait que le croisement:
            - Âge > 55 ans
            - Cholestérol élevé
            
            ...marque un **point de bascule** critique.
            
            **Avantage pour le modèle**:
            Offre un "**raccourci décisionnel**". Au lieu de calculer plusieurs branches dans un arbre:
            
            ```
            if high_risk_profile == 1:
                then cardio = 1 (forte probabilité)
            ```
            
            **Valeur**: Binaire (0 ou 1)
            """)
        
        with col2:
            fig, axes = plt.subplots(2, 2, figsize=(10, 8))
            
            # Distribution
            ax = axes[0, 0]
            risk_counts = data['high_risk_profile'].value_counts()
            colors_risk = ['#2ecc71', '#e74c3c']
            ax.bar(['Bas Risque', 'Haut Risque'], 
                  [risk_counts[0], risk_counts[1]], 
                  color=colors_risk, edgecolor='black')
            ax.set_title('Distribution: high_risk_profile', fontweight='bold')
            ax.set_ylabel('Nombre de Patients')
            for i, (label, count) in enumerate(zip(['Bas Risque', 'Haut Risque'], 
                                                     [risk_counts[0], risk_counts[1]])):
                ax.text(i, count+50, str(count), ha='center', fontweight='bold')
            
            # Corrélation avec cardio
            ax = axes[0, 1]
            corr = abs(data['high_risk_profile'].corr(y))
            ax.barh(['high_risk_profile'], [corr], color='#e74c3c', edgecolor='black', height=0.5)
            ax.set_xlim([0, 0.5])
            ax.set_title('Corrélation avec Cible', fontweight='bold')
            ax.set_xlabel('|Corrélation|')
            ax.text(corr+0.01, 0, f'{corr:.3f}', va='center', fontweight='bold')
            
            # Taux de maladie par profil
            ax = axes[1, 0]
            disease_rate = data.groupby('high_risk_profile')[TARGET].mean()
            ax.bar(['Bas Risque', 'Haut Risque'], 
                  [disease_rate[0], disease_rate[1]], 
                  color=['#2ecc71', '#e74c3c'], edgecolor='black')
            ax.set_title('Taux de Maladie par Profil', fontweight='bold')
            ax.set_ylabel('Proportion de Malades')
            ax.set_ylim([0, 1])
            for i, v in enumerate([disease_rate[0], disease_rate[1]]):
                ax.text(i, v+0.02, f'{v:.1%}', ha='center', fontweight='bold')
            
            # Critères de la règle
            ax = axes[1, 1]
            age_threshold = 55 * 365  # convertir en jours
            criteria = {
                'Age > 55': (data['age'] > age_threshold).sum(),
                'Cholesterol > 1': (data['cholesterol'] > 1).sum(),
                'BOTH': ((data['age'] > age_threshold) & (data['cholesterol'] > 1)).sum()
            }
            ax.bar(criteria.keys(), criteria.values(), color=['#3498db', '#f39c12', '#e74c3c'], 
                  edgecolor='black')
            ax.set_title('Critères de Haut Risque', fontweight='bold')
            ax.set_ylabel('Nombre de Patients')
            for i, (k, v) in enumerate(criteria.items()):
                ax.text(i, v+50, str(v), ha='center', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("""
    ---
    ## 📊 Résumé: Pourquoi Ces Features?
    
    | Feature | Type | Raison |
    |---------|------|--------|
    | **pulse_pressure** | Dérivée | Marqueur d'élasticité artérielle |
    | **tension_ratio** | Dérivée | Détecte déséquilibre systolique/diastolique |
    | **hypertension_stage** | Catégorique | Paliers cliniques discrets |
    | **age_ap_hi_interaction** | Multiplicatif | Risque croît avec âge + tension |
    | **imc_age_interaction** | Multiplicatif | Charge cumulative: surpoids + âge |
    | **weight_age_ratio** | Ratio | Charge physiologique relative |
    | **cardio_risk_score** | Agrégée | Super-feature: santé globale |
    | **high_risk_profile** | Règle | Raccourci décisionnel médical |
    | **height_deviation** | Normalisée | Équité par genre, Z-score |
    
    ---
    
    ## 🎯 Impact Attendu
    
    Ces 9 features **remplacent** 4 features brutes (weight, height, ap_hi, ap_lo) par des représentations plus riches.
    
    ✅ **Avant**: 12 features brutes
    ✅ **Après**: 8 features originales + 9 engineered = **17 features optimisées**
    
    Le modèle bénéficie de:
    - Signaux cliniquement pertinents
    - Interprétabilité améliorée
    - Meilleure séparation des classes
    - Réduction du bruit
    """)
