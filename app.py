import streamlit as st
import pandas as pd
import joblib
import re

# Configuration de la page
st.set_page_config(page_title="Analyseur d'Assurances", page_icon="🛡️", layout="wide")

# Chargement des données et des modèles
@st.cache_data
def load_data():
    df = pd.read_csv("data/processed/dataset_propre_et_traduit.csv")
    return df

@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load('models/tfidf_vectorizer.pkl')
        model = joblib.load('models/sentiment_model.pkl')
        return tfidf, model
    except:
        return None, None

df = load_data()
tfidf, model = load_models()

# Menu
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Aller à :", [
    "🔍 Recherche d'Avis (IR)", 
    "🔮 Prédiction & Explication", 
    "📊 Analyse par Assureur (Summary)",
    "🤖 Assistant Virtuel (RAG / QA)"
])

# INFORMATION RETRIEVAL (IR) - PAGE 1
if menu == "🔍 Recherche d'Avis (IR)":
    st.title("Recherche d'Avis Spécifiques")
    st.write("Trouvez des avis en utilisant des mots-clés ou des filtres.")
    
    # Filtres
    col1, col2 = st.columns(2)
    with col1:
        mot_cle = st.text_input("Mot-clé (ex: sinistre, remboursement, arnaque) :")
    with col2:
        assureur_choisi = st.selectbox("Filtrer par assureur :", ["Tous"] + list(df['assureur'].unique()))
        
    # Application des filtres
    df_filtre = df.copy()
    if assureur_choisi != "Tous":
        df_filtre = df_filtre[df_filtre['assureur'] == assureur_choisi]
    if mot_cle:
        df_filtre = df_filtre[df_filtre['avis'].str.contains(mot_cle, case=False, na=False)]
        
    st.write(f"**{len(df_filtre)} avis trouvés**")
    st.dataframe(df_filtre[['assureur', 'note', 'avis']].head(50)) # On affiche les 50 premiers

# Prediction & Explication - PAGE 2
elif menu == "🔮 Prédiction & Explication":
    st.title("Prédire le sentiment d'un avis")
    
    if model is None or tfidf is None:
        st.error("Les modèles ML ne sont pas trouvés. Placez les fichiers .pkl dans le dossier.")
    else:
        user_input = st.text_area("Entrez un avis (ex: 'Le prix est super mais le service est lent') :")
        
        if st.button("Prédire le sentiment"):
            if user_input:
                # 1. Prediction
                clean_input = re.sub(r'[^\w\s]', ' ', user_input.lower())
                X_vec = tfidf.transform([clean_input])
                prediction = model.predict(X_vec)[0]
                
                st.subheader("Résultat de la prédiction :")
                if prediction == 'positif': st.success("🟢 Sentiment POSITIF")
                elif prediction == 'negatif': st.error("🔴 Sentiment NÉGATIF")
                else: st.warning("🟡 Sentiment NEUTRE")
                
                # 2. Explanation
                st.subheader("Pourquoi cette prédiction ? (Explication)")
                st.write("Le modèle a repéré ces mots-clés importants dans votre phrase :")
                
                # On regarde quels mots de l'utilisateur existent dans le vocabulaire du TF-IDF
                mots_utilisateur = clean_input.split()
                mots_connus = [mot for mot in mots_utilisateur if mot in tfidf.vocabulary_]
                
                if mots_connus:
                    st.info(", ".join(mots_connus))
                    st.write("*Note : Le modèle TF-IDF se base sur la présence et la rareté de ces mots spécifiques dans le corpus d'entraînement.*")
                else:
                    st.write("Aucun mot décisif n'a été reconnu par le modèle.")

# Analyse par Assureur (Summary) - PAGE 3
elif menu == "📊 Analyse par Assureur (Summary)":
    st.title("Analyse des Performances par Assureur")
    st.write("Vue d'ensemble des métriques et des résumés par compagnie d'assurance.")
    
    # Sélection de l'assureur   
    assureurs_dispo = df['assureur'].dropna().unique().tolist()
    assureur_choisi = st.selectbox("Choisissez un assureur à analyser :", sorted(assureurs_dispo))
    
    # Filtrage des données pour cet assureur
    df_assureur = df[df['assureur'] == assureur_choisi].copy()
    
    if len(df_assureur) > 0:
        # Métriques
        st.subheader(f"Chiffres clés : {assureur_choisi}")
        col1, col2, col3 = st.columns(3)
        
        note_moyenne = df_assureur['note'].mean()
        nb_avis = len(df_assureur)
        part_positifs = (len(df_assureur[df_assureur['note'] >= 4]) / nb_avis) * 100
        
        col1.metric("Note Moyenne", f"{note_moyenne:.2f} / 5")
        col2.metric("Volume d'avis", f"{nb_avis}")
        col3.metric("Avis Positifs", f"{part_positifs:.1f} %")
        
        st.divider()
        
        # Visualisation
        st.subheader("Répartition des Notes")
        repartition = df_assureur['note'].value_counts().sort_index()
        st.bar_chart(repartition, color="#ff4b4b")
        
        st.divider()
        
        st.subheader("Résumé automatique (Summary)")
        st.write("Voici les mots et thématiques les plus fréquents pour cet assureur :")
        
        # Mini NLP pour extraire les mots fréquents spécifiques à cet assureur
        texte_complet = " ".join(df_assureur['avis'].dropna().astype(str).tolist())
        texte_propre = re.sub(r'[^\w\s]', ' ', texte_complet.lower())
        mots = [m for m in texte_propre.split() if len(m) > 4 and m not in ['assurance', 'contrat', 'assureur', 'client']]
        
        from collections import Counter
        mots_frequents = Counter(mots).most_common(10)
        
        # Affichage sous forme de badges/tags
        cols = st.columns(5)
        for i, (mot, freq) in enumerate(mots_frequents[:5]):
            cols[i].button(f"{mot} ({freq})", key=f"btn_{i}")
            
    else:
        st.warning("Aucune donnée disponible pour cet assureur.")

# RAG / QA - PAGE 4
elif menu == "🤖 Assistant Virtuel (RAG / QA)":
    st.title("Assistant Virtuel (RAG & QA)")
    st.markdown("Posez une question. Le système récupère le contexte pertinent (Retrieval) pour vous apporter une réponse factuelle basée sur les données.")
    
    st.info("Exemples de questions : 'Quels sont les problèmes avec Direct Assurance ?' ou 'Que disent les clients sur le prix chez GMF ?'")
    
    question = st.text_input("Votre question :")
    
    if st.button("Demander à l'assistant"):
        if question:
            with st.spinner("Analyse de la question et recherche dans la base de données..."):

                question_lower = question.lower()
                assureur_concerne = None
                
                # Liste des assureurs pour la détection
                assureurs = df['assureur'].dropna().unique()
                for assureur in assureurs:
                    if str(assureur).lower() in question_lower:
                        assureur_concerne = assureur
                        break
                
                # Détection du thème
                theme = "général"
                mots_prix = ['prix', 'tarif', 'cher', 'cotisation', 'augmentation']
                mots_service = ['service', 'client', 'téléphone', 'joignable', 'conseiller']
                mots_sinistre = ['sinistre', 'accident', 'remboursement', 'expert', 'dégât']
                
                if any(m in question_lower for m in mots_prix): theme = "prix"
                elif any(m in question_lower for m in mots_service): theme = "service client"
                elif any(m in question_lower for m in mots_sinistre): theme = "gestion des sinistres"
                
                # Filtrage du contexte (Retrieval)
                df_rag = df.copy()
                if assureur_concerne:
                    df_rag = df_rag[df_rag['assureur'] == assureur_concerne]
                    
                # On filtre sur les avis contenant des mots du thème
                if theme == "prix":
                    df_rag = df_rag[df_rag['avis'].str.contains('|'.join(mots_prix), case=False, na=False)]
                elif theme == "service client":
                    df_rag = df_rag[df_rag['avis'].str.contains('|'.join(mots_service), case=False, na=False)]
                elif theme == "gestion des sinistres":
                    df_rag = df_rag[df_rag['avis'].str.contains('|'.join(mots_sinistre), case=False, na=False)]
                
                # GENERATION (Réponse structurée basée sur les données récupérées)
                st.subheader("💡 Réponse de l'Assistant")
                
                if len(df_rag) == 0:
                    st.warning("Je n'ai pas trouvé suffisamment d'informations dans les avis clients pour répondre à cette question avec certitude.")
                else:
                    # Calcul du sentiment global sur ce contexte précis
                    note_moyenne = df_rag['note'].mean()
                    sentiment = "plutôt positif" if note_moyenne >= 3.5 else "plutôt négatif" if note_moyenne <= 2.5 else "mitigé"
                    
                    nom_assur = assureur_concerne if assureur_concerne else "les assureurs en général"
                    
                    # Construction de la réponse synthétique (Génération basée sur le contexte)
                    reponse = f"""
                    D'après ma base de données (basée sur {len(df_rag)} avis pertinents récupérés), le sentiment concernant **{theme}** chez **{nom_assur}** est **{sentiment}** (Note moyenne : {note_moyenne:.1f}/5).
                    
                    Voici les points clés qui ressortent des avis :
                    """
                    st.write(reponse)
                    
                    # Affichage des 3 avis les plus pertinents pour justifier (Extractive QA)
                    st.write("**Extraits représentatifs des clients :**")
                    avis_a_afficher = df_rag['avis'].dropna().head(3).tolist()
                    for avis in avis_a_afficher:
                        st.info(f'"{avis}"')
                        
                    st.success("Processus RAG terminé : Retrieval (Filtrage sémantique) + Generation (Synthèse métrique et extractive).")