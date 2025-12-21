import streamlit as st
import requests
from datetime import datetime

# Page config
st.set_page_config(
    page_title="SmartCOVER Chat",
    page_icon="💬",
    layout="centered"
)

# Header
st.title("💬 SmartCOVER Assistant")
st.markdown("Décrivez votre sinistre en français, anglais ou arabe")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.claim_submitted = False

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        
        # If this message has acceptance info, show it
        if msg.get("acceptance_info"):
            info = msg["acceptance_info"]
            
            st.success("✅ Réclamation soumise avec succès!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Probabilité d'acceptation",
                    f"{info['acceptance']*100:.0f}%"
                )
            with col2:
                st.metric(
                    "Numéro de dossier",
                    info['claim_id']
                )
            
            st.info("📋 Votre dossier est en cours de traitement par la compagnie d'assurance.")

# Chat input
if not st.session_state.claim_submitted:
    if prompt := st.chat_input("Décrivez votre sinistre..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Call FastAPI backend
        with st.chat_message("assistant"):
            with st.spinner("🤔 Traitement en cours..."):
                try:
                    response = requests.post(
                        "http://localhost:5000/chat",
                        json={
                            "user_id": "user_123",
                            "message": prompt,
                            "thread_id": st.session_state.thread_id
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Create assistant message
                        assistant_msg = {
                            "role": "assistant",
                            "content": result["reply"]
                        }
                        
                        # Display reply
                        st.write(result["reply"])
                        
                        # If claim is complete, show acceptance info
                        if result.get("is_complete"):
                            acceptance_info = {
                                "acceptance": result.get("acceptance_probability", 0),
                                "claim_id": result.get("claim_id", "N/A")
                            }
                            
                            assistant_msg["acceptance_info"] = acceptance_info
                            st.session_state.claim_submitted = True
                            
                            # Display acceptance
                            st.success("✅ Réclamation soumise avec succès!")
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    "Probabilité d'acceptation",
                                    f"{acceptance_info['acceptance']*100:.0f}%"
                                )
                            with col2:
                                st.metric(
                                    "Numéro de dossier",
                                    acceptance_info['claim_id']
                                )
                            
                            st.info("📋 Votre dossier est en cours de traitement par la compagnie d'assurance.")
                            st.balloons()
                        
                        # Save message
                        st.session_state.messages.append(assistant_msg)
                        st.rerun()
                    
                    else:
                        st.error(f"❌ Erreur serveur: {response.status_code}")
                
                except requests.exceptions.ConnectionError:
                    st.error("❌ Impossible de se connecter au serveur. Vérifiez que FastAPI est démarré sur le port 5000.")
                except requests.exceptions.Timeout:
                    st.error("❌ Le serveur met trop de temps à répondre. Réessayez.")
                except Exception as e:
                    st.error(f"❌ Erreur: {str(e)}")

else:
    st.info("✅ Réclamation déjà soumise. Rafraîchissez la page pour une nouvelle réclamation.")

# Sidebar
with st.sidebar:
    st.header("ℹ️ Session Info")
    st.write(f"**Thread ID:** `{st.session_state.thread_id}`")
    st.write(f"**Messages:** {len(st.session_state.messages)}")
    
    if st.button("🔄 Nouvelle réclamation"):
        st.session_state.messages = []
        st.session_state.thread_id = f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.session_state.claim_submitted = False
        st.rerun()
    
    st.divider()
    
    st.caption("SmartCOVER AI Assistant")
