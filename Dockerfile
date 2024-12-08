FROM python:3.9-slim

# Mise à jour des packages et installation des outils nécessaires pour la compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    libasound2-dev  

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans l'image
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source dans le conteneur
COPY . .

# Exposer le port si nécessaire
EXPOSE 8501

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py"]
