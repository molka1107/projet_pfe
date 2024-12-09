FROM python:3.9-slim

# Mise à jour des packages et installation de python3 et pip, ainsi que des outils nécessaires pour la compilation
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    build-essential \
    libasound2-dev  

# Mettre à jour pip à la dernière version
RUN python3 -m pip install --upgrade pip

# Définir le répertoire de travail
WORKDIR /app

# Copier le fichier requirements.txt dans l'image
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le code source dans le conteneur
COPY . .

# Exposer le port si nécessaire
EXPOSE 8601

# Lancer l'application Streamlit
CMD ["streamlit", "run", "app.py"]
