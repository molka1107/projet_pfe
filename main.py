import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize
import torch.nn.functional as F
import sys
import signal  
sys.path.append('/home/molka/Bureau/stage/projet_pfe/yolov7')
from yolov7.models.experimental import attempt_load
from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.torch_utils import  time_synchronized
import warnings
warnings.filterwarnings("ignore")
import os
import time
import simpleaudio as sa
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import tempfile
import streamlit as st
from plyer import notification
import subprocess
import webbrowser  


st.set_page_config(
    page_title="Assistant Virtuel pour Personnes Malvoyantes",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
    <style>
    .title-with-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: row-reverse;
        margin: 0 auto; 
    }
    .title-with-icon img {
        width: 70px;
        height: 70px;
    }
    </style>
""", unsafe_allow_html=True)
st.markdown("""
    <div class="title-with-icon">
        <img src="https://cdn.iconscout.com/icon/premium/png-512-thumb/assistant-virtuel-7660458-6297102.png?f=webp&w=256" alt="icone">
        <h1>Assistant virtuel pour les personnes malvoyantes</h1>
    </div>
""", unsafe_allow_html=True)


# Charger le CSS à partir d'un fichier externe
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


# Définition de l'appareil (CPU ou GPU)
device= 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device: %s" % device)
if device == 'cuda':
    gpu_name = torch.cuda.get_device_name(0)  
    print("Device: %s (%s)" % (device, gpu_name))
else:
    print("Device: %s" % device)


# Charger le modèle YOLOv7 pour la détection d'objets
model_path = 'yolov7/modele_a.pt'
model = attempt_load(model_path, map_location=device)
# Mettre le modèle YOLOv7 en mode évaluation
model.eval()
# Convertir le modèle en demi-précision (Utiliser FP16 pour YOLO)
model.half() 
# Utiliser stride et taille d'image en fonction du modèle chargé
stride = int(model.stride.max())
imgsz = check_img_size(320, s=stride)


# Classes pour la détection d'objets
with open("classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]


# Avant de charger le modèle d'estimation de profondeur
torch.cuda.empty_cache()
# Chargement du modèle d'estimation de profondeur
encoder = 'vits'
depth_model = DepthAnything.from_pretrained(f'LiheYoung/depth_anything_{encoder}14').to(device)
depth_model.eval()
depth_model.half() 


# Définition des transformations pour l'estimation de profondeur
depth_transform = Compose([
    Resize(width=320, height=240, resize_target=True, keep_aspect_ratio=True, ensure_multiple_of=14, resize_method='lower_bound', image_interpolation_method=cv2.INTER_CUBIC),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])


def logout():
    st.session_state.clear()  
    st.success("Déconnexion réussie !")
    time.sleep(3)  
    subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501"]) 
    webbrowser.open("http://localhost:8501")  
    os.kill(os.getpid(), signal.SIGTERM)


# Fonction pour convertir la carte de profondeur en carte de distance ajustée
def depth_to_distance(depth_map):
    min_depth = depth_map.min() 
    max_depth = depth_map.max()  
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth) 
    return 1.0 / (0.1 * normalized_depth + 0.01)  


def notifier_utilisateur(message):
     notification.notify(
        title='⚠️ Alerte de Détection', 
        message=f"🚨 {message}",          
        timeout=5
    )


# Paramètres pour la zone centrale (ajustez ces valeurs pour changer la taille de la zone centrale)
center_exclude_factor_w = 0.4  
center_exclude_factor_h = 0.4  


# Charger le fichier audio pour l'alarme
alarm_sound = sa.WaveObject.from_wave_file('/home/molka/Bureau/stage/projet_pfe/bip.wav')
alarm_playing = None


# Initialisation de l'état de la webcam
if 'cap' not in st.session_state:
    st.session_state.cap = None


st.sidebar.markdown("""
# Bienvenue <img src="https://cdn-icons-png.flaticon.com/128/2339/2339864.png" alt="Hand Icon" width="40">
""", unsafe_allow_html=True)
st.sidebar.header("Options")
option = st.sidebar.selectbox("Choisissez une option", ["Caméra", "Importer une image", "Importer une vidéo"])
st.sidebar.header("Confidence")
confidence = st.sidebar.slider("Choisissez une confidence", min_value =0.0,max_value = 1.0,value = 0.3)
st.sidebar.header("IoU")
iou_threshold = st.sidebar.slider("Choisissez une valeur d'IoU", min_value=0.0, max_value=1.0, value=0.5)
st.sidebar.header("Classes")
options = ["all objects"] + classes
default_classes = ["all objects"]
selected_classes = st.sidebar.multiselect("Choisissez les classes pour la détection",options=options,default=default_classes)
if "all objects" in selected_classes:
    class_indices = None
else:
    class_indices = [i for i, cls in enumerate(classes) if cls in selected_classes]
stframe = st.empty()
st.sidebar.button("Déconnexion", on_click=logout)  
    


# Option 1: Caméra
if "detection_done" not in st.session_state:
    st.session_state["detection_done"] = False
if option == "Caméra":
    st.markdown("""
    <div class="title-with-icon">
        <img src="https://cdn.iconscout.com/icon/premium/png-512-thumb/camera-3402-1091292.png?f=webp&w=256" alt="icone">
        <h2>Détection en temps réel par caméra</h2>
    </div>
    """, unsafe_allow_html=True)


    if 'cap' in st.session_state and st.session_state.cap is not None:
        st.session_state.cap.release()
        

    col1, col2, col3, col4, col5= st.columns(5)
    with col1:
        st.empty()
    with col2:
        start_button = st.button("Démarrer la caméra")
    with col3:
        st.empty()
    with col4:
        stop_button = st.button("Arrêter la caméra")
    with col5:
        st.empty()


    # Démarrer la caméra si le bouton est cliqué
    if start_button:
        if st.session_state.cap is None or not isinstance(st.session_state.cap, cv2.VideoCapture) or not st.session_state.cap.isOpened():
            st.session_state.cap = cv2.VideoCapture(0)
        # Si la caméra n'est toujours pas ouverte, afficher un message d'erreur
        if not st.session_state.cap.isOpened():
            message_placeholder = st.empty()
            message_placeholder.error("Impossible d'ouvrir la caméra ! Veuillez vérifier si elle est utilisée par une autre application !")
            time.sleep(5)
            message_placeholder.empty()
       

    # Arrêter la caméra si le bouton est cliqué
    if stop_button and st.session_state.cap is not None:
        if isinstance(st.session_state.cap, cv2.VideoCapture):
            st.session_state.cap.release()
        st.session_state.cap = None
        st.session_state["detection_done"] = True 
        stframe.empty() 


    # Vérifier si la webcam est bien initialisée avant de démarrer le flux
    if st.session_state.cap is not None and isinstance(st.session_state.cap, cv2.VideoCapture) and st.session_state.cap.isOpened():
        stframe = st.empty()  
        detection_text = st.empty()


        prev_time = time.time()
        while st.session_state.cap.isOpened():
            ret, frame = st.session_state.cap.read()
            if not ret:
                break
            # Obtenir les dimensions de l'image
            height, width, channels = frame.shape


            # Calculer les coordonnées de la zone centrale à exclure en fonction des paramètres
            center_top_left = (int(width * (1 - center_exclude_factor_w) / 2), int(height * (1 - center_exclude_factor_h) / 2))
            center_bottom_right = (int(width * (1 + center_exclude_factor_w) / 2), int(height * (1 + center_exclude_factor_h) / 2))


            # Estimation de la profondeur pour chaque frame capturée
            input_frame = depth_transform({'image': cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0})['image']
            input_tensor = torch.from_numpy(input_frame).unsqueeze(0).to(device)
            input_tensor = input_tensor.half() 
            with torch.no_grad():
                depth_map = depth_model(input_tensor)


            # Prétraitement de l'image pour le modèle YOLOv7
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (imgsz, imgsz))
            img = img.transpose(2, 0, 1) 
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.float() / 255.0 
            img = img.unsqueeze(0) 


            # Inférence avec YOLOv7
            t1 = time_synchronized()
            # Convertir l'image en demi-précision avant de la passer au modèle
            img = img.half()
            # Effectuer l'inférence avec torch.no_grad() pour économiser de la mémoire
            with torch.no_grad():
                pred = model(img, augment=False)[0]
            t2 = time_synchronized()
            # Application de la suppression des non-maxima (NMS) pour la détection d'objets
            pred = non_max_suppression(pred, confidence, iou_threshold, classes=class_indices, agnostic=False)
            

            # Normalisation et mise à l'échelle de la carte de profondeur
            if depth_map.dim() == 3:
                depth_map = depth_map.unsqueeze(1)  
            depth_map = F.interpolate(depth_map, (frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)
            depth_map = depth_map[0, 0].cpu().numpy()
            depth_map_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
            depth_map_normalized = depth_map_normalized.astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)
            

            distance_map = depth_to_distance(depth_map)
            alarm_triggered = False


            # Traitement des détections d'objets et affichage des résultats
            class_counts = {cls: 0 for cls in classes}
            for i, det in enumerate(pred):
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                    for *xyxy, conf, cls in reversed(det):
                        label = f'{classes[int(cls)]} : {conf:.2f}'
                        color = (0, 255, 0)
                        x, y, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                        w, h = x2 - x, y2 - y
                        y_center = int(min(max(y + h / 2, 0), distance_map.shape[0] - 1))
                        x_center = int(min(max(x + w / 2, 0), distance_map.shape[1] - 1))
                        distance = distance_map[y_center, x_center]
                        #Exclure les objets dans la zone centrale
                        if not (center_top_left[0] < x_center < center_bottom_right[0] and
                                center_top_left[1] < y_center < center_bottom_right[1]):
                            frame = cv2.rectangle(frame, (x, y), (x2, y2), color, thickness=3)
                            frame = cv2.putText(frame, "{}".format(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            frame = cv2.rectangle(frame, (x,y-3), (x+200, y+23),(255,255,255),-1)
                            frame = cv2.putText(frame, f"Distance : {format(distance,'.2f')} unite", (x+5, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            if distance <= 16:
                                alarm_triggered = True
                            class_counts[classes[int(cls)]] += 1  
                            

            if alarm_triggered and alarm_playing is None:
                alarm_playing = alarm_sound.play()
                notifier_utilisateur("Objet détecté à une distance dangereuse !")
            elif not alarm_triggered and alarm_playing is not None:
                alarm_playing.stop()
                alarm_playing = None
            
            
            detection_summary = " | ".join([f"Nombre d'objets de la classe {cls} est : {count}" for cls, count in class_counts.items() if count > 0])
            st.markdown("""
                    <style>
                        .detection-text {
                            font-weight: bold; 
                            color: #FFFFFF; 
                            font-size: 20px; 
                            text-align: left; 
                            background-color: #004097; 
                            padding: 8px; 
                            border: 5px solid #1560c9; 
                            border-radius: 8px;    
                        }
                    </style>
                """, unsafe_allow_html=True)
            detection_text.markdown(f'<div class="detection-text">{detection_summary}</div>', unsafe_allow_html=True)
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
            prev_time = current_time 
            cv2.rectangle(frame, center_top_left, center_bottom_right, (0, 0, 255), 2) 
            cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


            # Combiner les deux images horizontalement (carte de profondeur + détection d'objets)
            combined_result = cv2.hconcat([depth_color, frame])
            # Affichage du résultat combiné dans Streamlit
            stframe.image(combined_result, channels="BGR", use_column_width=True)
            if stop_button:
                break


        if alarm_playing is not None:
            alarm_playing.stop()


        # Libérer les ressources si l'utilisateur appuie sur le bouton arrêter
        if stop_button and st.session_state.cap is not None:
            if isinstance(st.session_state.cap, cv2.VideoCapture):
                st.session_state.cap.release()
            st.session_state.cap = None
            stframe.empty()
   

    if st.session_state.get("detection_done", False):
        st.session_state["detection_done"] = False
        message_placeholder = st.empty()
        message_placeholder.success("Détection terminée !")
        time.sleep(3)
        message_placeholder.empty() 



# Option 2: Imporeter une image
if option == "Importer une image":
    st.markdown("""
    <div class="title-with-icon">
        <img src="https://cdn-icons-png.flaticon.com/128/1375/1375106.png" alt="icone">
        <h2>Détection à partir d'une image</h2>
    </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Téléchargez une image", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        if 'image' in st.session_state and st.session_state.image is not None:
            st.session_state.image = None
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, 1)

       
        st.session_state.image = frame
        # Prétraitement de l'image pour le modèle YOLOv7
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (imgsz, imgsz))
        img = img.transpose(2, 0, 1) 
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0 
        img = img.unsqueeze(0)


        # Inférence avec YOLOv7
        t1 = time_synchronized()
        # Convertir l'image en demi-précision avant de la passer au modèle
        img = img.half()
        # Effectuer l'inférence avec torch.no_grad() pour économiser de la mémoire
        with torch.no_grad():
            pred = model(img, augment=False)[0]
        t2 = time_synchronized()
        # Application de la suppression des non-maxima (NMS) pour la détection d'objets
        pred = non_max_suppression(pred, confidence, iou_threshold, classes=class_indices, agnostic=False)

        
        # Traitement des détections d'objets et affichage des résultats
        class_counts = {cls: 0 for cls in classes}
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f'{classes[int(cls)]} : {conf:.2f}'
                    color = (0, 255, 0)
                    x, y, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    w, h = x2 - x, y2 - y
                    frame = cv2.rectangle(frame, (x, y), (x2, y2), color, thickness=3)
                    frame = cv2.putText(frame, "{}".format(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    class_counts[classes[int(cls)]] += 1


        st.image(frame, channels="BGR", use_column_width=True)
        detection_summary = " | ".join([f"Nombre d'objets de la classe {cls} est : {count}" for cls, count in class_counts.items() if count > 0]) 
        st.markdown("""
            <style>
                .custom-text {
                    font-weight: bold;
                    color: #FFFFFF; 
                    font-size: 20px;
                    text-align: center;
                    background-color: #004097;
                    padding: 8px;
                    border: 5px solid #1560c9;
                    border-radius: 8px;
                }
            </style>
        """, unsafe_allow_html=True)
        st.markdown(f'<div class="custom-text">{detection_summary}</div>', unsafe_allow_html=True)
        message_placeholder = st.empty()
        message_placeholder.success("Détection terminée !")
        time.sleep(3)
        message_placeholder.empty()
        


# Option 3: Importer une vidéo
if "is_video_running" not in st.session_state:
    st.session_state["is_video_running"] = False
if option == "Importer une vidéo":
    st.markdown("""
    <div class="title-with-icon">
        <img src="https://cdn-icons-png.flaticon.com/128/9582/9582436.png" alt="icone">
        <h2>Détection à partir d'une vidéo</h2>
    </div>
    """, unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Téléchargez une vidéo", type=['mp4', 'avi', 'mov'])  
    if uploaded_file is not None:
        if st.session_state.cap is not None:
            st.session_state.cap.release()
            st.session_state.cap = None
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())


        # Boutons de contrôle (seulement disponibles après téléchargement)
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.empty()
        with col2:
            start_video_button = st.button("Démarrer la vidéo")
        with col3:
            st.empty()
        with col4:
            stop_video_button = st.button("Arrêter la vidéo")
        with col5:
            st.empty()


        # Démarrer la capture vidéo lorsque le bouton est cliqué
        if start_video_button and not st.session_state.is_video_running:
            st.session_state.cap = cv2.VideoCapture(tfile.name)
            st.session_state.is_video_running = True


        # Arrêter la capture vidéo lorsque le bouton est cliqué
        if stop_video_button and st.session_state.is_video_running:
            st.session_state.is_video_running = False
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            stframe.empty()  
            message_placeholder = st.empty()
            message_placeholder.success("Détection terminée !")
            time.sleep(3)
            message_placeholder.empty()


        # Vérifier si la capture vidéo est bien initialisée
        if st.session_state.is_video_running and st.session_state.cap is not None and st.session_state.cap.isOpened():
            stframe = st.empty()
            detection_text = st.empty()


            prev_time = time.time()
            while st.session_state.is_video_running and st.session_state.cap.isOpened():
                ret, frame = st.session_state.cap.read()
                if not ret:
                    break


                # Prétraitement de l'image pour le modèle YOLOv7
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (imgsz, imgsz))
                img = img.transpose(2, 0, 1) 
                img = np.ascontiguousarray(img)
                img = torch.from_numpy(img).to(device)
                img = img.float() / 255.0 
                img = img.unsqueeze(0) 

                
                # Inférence avec YOLOv7
                t1 = time_synchronized()
                # Convertir l'image en demi-précision avant de la passer au modèle
                img = img.half()
                # Effectuer l'inférence avec torch.no_grad() pour économiser de la mémoire
                with torch.no_grad():
                    pred = model(img, augment=False)[0]
                t2 = time_synchronized()
                # Application de la suppression des non-maxima (NMS) pour la détection d'objets
                pred = non_max_suppression(pred, confidence, iou_threshold, classes=class_indices, agnostic=False)
                

                # Traitement des détections d'objets et affichage des résultats
                class_counts = {cls: 0 for cls in classes}
                for i, det in enumerate(pred):
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            label = f'{classes[int(cls)]} : {conf:.2f}'
                            color = (0, 255, 0)
                            x, y, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                            w, h = x2 - x, y2 - y
                            frame = cv2.rectangle(frame, (x, y), (x2, y2), color, thickness=3)
                            frame = cv2.putText(frame, "{}".format(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            class_counts[classes[int(cls)]] += 1
                

                detection_summary = " | ".join([f"Nombre d'objets de la classe {cls} est : {count}" for cls, count in class_counts.items() if count > 0])
                st.markdown("""
                    <style>
                        .detection-text {
                            font-weight: bold; 
                            color: #FFFFFF; 
                            font-size: 20px; 
                            text-align: left; 
                            background-color: #004097; 
                            padding: 8px; 
                            border: 5px solid #1560c9; 
                            border-radius: 8px;    
                        }
                    </style>
                """, unsafe_allow_html=True)
                detection_text.markdown(f'<div class="detection-text">{detection_summary}</div>', unsafe_allow_html=True)
                current_time = time.time()
                fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                prev_time = current_time
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                stframe.image(frame, channels="BGR", use_column_width=True)
                # Sortir de la boucle si le bouton d'arrêt est cliqué
                if not st.session_state.is_video_running:
                    break


            # Libérer les ressources vidéo si la boucle est terminée
            if not st.session_state.is_video_running:
                st.session_state.cap.release()
                st.session_state.cap = None
                stframe.empty()
                message_placeholder = st.empty()
                message_placeholder.success("Détection terminée !")
                time.sleep(3)
                message_placeholder.empty()

