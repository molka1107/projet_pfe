import torch
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize
import torch.nn.functional as F
import sys
sys.path.append('/home/molka/Bureau/stage/assistant virtuel pour personnes malvoyantes/yolov7')
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
with open("yolov7_classes.txt", "r") as f:
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


# Fonction pour convertir la carte de profondeur en carte de distance ajustée
def depth_to_distance(depth_map):
    min_depth = depth_map.min() 
    max_depth = depth_map.max()  
    normalized_depth = (depth_map - min_depth) / (max_depth - min_depth) 
    return 1.0 / (0.1 * normalized_depth + 0.01)  


# Paramètres pour la zone centrale (ajustez ces valeurs pour changer la taille de la zone centrale)
center_exclude_factor_w = 0.4   
center_exclude_factor_h = 0.4  


# Charger le fichier audio pour l'alarme
alarm_sound = sa.WaveObject.from_wave_file('/home/molka/Bureau/stage/assistant virtuel pour personnes malvoyantes/bip.wav')
alarm_playing = None


# Initialiser la capture vidéo depuis la webcam
cap = cv2.VideoCapture(0) 
prev_time = time.time() 
while True:
    ret, frame = cap.read()
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
    pred = non_max_suppression(pred, 0.4, 0.5, classes=None, agnostic=False)
    

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
                    cv2.rectangle(frame, (x, y), (x2, y2), color, thickness=3)
                    cv2.putText(frame, "{}".format(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    cv2.rectangle(frame, (x,y-3), (x+200, y+23),(255,255,255),-1)
                    cv2.putText(frame, f"Distance : {format(distance,'.2f')} unite", (x+5, y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    if distance <= 16:
                        alarm_triggered = True


    if alarm_triggered and alarm_playing is None:
        alarm_playing = alarm_sound.play()
    elif not alarm_triggered and alarm_playing is not None:
        alarm_playing.stop()
        alarm_playing = None


    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time 
    cv2.rectangle(frame, center_top_left, center_bottom_right, (0, 0, 255), 2) 
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Affichage du résultat final combiné de la carte de profondeur et de l'image originale
    final_result = cv2.hconcat([depth_color, frame])
    cv2.imshow('Depth Estimation', final_result)
    if cv2.waitKey(1) == ord('q'):  
        break


if alarm_playing is not None:
    alarm_playing.stop()


# Libération des ressources et fermeture des fenêtres
cap.release()
cv2.destroyAllWindows()
