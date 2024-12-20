import torch
import sys
sys.path.append('/home/molka/Bureau/stage/projet_pfe/yolov7')
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox
import numpy as np
import cv2
import pytest
import os


def load_class_names(names_path):
    with open(names_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def load_model():
    model_path = 'yolov7/modele_a.pt'  
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle {model_path} est introuvable.")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(model_path, map_location=device)
    model.eval()
    model.half()
    return model, device


def detect_objects(model, device, image_path):
    img = cv2.imread(image_path)
    if img is None: 
        raise FileNotFoundError(f"Image non trouvée ou invalide : {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = letterbox(img, new_shape=(320, 320))[0]  
    img_resized = img_resized.transpose((2, 0, 1)) 
    img_resized = np.ascontiguousarray(img_resized)

    # Prétraiter l'image
    img_tensor = torch.from_numpy(img_resized).to(device)
    img_tensor = img_tensor.half() / 255.0  
    if img_tensor.ndimension() == 3 :
        img_tensor = img_tensor.unsqueeze(0)

    # Inférence avec YOLOv7
    pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45) 

    objects = []
    for det in pred:  
        if len(det):
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                objects.append({
                    "box": [int(coord) for coord in xyxy],  
                    "score": float(conf),  
                    "class": int(cls)  
                })

    return objects


def test_load_model():
    model, device = load_model()
    assert model is not None, "Le modèle n'a pas pu être chargé."
    assert device is not None, "Le dispositif (CPU/GPU) n'a pas pu être initialisé."


def test_detect_objects():
    # Chargez le modèle
    model, device = load_model()

    # Charger les noms des classes
    class_names = load_class_names("/home/molka/Bureau/stage/projet_pfe/classes.txt")


    # Chemin d'une image d'exemple
    image_path = "/home/molka/Bureau/stage/projet_pfe/person.jpg"

    # Détectez les objets
    objects = detect_objects(model, device, image_path)

    # Vérifiez qu'au moins un objet est détecté
    assert len(objects) > 0, "Aucun objet n'a été détecté dans l'image."

    # Vérifiez que les résultats contiennent les champs attendus
    for obj in objects:
        assert "box" in obj, "Le champ 'box' est manquant dans les résultats."
        assert "score" in obj, "Le champ 'score' est manquant dans les résultats."
        assert "class" in obj, "Le champ 'class' est manquant dans les résultats."

    # Validez les noms des classes
    for obj in objects:
        class_idx = obj["class"]
        # Vérifiez que l'indice de classe est valide
        assert 0 <= class_idx < len(class_names), f"Indice de classe invalide : {class_idx}"
        detected_class_name = class_names[class_idx]
        assert detected_class_name in class_names, f"Classe détectée invalide : {detected_class_name}"

    # Vérifiez que les scores de confiance sont supérieurs à un seuil
    confidence_threshold = 0.25
    for obj in objects:
        assert obj["score"] >= confidence_threshold, f"Score de confiance trop faible : {obj['score']}."


def test_detect_objects_invalid_image():
    # Chargez le modèle
    model, device = load_model()

    # Chemin vers une image inexistante
    invalid_image_path = "/home/molka/Bureau/stage/projet_pfe/nonexistent_image.jpg"

    # Vérifiez que l'erreur est levée pour une image inexistante
    with pytest.raises(FileNotFoundError):
        detect_objects(model, device, invalid_image_path)

