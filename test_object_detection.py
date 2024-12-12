import torch
import sys
sys.path.append('/home/molka/Bureau/stage/projet_pfe/yolov7')
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords
from yolov7.utils.datasets import letterbox
import numpy as np
import cv2
import pytest


def load_model():
    # Charger le modèle YOLOv7
    model_path = 'yolov7/modele_a.pt'  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = attempt_load(model_path, map_location=device)
    model.eval() 
    model.half()  
    return model,device


def detect_objects(model, device, image_path):
    # Charger l'image
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
    if img_tensor.ndimension() == 3:
        img_tensor = img_tensor.unsqueeze(0)

    # Inférence avec YOLOv7
    pred = model(img_tensor, augment=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.25, iou_thres=0.45) 

    objects = []
    for det in pred:  
        if len(det):
            # Redimensionner les coordonnées vers l'image originale
            det[:, :4] = scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()
            for *xyxy, conf, cls in reversed(det):
                objects.append({
                    "box": [int(coord) for coord in xyxy],  
                    "score": float(conf),  
                    "class": int(cls)  
                })

    return objects


def test_load_model():
    model= load_model()
    assert model is not None, "Le modèle n'a pas pu être chargé."


def test_detect_objects():
    # Chargez le modèle
    model, device = load_model()

    # Chemin d'une image d'exemple (assurez-vous que cette image existe dans votre projet)
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

    valid_classes = range(0, 25) 
    for obj in objects:
        assert obj["class"] in valid_classes, f"Classe détectée invalide : {obj['class']}."

    # Vérifiez que les scores de confiance sont supérieurs à un seuil
    confidence_threshold = 0.25
    for obj in objects:
        assert obj["score"] >= confidence_threshold, f"Score de confiance : {obj['score']}."

def test_detect_objects_invalid_image():
    # Chargez le modèle
    model, device = load_model()

    # Chemin vers une image inexistante
    invalid_image_path = "/home/molka/Bureau/stage/projet_pfe/nonexistent_image.jpg"

    # Vérifiez que l'erreur est levée pour une image inexistante
    with pytest.raises(FileNotFoundError):
        detect_objects(model, device, invalid_image_path)


