import streamlit as st
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms
from PIL import Image, ImageDraw
import osmnx as ox
import folium
from geopy.geocoders import Nominatim
import pandas as pd
from streamlit_folium import st_folium

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource(show_spinner=True)
def load_model():
    num_classes = 3 
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    checkpoint = torch.load(r"C:\Harsh\Desktop\DL_project\parking_rcnn_best.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -------------------- IMAGE TRANSFORM --------------------
transform = transforms.Compose([transforms.ToTensor()])

# -------------------- PREDICTION & VISUALIZATION --------------------
def predict_and_visualize(image: Image.Image, score_thresh=0.4):
    img_tensor = transform(image).to(DEVICE)
    with torch.no_grad():
        pred = model([img_tensor])

    draw = ImageDraw.Draw(image)
    boxes = pred[0]["boxes"].cpu().numpy()
    labels = pred[0]["labels"].cpu().numpy()
    scores = pred[0]["scores"].cpu().numpy()

    for box, label, score in zip(boxes, labels, scores):
        if score < score_thresh:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = "red" if label == 2 else "green"
        text = f"{'Occupied' if label == 2 else 'Vacant'} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1, y1-10), text, fill=color)

    return image

# -------------------- PARKING FINDER --------------------
class ParkingFinder:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="parking_finder")

    def get_coordinates(self, address):
        try:
            location = self.geolocator.geocode(address)
            if location:
                return (location.latitude, location.longitude)
            return None
        except:
            return (19.0760, 72.8777)

    def find_nearby_parking(self, center_point, radius=1000):
        try:
            tags = {'amenity': 'parking'}
            parking_areas = ox.features_from_point(center_point, tags=tags, dist=radius)
            return parking_areas
        except:
            return pd.DataFrame()

    def get_nearest_parking(self, user_location, parking_data):
        if parking_data.empty:
            return None

        nearest_parking = None
        min_distance = float('inf')

        for idx, parking in parking_data.iterrows():
            try:
                if hasattr(parking, 'geometry'):
                    parking_center = parking.geometry.centroid
                    parking_coords = (parking_center.y, parking_center.x)
                    distance = ox.distance.great_circle_vec(
                        user_location[0], user_location[1],
                        parking_coords[0], parking_coords[1]
                    )
                    if distance < min_distance:
                        min_distance = distance
                        nearest_parking = {
                            'name': parking.get('name', 'Unknown Parking'),
                            'coordinates': parking_coords,
                            'distance': round(distance, 2),
                            'type': parking.get('parking', 'Unknown'),
                            'address': parking.get('addr:street', 'Address not available')
                        }
            except:
                continue
        return nearest_parking

    def create_map(self, user_location, parking_spots, nearest_parking):
        m = folium.Map(location=user_location, zoom_start=15)
        folium.Marker(user_location, popup='Your Location', icon=folium.Icon(color='red', icon='user')).add_to(m)

        if not parking_spots.empty:
            for idx, parking in parking_spots.iterrows():
                try:
                    if hasattr(parking, 'geometry'):
                        center = parking.geometry.centroid
                        folium.Marker(
                            [center.y, center.x],
                            popup=f"Parking: {parking.get('name', 'Unknown')}",
                            icon=folium.Icon(color='blue', icon='car')
                        ).add_to(m)
                except:
                    continue

        if nearest_parking:
            folium.Marker(
                nearest_parking['coordinates'],
                popup=f"NEAREST: {nearest_parking['name']} - {nearest_parking['distance']}m",
                icon=folium.Icon(color='green', icon='flag')
            ).add_to(m)
            folium.PolyLine([user_location, nearest_parking['coordinates']], color='green', weight=3, opacity=0.8).add_to(m)
        else:
            folium.Marker(user_location, popup='No parking found nearby.', icon=folium.Icon(color='orange', icon='info-sign')).add_to(m)

        return m

# -------------------- STREAMLIT UI --------------------
st.title("🚗 Parking Space Detection & Nearby Parking Finder 🅿️")

# Image detection section
st.header("1️⃣ Parking Space Detection from Image")
uploaded_file = st.file_uploader("Upload a parking image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.write("Detecting parking spaces...")
        result_img = predict_and_visualize(image)
        st.image(result_img, caption="Prediction", use_column_width=True)

# Nearby parking section
st.header("2️⃣ Find Nearby Parking")
location_input = st.text_input("Enter your location (city, area, or address):")
radius_input = st.slider("Search radius (meters):", min_value=500, max_value=5000, value=2000, step=500)

if location_input:
    finder = ParkingFinder()
    user_coords = finder.get_coordinates(location_input)
    st.write(f"Your coordinates: {user_coords}")

    parking_data = finder.find_nearby_parking(user_coords, radius=radius_input)
    nearest_parking = finder.get_nearest_parking(user_coords, parking_data)

    if nearest_parking:
        st.success(f"Nearest Parking: {nearest_parking['name']} - {nearest_parking['distance']} meters away")
    else:
        st.warning("No parking found nearby.")

    st_folium(finder.create_map(user_coords, parking_data, nearest_parking), width=700, height=500)
