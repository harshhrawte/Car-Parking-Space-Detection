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
import time

# -------------------- DEVICE CONFIG --------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- MODEL LOADING --------------------
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
transform = transforms.Compose([transforms.ToTensor()])

# -------------------- PREDICTION --------------------
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
        self.geolocator = Nominatim(user_agent="worldwide_parking_finder")

    @st.cache_data(show_spinner=False)
    def get_coordinates(_self, address: str):
        try:
            location = _self.geolocator.geocode(address, timeout=15, language="en")
            if location:
                return (float(location.latitude), float(location.longitude))
            st.warning("Could not find location. Using default Mumbai.")
            return (19.0760, 72.8777)
        except Exception as e:
            st.error(f"Geocoding error: {e}")
            return (19.0760, 72.8777)

    @st.cache_data(show_spinner=False)
    def find_nearby_parking(_self, center_point, radius=1000, max_radius=10000):
        tags_list = [
            {"amenity":"parking"}, {"landuse":"parking"},
            {"highway":"rest_area"}, {"parking":"surface"},
            {"parking":"underground"}, {"parking":"multi-storey"}
        ]
        all_parking = pd.DataFrame()
        for tags in tags_list:
            try:
                parking_data = ox.features_from_point(center_point, tags=tags, dist=radius)
                if not parking_data.empty:
                    all_parking = pd.concat([all_parking, parking_data], ignore_index=True)
            except Exception:
                time.sleep(1)
                continue
        if not all_parking.empty:
            all_parking = all_parking.drop_duplicates(subset='geometry')
            st.success(f"Found {len(all_parking)} parking spots around you!")
            return all_parking
        if radius < max_radius:
            time.sleep(1)
            return _self.find_nearby_parking(center_point, radius+2000, max_radius)
        return pd.DataFrame()

    def get_nearest_parking(self, user_location, parking_data):
        if parking_data.empty:
            return None
        nearest = None
        min_dist = float("inf")
        for _, row in parking_data.iterrows():
            try:
                if hasattr(row,"geometry"):
                    center = row.geometry.centroid
                    coords = (center.y, center.x)
                    dist = ox.distance.great_circle_vec(user_location[0], user_location[1], coords[0], coords[1])
                    if dist < min_dist:
                        min_dist = dist
                        nearest = {
                            "name": row.get("name","Unnamed Parking"),
                            "coordinates": coords,
                            "distance": round(dist,1),
                            "address": row.get("addr:street","No address info")
                        }
            except Exception:
                continue
        return nearest

    def create_map(self, user_location, parking_data, nearest_parking):
        m = folium.Map(location=user_location, zoom_start=15)
        folium.Marker(user_location, popup="Your Location", icon=folium.Icon(color="red", icon="user")).add_to(m)
        if not parking_data.empty:
            for _, row in parking_data.iterrows():
                try:
                    center = row.geometry.centroid
                    folium.Marker([center.y, center.x], popup=f"Parking: {row.get('name','Unknown')}",
                                  icon=folium.Icon(color="blue", icon="car")).add_to(m)
                except Exception:
                    continue
        if nearest_parking:
            folium.Marker(nearest_parking["coordinates"],
                          popup=f"Nearest: {nearest_parking['name']} ({nearest_parking['distance']} m)",
                          icon=folium.Icon(color="green", icon="flag")).add_to(m)
            folium.PolyLine([user_location, nearest_parking["coordinates"]],
                            color="green", weight=3, opacity=0.7).add_to(m)
        return m

# -------------------- STREAMLIT UI --------------------
st.title("🌍 Global Parking Space Detection & Finder 🅿️")

# Section 1: Parking Detection
st.header("1️⃣ Parking Space Detection from Image")
uploaded_file = st.file_uploader("Upload parking lot image...", type=["jpg","jpeg","png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    col1, col2 = st.columns(2)
    with col1: st.image(image, caption="Original Image", use_column_width=True)
    with col2:
        st.info("Detecting parking spaces...")
        result = predict_and_visualize(image)
        st.image(result, caption="Prediction Result", use_column_width=True)

# Section 2: Worldwide Parking Finder
st.header("2️⃣ Find Nearby Parking (OSM only, Worldwide 🌐)")
location_input = st.text_input("Enter your location (anywhere in the world):")
radius = st.slider("Search radius (meters)", 500, 10000, 2000, step=500)

if location_input:
    finder = ParkingFinder()
    coords = finder.get_coordinates(location_input)
    st.write(f"📍 Coordinates: {coords}")
    data = finder.find_nearby_parking(coords, radius)
    nearest = finder.get_nearest_parking(coords, data)

    if nearest:
        st.success(f"Nearest Parking: **{nearest['name']}** — {nearest['distance']} m away")
    else:
        st.warning("No parking data found via OpenStreetMap.")
        google_url = f"https://www.google.com/maps/search/parking+near+{location_input.replace(' ','+')}"
        st.markdown(f"[🔗 Search on Google Maps instead]({google_url})", unsafe_allow_html=True)

    st_folium(finder.create_map(coords, data, nearest), width=700, height=500)
