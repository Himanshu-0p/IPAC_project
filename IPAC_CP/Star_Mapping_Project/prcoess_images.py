import os
import cv2
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Preprocess the Image
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary_img = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    return binary_img

# Star Detection
def detect_stars(binary_img):
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    star_coords = [cv2.moments(cnt) for cnt in contours]
    star_coords = [(int(m['m10']/m['m00']), int(m['m01']/m['m00'])) for m in star_coords if m['m00'] != 0]
    return star_coords

# Graph Construction
def create_graph(star_coords, distance_threshold=50):
    G = nx.Graph()
    for i, coord in enumerate(star_coords):
        G.add_node(i, pos=coord)

    for i in range(len(star_coords)):
        for j in range(i + 1, len(star_coords)):
            dist = np.linalg.norm(np.array(star_coords[i]) - np.array(star_coords[j]))
            if dist <= distance_threshold:
                G.add_edge(i, j, weight=dist)
    
    return G

# Save the Graph
def save_graph(G, file_name):
    nx.write_edgelist(G, file_name, data=['weight'])

# Process all images in the directory
def process_images_in_directory(directory_path, output_dir):
    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            binary_img = preprocess_image(image_path)
            star_coords = detect_stars(binary_img)
            G = create_graph(star_coords)
            graph_filename = os.path.splitext(filename)[0] + '_graph.edgelist'
            save_graph(G, os.path.join(output_dir, graph_filename))
            print(f"Processed {filename} and saved graph as {graph_filename}")

            # Optional: Visualize the graph
            pos = nx.get_node_attributes(G, 'pos')
            nx.draw(G, pos, with_labels=True, node_size=20, font_size=10)
            plt.show()

# Example Usage
process_images_in_directory('Star_Mapping_Project/dataset', 'Star_Mapping_Project/graphs')
