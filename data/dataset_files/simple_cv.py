import numpy as np
import cv2
import sys
import pandas as pd

def process_single_image(image_path):
    # Učitaj sliku i konvertuj u RGB
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Izdvoj ROI (Region of Interest)
    img = img[250:800, 200:800]
    
    # Konverzija u grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    # Binarizacija slike
    img_bin = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 61, 31)
    img_bin = cv2.bitwise_not(img_bin)
    
    # Priprema kernela za morfološke operacije
    kernel_ero = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    img_ero = cv2.erode(img_bin, kernel_ero, iterations=2)
    img_open = cv2.dilate(img_ero, kernel_open, iterations=12)
    img_ero = cv2.erode(img_open, kernel_ero, iterations=26)
    
    # Pronalaženje kontura
    contours, hierarchy = cv2.findContours(img_ero, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtriranje kontura na osnovu definisanih uslova
    duck_contours = []
    for contour in contours:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        
        is_in_corner = (x < 100 and y > 300) or (x > 500 and y > 500) or (x < 200 and y > 500) or (y < 50 and x > 500) or (x < 100 and y < 100)
        is_grass = (350 < x < 400) and (y > 480)
        is_on_edge = x > 580 or y < 20

        if 14 < radius < 75 and not is_in_corner and not is_on_edge and not is_grass:
            duck_contours.append(contour)

    return len(duck_contours)

def main(folder_path):
    csv_path = folder_path + 'duck_count.csv'
    duck_data = pd.read_csv(csv_path)
    
    absolute_errors = []


    for index, row in duck_data.iterrows():
        image_name = row['picture']
        true_ducks = row['ducks']
        

        image_path = folder_path + image_name
        predicted_ducks = process_single_image(image_path)
        

        absolute_error = abs(true_ducks - predicted_ducks)
        absolute_errors.append(absolute_error)
    

    mae = sum(absolute_errors) / len(absolute_errors)
    print(mae)


if __name__ == "__main__":
    folder_path = sys.argv[1]
    main(folder_path)