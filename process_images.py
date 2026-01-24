from PIL import Image
import os
import pandas as pd

tifs = ["test-labels.tif", "test-volume.tif", "train-labels.tif", "train-volume.tif"]

for tif in tifs:
    image = Image.open(f"unprocessed_data/{tif}")

    train_or_test, image_or_label = tif.split("-")

    if image_or_label[:-4] == "labels":
        image_or_label = "labels"
    elif image_or_label[:-4] == "volume":
        image_or_label = "images"
    destination_directory = f"processed_data/{train_or_test}/{image_or_label}" 
        
    for i in range(image.n_frames):
        image.seek(i)
        image.save(f"{destination_directory}/image_{i}.tif")
    
for test_or_train in ["test", "train"]:
    data = []
    directory = f"processed_data/{test_or_train}"
    for file in os.listdir(f"{directory}/images"):
        data.append([f"{directory}/images/{file}", f"{directory}/labels/{file}"])

    df = pd.DataFrame(data, columns=["images", "labels"])
    df.to_csv(f"processed_data/{test_or_train}_dataset.csv")