

from ultralytics import YOLO

import numpy as np


model = YOLO('runs-20240311T165335Z-001/runs/classify/train6/weights/last.pt')  # load a custom model

results = model('images/download (2).jpeg')  # predict on an image

names_dict = results[0].names

probs = results[0].probs.data.tolist()

#print(names_dict)
#print(probs)

#print(names_dict[np.argmax(probs)])


label_mapping = {
    'catterpillar': ' Caterpillar',
    'grasshopper': 'Locusts ',
    'slug': 'Slug',
    'snail': 'Gastropoda',
    'weevil': 'Curculionoidea'
}

predicted_label = names_dict[np.argmax(probs)]  # Get predicted label with highest probability
display_label = label_mapping.get(predicted_label, predicted_label)  # Map or use default

print(names_dict)
print(probs)

print(f"Predicted Label (Display): {display_label}")





