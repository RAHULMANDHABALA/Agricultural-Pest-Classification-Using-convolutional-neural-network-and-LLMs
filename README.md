# Agricultural Pest Classification Using convolutional neural network & LLMs

## Overview
This project aims to develop an AI-powered pest identification system that classifies pest images into five categories and provides effective control recommendations. The system integrates deep learning models for classification and leverages Google's Gemini 2.0 via OpenRouter API for pest management insights.

## Features
- **Pest Classification:** Classifies pests into five categories: Caterpillar, Locusts, Slug, Gastropoda, and Curculionoidea.
- **Deep Learning Models:** Implements MobileNetV2 (97.5%), YOLOv8 (77%), ResNet50, ResNet101V4, and EfficientNet.
- **Data Augmentation:** Uses resizing, normalization, flipping, rotation, zooming, and contrast adjustments for better model generalization.
- **Model Evaluation:** Trained multiple CNN architectures with hyperparameter tuning and early stopping.
- **Deployment:** A Streamlit web application enables pest classification.
- **LLM Integration:** Uses OpenRouter API with Gemini 2.0 to provide detailed pest insights and control strategies.

## Project Workflow
### 1. Data Preprocessing & Preparation
- Organized pest images into five categories with an 80-20 train-test split.
- Applied preprocessing techniques like resizing, normalization, and augmentation.

### 2. Model Development & Evaluation
- Trained and evaluated various models:
  - **ResNet50:** 80%
  - **ResNet101V4:** 87.7%
  - **MobileNetV2:** 97.5%
  - **YOLOv8:** 77%
  - **EfficientNet:** 89%
- Used early stopping and hyperparameter tuning for optimization.

### 3. Model Selection & Optimization
- Chose **YOLOv8** as the primary model due to its balance of accuracy and generalization.
- Optimized the model for efficient classification.

### 4. Deployment with Streamlit
- Developed an interactive **Streamlit** web application for real-time classification.
- Integrated **OpenRouter API** with **Gemini 2.0 Flash Thinking Model** for pest insights and control strategies.

## Results
- **MobileNetV2 (97.5%)** and **YOLOv8 (77%)** were selected as the best models.
- The Streamlit app successfully classifies pests and provides actionable pest control strategies, including organic, chemical, and integrated pest management approaches.

## Installation & Usage
### Prerequisites
- Python 3.8+
- Streamlit
- TensorFlow/Keras
- OpenRouter API access
- YOLOv8

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Premaramkarthik/Agricultural-Pest-Classification-Using-convolutional-neural-network-and-LLMs
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
4. Upload an image and get classification results along with pest control strategies.

## API Integration
- The system uses OpenRouter API to connect with **Google's Gemini 2.0**, providing detailed insights about identified pests and their control methods.
- Ensure you have an API key and add it to your environment variables.

## Future Enhancements
- Improve YOLOv8 model accuracy.
- Expand dataset to include more pest species.
- Integrate real-time object detection.
- Develop a mobile application for field use.

## Contributing
Feel free to fork this repository, make improvements, and submit a pull request. Contributions are welcome!


## Contact
For questions or collaborations, reach out via:
- **Email:** mandhabalarahul@gmail.com



