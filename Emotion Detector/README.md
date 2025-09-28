# Real-Time Emotion Detection System

A deep learning-based emotion detection system that uses Convolutional Neural Networks (CNN) to classify facial expressions in real-time using computer vision.

## 🎯 Features

- **Real-time emotion detection** from webcam feed
- **7 emotion categories**: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- **Custom CNN architecture** built with PyTorch
- **Face detection** using OpenCV Haar Cascades
- **Jupyter Notebook implementation** for easy experimentation
- **Compatible with Python 3.13+**

## 📊 Dataset

This project uses the **FER-2013 (Facial Expression Recognition 2013)** dataset:
- **Training samples**: ~28,000 images
- **Test samples**: ~7,000 images  
- **Image size**: 48x48 grayscale
- **Classes**: 7 emotional expressions

The dataset is automatically downloaded using Kaggle Hub integration.

## 🏗️ Model Architecture

**EmotionCNN** - Custom Convolutional Neural Network:
```
- Conv2D (1→32) + ReLU + MaxPool2D
- Conv2D (32→64) + ReLU + MaxPool2D  
- Conv2D (64→128) + ReLU + MaxPool2D
- Fully Connected (128*6*6 → 512) + ReLU + Dropout
- Output Layer (512 → 7)
```

## 🚀 Quick Start

### Prerequisites
```bash
# Required packages
pip install torch torchvision
pip install opencv-python-headless  # For Python 3.13+
pip install kagglehub
pip install jupyter
pip install matplotlib pandas numpy pillow
```

### Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/emotion-detector.git
   cd emotion-detector
   ```

2. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook emotion_detector.ipynb
   ```

3. **Run the cells sequentially**:
   - **Cell 1-2**: Download and explore the FER-2013 dataset
   - **Cell 3-4**: Load and preprocess the data
   - **Cell 5-6**: Define and initialize the CNN model
   - **Cell 7-8**: Train the model (20 epochs)
   - **Cell 9**: Evaluate model performance
   - **Cell 10**: Run real-time emotion detection

## 📈 Model Performance

The CNN achieves the following performance on the FER-2013 test set:
- **Test Accuracy**: ~65-70% (typical for FER-2013)
- **Training Time**: ~30-45 minutes (20 epochs on CPU)
- **Inference Speed**: Real-time detection at ~15-20 FPS

### Emotion Class Distribution:
```
Happy: ~25% | Neutral: ~15% | Sad: ~15%
Angry: ~12% | Surprise: ~10% | Fear: ~12% | Disgust: ~11%
```

## 🛠️ Technical Implementation

### Key Components:

1. **Data Pipeline**:
   - Automatic dataset download via Kaggle Hub
   - Image preprocessing with normalization
   - Data augmentation (horizontal flip, rotation)

2. **Model Training**:
   - Adam optimizer with learning rate 0.001
   - Cross-entropy loss function
   - Dropout regularization (25%)

3. **Real-time Detection**:
   - OpenCV for face detection and video capture
   - PyTorch for emotion classification
   - IPython display for notebook visualization

### Compatible Environments:
- ✅ **Python 3.13+** (with opencv-python-headless)
- ✅ **Windows, macOS, Linux**
- ✅ **CPU and GPU** (CUDA support)
- ✅ **Jupyter Notebook/Lab**

## 🔧 Troubleshooting

### Common Issues:

**OpenCV Installation (Python 3.13)**:
```bash
pip uninstall opencv-python opencv-python-headless -y
pip install opencv-python-headless --force-reinstall --no-deps
```

**Camera Access Issues**:
- The system gracefully handles cases where no camera is available
- Falls back to test image display mode

**Memory Issues**:
- Reduce batch size from 64 to 32 or 16
- Use CPU instead of GPU if CUDA memory is insufficient

## 📁 Project Structure

```
emotion-detector/
├── emotion_detector.ipynb      # Main Jupyter notebook
├── README.md                   # This file
├── haarcascade_frontalface_default.xml  # Face detection cascade (auto-downloaded)
└── requirements.txt           # Python dependencies (optional)
```

## 🎯 Usage Examples

### Training from Scratch:
```python
# Run cells 1-8 sequentially to train a new model
model = EmotionCNN(num_classes=7)
# ... training code in notebook
```

### Real-time Detection:
```python
# After training, run the emotion detection cell
# Detects emotions from webcam feed for 10 seconds
# Press Ctrl+C to interrupt
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] Data augmentation techniques
- [ ] Advanced model architectures (ResNet, EfficientNet)
- [ ] Real-time performance optimization
- [ ] Web interface development
- [ ] Mobile app integration

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **FER-2013 Dataset**: Challenges in Representation Learning: A report on three machine learning contests (ICML 2013)
- **OpenCV**: Computer vision library for face detection
- **PyTorch**: Deep learning framework for model implementation
- **Kaggle**: Dataset hosting and community

## 📞 Contact

Feel free to reach out for questions, suggestions, or collaborations!

---

⭐ **Star this repository** if you found it helpful!