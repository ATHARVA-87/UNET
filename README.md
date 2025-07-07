# HumanSegBinary

A modular PyTorch pipeline for **binary human segmentation** using a custom U-Net architecture. This project is designed for training and evaluating deep learning models on binary segmentation tasks where the objective is to segment humans from the background.

---

## 📁 Project Structure

HumanSegBinary/
├── config.yaml # Central config for hyperparameters and paths
├── train.py # Training script with progress bar
├── test.py # Evaluation script with metric logging
└── src/
├── data/
│ └── human_dataset.py # Dataset class and DataLoader factory
├── transforms/
│ └── custom_transforms.py # Transformations for input images/masks
├── models/
│ ├── layers.py # Basic U-Net blocks
│ └── model.py # Full U-Net model
└── utils/
└── metrics.py # Accuracy, Dice, IoU metrics

yaml
Copy
Edit

---

## 🧠 Key Features

- ⚡ U-Net based binary segmentation model  
- 📊 Dice score, IoU, and pixel accuracy metrics  
- ⚙️ Configurable via a single YAML file  
- 🧹 Modular and clean structure  
- 📈 Epoch-wise progress tracking with `tqdm`  
- 📝 Evaluation metrics saved to a `.txt` file  

---

## 📊 Dataset Format

- **Images**: `data/images/*.jpg` or `.png`  
- **Masks**: `data/masks/*.png` (binary masks with values 0 or 1)  
- **Image Size**: `640×480` pixels  

---

## 🚀 Quickstart

1. **Clone the repository:**

   ```bash
   git clone https://github.com/ATHARVA-87/UNET.git
   cd UNET
Prepare your dataset under:

data/images/

data/masks/

Train the model:

bash
Copy
Edit
python train.py
Evaluate the model:

bash
Copy
Edit
python test.py
🧩 Customization
Edit config.yaml to update paths, model parameters, and training options.

Plug in your own transforms or architectures inside the src/ folder as needed.

