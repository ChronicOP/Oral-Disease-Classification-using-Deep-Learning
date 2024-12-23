# Oral-Disease-Classification-using-Deep-Learning
This repository provides a deep learning solution for classifying oral diseases from images. It uses a pre-trained ResNet-18 model to classify conditions like healthy, dental caries, and gingivitis. The project handles variations in image quality, lighting, and orientation, ensuring robust performance across diverse demographics.
Key Features

	•	Multiclass Classification: Predicts one class per image with confidence scores for each disease.
	•	Robust Preprocessing: Includes resizing, normalization, and augmentation to generalize well across various datasets.
	•	Deep Learning Architecture: Utilizes a ResNet-18 backbone with a custom classification head.
	•	Comprehensive Evaluation: Includes confusion matrices, classification reports, and misclassification analysis.

Repository Structure

.
├── dataset/                   # Dataset folder (TRAIN and TEST subdirectories)
├── models/                    # Trained model weights
├── notebooks/                 # Jupyter notebooks for training and testing
├── src/                       # Python scripts for preprocessing, training, and evaluation
├── outputs/                   # Evaluation results (confusion matrix, classification reports)
├── README.md                  # Project description and usage guide
└── requirements.txt           # Python dependencies

Getting Started

Follow these steps to set up and run the project on your machine.

Prerequisites

	•	Python 3.7 or higher
	•	PyTorch, torchvision
	•	scikit-learn
	•	matplotlib
	•	PIL (Pillow)

Install all dependencies using:

pip install -r requirements.txt

Dataset

Download the dataset from this link. Place the dataset in the dataset/ directory and ensure it is organized into TRAIN and TEST subfolders, with one subfolder per class.

Running the Code

	1.	Training the Model
Run the training script to train the model on the provided dataset:

python src/train.py


	2.	Testing the Model
Evaluate the model on the test dataset:

python src/test.py


	3.	View Results
	•	The confusion matrix and classification report will be saved in the outputs/ directory.
	•	The trained model weights will be saved in the models/ folder as oral_disease_model.pth.

Jupyter Notebook

For interactive exploration, use the provided notebook in the notebooks/ directory:

jupyter notebook notebooks/Oral_Disease_Classification.ipynb

Results

	•	Model Performance:
	•	Accuracy: XX.XX%
	•	Precision, Recall, F1-Score: XX.XX%
	•	Confusion Matrix:


Deliverables

	1.	Python Scripts: Preprocessing, training, and evaluation scripts.
	2.	Trained Model: Pre-trained weights for reproducibility.
	3.	Documentation: Comprehensive report with metrics, confusion matrix, and insights.
	4.	Deployment Demo: Video showing predictions on test images.

Limitations & Future Work

	•	Limitations:
	•	Small dataset size may limit generalization.
	•	High variation in lighting and image quality affects predictions.
	•	Future Work:
	•	Include more diverse datasets for training.
	•	Experiment with advanced architectures like EfficientNet or ViT.

Acknowledgements

	•	Dataset provided strictly for academic purposes.
	•	Pre-trained ResNet-18 model from PyTorch’s model zoo.

License

This project is licensed under the MIT License.

Author

Prathamesh Patil
Engineering Intern, AI & ML Enthusiast
Feel free to connect via GitHub or LinkedIn for feedback or collaboration!

This description is professional, informative, and highlights all the key aspects of the project. You can modify the dataset link, metrics, and other details as you finalize your project.
