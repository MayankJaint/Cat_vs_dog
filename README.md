# Cat vs Dog Image Classification

A PyTorch convolutional neural network for binary cat-versus-dog image classification. The project includes a training and evaluation notebook plus a Streamlit app for testing individual images.

## Features

- Resize and normalize cat and dog images
- Split data into training, validation, and test sets
- Train a custom three-block CNN with PyTorch
- Evaluate with loss, accuracy, a confusion matrix, and a classification report
- Export model weights to a reusable checkpoint
- Upload an image in Streamlit and view the predicted class and probabilities

## Project Structure

```text
CatVsDog/
|-- Cat_vs_Dog.ipynb          # Data preparation, training, and evaluation
|-- app.py                    # Streamlit image-testing application
|-- requirements.txt          # Python dependencies
|-- simple_cnn_cats_dogs.pt   # Exported model checkpoint
`-- README.md
```

The project also uses the `dogs-vs-cats` dataset directory. Keep the training images available at the path configured in the notebook before running the training workflow.

## Requirements

- Python 3.10 or newer
- A CPU or CUDA-compatible GPU
- The Dogs vs Cats image dataset

## Installation on Windows

Open PowerShell in this project directory and create a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks environment activation, run the following command in an Administrator PowerShell only when appropriate for your machine policy:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Training and Evaluation

1. Open `Cat_vs_Dog.ipynb` in VS Code or Jupyter.
2. Run the cells in order.
3. Set `dataset_path` in the data import cell to the directory containing the training images.
4. Complete preprocessing, data loading, model creation, and training.
5. Run the evaluation cells to generate test metrics and visualizations.
6. Run the **Single-Image Inference and Streamlit Testing** export cell. It writes:

```text
simple_cnn_cats_dogs.pt
```

The notebook currently contains a Google Colab drive-mount cell. When running locally, skip that cell and use a local dataset path instead.

## Model and Preprocessing Contract

The model is a custom `SimpleCNN` with three convolutional blocks and two fully connected layers. Images must be processed as follows:

- Convert BGR input to RGB
- Resize to `128 x 128` pixels
- Scale pixel values from `[0, 255]` to `[0, 1]`
- Convert from `(height, width, channels)` to `(channels, height, width)`
- Use class index `0` for `Cat` and class index `1` for `Dog`

The Streamlit app applies the same preprocessing before inference.

## Run the Streamlit App

From the project directory, activate the virtual environment and run:

```powershell
.\venv\Scripts\Activate.ps1
streamlit run app.py
```

Then open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

Upload a `.jpg`, `.jpeg`, `.png`, or `.webp` image. The app displays the image, predicted class, confidence, and probability for each class.

## Checkpoint

The app loads `simple_cnn_cats_dogs.pt` from the same directory as `app.py`. If the file is missing, run the notebook export cell after training. The checkpoint stores:

- `model_state_dict`
- `class_names`
- `image_size`

Do not replace the checkpoint with weights from a different architecture unless `app.py` is updated to match that architecture.

## Dataset Naming

The notebook derives labels from image filenames. It expects names beginning with `cat` or `dog`, such as:

```text
cat.0.jpg
dog.0.jpg
```

Images should be readable by OpenCV and stored in the directory supplied through `dataset_path`.

## Troubleshooting

**Checkpoint not found**

Run the notebook export cell and confirm that `simple_cnn_cats_dogs.pt` is beside `app.py`.

**Missing Python packages**

Activate the project virtual environment and run:

```powershell
python -m pip install -r requirements.txt
```

**Notebook cannot find the dataset**

Update `dataset_path` to the local directory containing the training images. The default notebook path is intended for Google Colab and is not a local Windows path.

**CUDA is unavailable**

The notebook and app automatically fall back to CPU execution. Training and inference may be slower without a GPU.
