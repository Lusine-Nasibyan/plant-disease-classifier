# Plant Disease Classifier

Classifies plant diseases from leaf images, independent of host plant species.
Input: leaf image. Output: disease name (e.g. "late blight", not "tomato late blight").

**API**: https://lusinen2004-plant-disease-api.hf.space/docs
**Experiments**: https://wandb.ai/lusinen2004-independent/plant-disease-classifier/overview
**Weights**: https://huggingface.co/Lusinen2004/plant-disease-classifier/tree/main
**Report**: reports/report.pdf

---

## Results

| Model | mAP | Top-1 Acc | Params |
|---|---|---|---|
| EfficientNet-B0 | **83.91%** | 77.44% | 4.06M |
| MobileNetV3-Small | 78.32% | 71.28% | 1.56M |

Evaluated on 390 images across 39 disease classes (10 per class).

---

## Setup

```powershell
git clone https://github.com/YOUR_USERNAME/plant-disease-classifier.git
cd plant-disease-classifier
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place dataset at `data/raw/train/` and `data/raw/val/`, then:

```powershell
python src/prepare_data.py  
python src/dataset.py       
python src/model.py         
```

---

## Training

Requires GPU. Open `notebooks/train_colab.ipynb` in Google Colab (T4 runtime).

```python
run_experiment({"backbone": "efficientnet_b0", "weighted_loss": True,
                "stage1_epochs": 5, "stage2_epochs": 20})
```

All runs logged to Weights & Biases automatically.

---

## Evaluation

```powershell
python src/evaluate.py \
  --checkpoint models/efficientnet_b0__wloss__s1-5_s2-20__best.pt \
  --class-map models/class_mapping.json
```

---

## API

```powershell
cd api
pip install -r requirements.txt
uvicorn app:app --reload
# Swagger UI at http://127.0.0.1:8000/docs
```

**POST /predict** — upload a JPG/PNG, returns predicted disease and top-5 scores.

---

## Structure

| Path | Description |
|---|---|
| `src/prepare_data.py` | Builds labels.csv from raw folder structure |
| `src/dataset.py` | PyTorch Dataset, transforms, class weights |
| `src/model.py` | EfficientNet/MobileNet classifier |
| `src/train.py` | Two-stage training loop |
| `src/evaluate.py` | mAP and per-class AP |
| `notebooks/train_colab.ipynb` | Colab training notebook |
| `api/app.py` | FastAPI endpoint |
| `data/metadata/class_mapping.csv` | Hand-verified disease label map |