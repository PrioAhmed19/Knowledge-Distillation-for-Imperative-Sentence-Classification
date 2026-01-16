# Knowledge Distillation for Imperative Sentence Classification

A PyTorch implementation of Knowledge Distillation for compressing a BERT-base model into a smaller BERT-tiny model for imperative sentence classification in code-mixed language (Bengali-English).

## 📋 Overview

This project demonstrates how to transfer knowledge from a large teacher model (BERT-base) to a smaller student model (BERT-tiny) using knowledge distillation. The resulting student model is **24.96x smaller** while retaining **98.53%** of the teacher's performance.

### Key Results

| Model | Parameters | Test Accuracy | Size Reduction |
|-------|-----------|---------------|----------------|
| Teacher (BERT-base) | 109.5M | 97.65% | - |
| Student (BERT-tiny) | 4.4M | 96.21% | 24.96x |

## 🎯 Features

- **Knowledge Distillation**: Transfer learning from large to small models
- **Custom Loss Function**: Combines soft targets (teacher) and hard labels (ground truth)
- **Temperature Scaling**: Adjustable temperature parameter for probability softening
- **Code-Mixed Language**: Works with Bengali-English mixed text
- **Production Ready**: Compact model suitable for deployment

## 🚀 Quick Start

### Prerequisites

```bash
python >= 3.7
torch >= 1.9.0
transformers >= 4.0.0
scikit-learn
pandas
numpy
tqdm
```

### Installation

```bash
git clone https://github.com/yourusername/knowledge-distillation-imperative.git
cd knowledge-distillation-imperative
pip install -r requirements.txt
```

### Download Dataset

The dataset will be automatically downloaded when you run the notebook, or you can download it manually:

```bash
wget -O Final-dataset.csv "https://www.dropbox.com/scl/fi/v7olloa8to9ixjvp3my2l/Final-dataset.csv?rlkey=zk1aasrpcaop79cfgogufs76q&st=jf24beip&dl=0"
```

### Training

Run the Jupyter notebook or execute the training script:

```bash
jupyter notebook KD.ipynb
```

Or convert to Python script and run:

```bash
jupyter nbconvert --to script KD.ipynb
python KD.py
```

## 📊 Dataset

The dataset contains code-mixed (Bengali-English) sentences labeled as:
- **0**: Non-Imperative
- **1**: Imperative

**Dataset Split:**
- Training: 70% (6,157 samples)
- Validation: 15% (1,320 samples)
- Test: 15% (1,320 samples)

## 🏗️ Architecture

### Teacher Model
- **Model**: `bert-base-uncased`
- **Parameters**: 109.5M
- **Training**: Standard fine-tuning with cross-entropy loss

### Student Model
- **Model**: `prajjwal1/bert-tiny`
- **Parameters**: 4.4M
- **Training**: Knowledge distillation with combined loss

### Distillation Loss

The loss function combines two components:

```
L_total = α * L_distillation + (1 - α) * L_student
```

Where:
- **L_distillation**: KL divergence between student and teacher soft predictions
- **L_student**: Cross-entropy loss with true labels
- **α (alpha)**: Balance parameter (default: 0.7)
- **T (temperature)**: Softening parameter (default: 3.0)

## 💻 Usage Example

### Inference with Trained Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained('./distilled_student_model')
tokenizer = AutoTokenizer.from_pretrained('./distilled_student_model')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

# Predict function
def predict_imperative(text):
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = F.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1)
    
    return "Imperative" if pred.item() == 1 else "Non-Imperative", probs[0].cpu().numpy()

# Test
text = "Please submit your assignment by Friday"
label, probs = predict_imperative(text)
print(f"Prediction: {label}")
print(f"Probabilities: Non-Imp={probs[0]:.3f}, Imp={probs[1]:.3f}")
```

## 📈 Training Progress

### Teacher Model (5 epochs)
```
Epoch 1/5 - Loss: 0.2009 - Val Accuracy: 0.9682
Epoch 2/5 - Loss: 0.0557 - Val Accuracy: 0.9727
Epoch 3/5 - Loss: 0.0311 - Val Accuracy: 0.9735
Epoch 4/5 - Loss: 0.0155 - Val Accuracy: 0.9712
Epoch 5/5 - Loss: 0.0091 - Val Accuracy: 0.9705
```

### Student Model with Distillation (5 epochs)
```
Epoch 1/5 - Total: 0.2883 - Distill: 0.3601 - CE: 0.1207 - Val Acc: 0.9538
Epoch 2/5 - Total: 0.2344 - Distill: 0.2925 - CE: 0.0990 - Val Acc: 0.9568
Epoch 3/5 - Total: 0.2159 - Distill: 0.2703 - CE: 0.0892 - Val Acc: 0.9629
Epoch 4/5 - Total: 0.1819 - Distill: 0.2267 - CE: 0.0773 - Val Acc: 0.9606
Epoch 5/5 - Total: 0.1584 - Distill: 0.1966 - CE: 0.0693 - Val Acc: 0.9598
```

## 🔬 Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Temperature | 3.0 | Controls softness of probability distributions |
| Alpha | 0.7 | Balance between teacher knowledge and ground truth |
| Learning Rate (Teacher) | 2e-5 | Teacher model learning rate |
| Learning Rate (Student) | 5e-5 | Student model learning rate |
| Batch Size | 16 | Training batch size |
| Max Length | 128 | Maximum sequence length |
| Epochs (Teacher) | 5 | Teacher training epochs |
| Epochs (Student) | 5 | Student training epochs |

## 📁 Project Structure

```
knowledge-distillation-imperative/
│
├── KD.ipynb                      # Main training notebook
├── README.md                     # Project documentation
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore file
├── LICENSE                       # License file
│
├── data/
│   └── Final-dataset.csv         # Dataset (downloaded automatically)
│
├── models/
│   ├── best_student_model.pt     # Best student model checkpoint
│   └── distilled_student_model/  # Final saved model
│       ├── config.json
│       ├── pytorch_model.bin
│       └── tokenizer files
│
└── src/
    ├── dataset.py                # Dataset class
    ├── model.py                  # Model definitions
    ├── loss.py                   # Distillation loss
    └── utils.py                  # Utility functions
```

## 🎓 Key Concepts

### What is Knowledge Distillation?

Knowledge distillation is a model compression technique where:
1. A large "teacher" model is trained on the task
2. A smaller "student" model learns to mimic the teacher
3. The student learns from both teacher's soft predictions and true labels
4. Result: Compact model with near-teacher performance

### Why Use Temperature?

Temperature (T) softens the probability distribution:
- **High T**: More uniform probabilities → reveals subtle patterns
- **Low T**: Sharp probabilities → focuses on confident predictions
- During distillation, we use T=3.0 to transfer "dark knowledge"

### Benefits

✅ **24x smaller model** - Reduced memory footprint  
✅ **98.5% performance retention** - Minimal accuracy loss  
✅ **Faster inference** - Suitable for production deployment  
✅ **Lower computational cost** - Ideal for edge devices  

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 References

- [Hinton et al., "Distilling the Knowledge in a Neural Network"](https://arxiv.org/abs/1503.02531)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)

## 👤 Author

Your Name - [@Prio](https://www.linkedin.com/in/refat-ahmed19/)

Project Link: [https://github.com/Prioahmed19/knowledge-distillation-imperative](https://github.com/PrioAhmed19/knowledge-distillation-imperative)

## 🙏 Acknowledgments

- Dataset created by Gemini Api
- Hugging Face team for the Transformers library
- PyTorch team for the deep learning framework

---

⭐ If you find this project useful, please consider giving it a star!
