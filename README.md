
# ✈️ End-to-End MLOps Pipeline: Airline Passenger Satisfaction

## 📖 Project Overview
This project demonstrates a fully automated **MLOps pipeline** deployed on **AWS**. It predicts airline passenger satisfaction (Churn Risk) using a **Random Forest** classifier. 

Unlike standard notebook projects, this solution focuses on **production engineering**:
- **Reproducibility:** The training environment is encapsulated in a custom **Docker container**.
- **Automation:** A **CI/CD pipeline** (GitHub Actions) automatically triggers model retraining in the cloud whenever code changes.
- **Scalability:** Uses **AWS SageMaker** for on-demand compute resources.

## 🛠️ Tech Stack
- **Cloud Infrastructure:** AWS SageMaker, S3, ECR (Elastic Container Registry)
- **Containerization:** Docker
- **Orchestration:** GitHub Actions (CI/CD)
- **Machine Learning:** Scikit-Learn, Pandas, Joblib
- **IaC / SDK:** Boto3 (AWS SDK for Python)

## 🏗️ Architecture
The pipeline follows a "Code-First" approach:

1.  **Data Ingestion:** Raw `Airline.csv` data is stored in **Amazon S3**.
2.  **Container Build:** A **Docker** image containing the training logic (`model.py`) is built and pushed to **Amazon ECR**.
3.  **CI/CD Trigger:** A push to the `main` branch triggers **GitHub Actions**.
4.  **Cloud Training:** GitHub Actions executes `trigger_training.py`, which instructs **SageMaker** to spin up an `ml.m5.xlarge` instance, download the data, train the model, and save the artifacts back to S3.

## 📂 Project Structure
```bash
airline-churn-mlops/
├── .github/workflows/
│   └── mlops-pipeline.yml   # CI/CD definition for GitHub Actions
├── data/                    # Local data folder (excluded from git)
├── models/                  # Local model artifacts
├── Dockerfile               # Blueprint for the custom ML container
├── model.py                 # The core ML training script (Scikit-Learn)
├── requirements.txt         # Python dependencies
└── trigger_training.py      # Boto3 script to launch SageMaker jobs
