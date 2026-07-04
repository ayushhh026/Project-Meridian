# Project Meridian

<div align="center">

```
██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗
██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝
██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║   
██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║   
██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║   
╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝   

                  M  E  R  I  D  I  A  N
```

### A Modular Inference Pipeline — Deployed Two Ways

<br/>

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-Deployed-FF9900?style=for-the-badge&logo=amazonaws&logoColor=white)](https://aws.amazon.com)
[![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/features/actions)
[![Status](https://img.shields.io/badge/Status-Active-22C55E?style=for-the-badge)](.)

<br/>

> **Modular Pipeline · Regression · FastAPI · Dual Deployment (EBS + Docker/ECR/EC2)**

</div>

---

## What is Project Meridian?

Project Meridian is a **modular machine learning pipeline** that separates data ingestion, preprocessing, model training, and inference into independent, reusable components. The pipeline is designed so that the overall architecture remains consistent across structured machine learning problems, while dataset-specific configuration—such as features, preprocessing rules, and prediction schema—can be adapted as needed.

The name reflects that idea: a **meridian** is a fixed line of reference against which everything else is measured. Similarly, the pipeline provides a stable engineering foundation while allowing the data and models to evolve. To demonstrate this, the same ML pipeline has been deployed to production **two different ways** without changing the underlying training or inference logic:

- ✅ **v1 — AWS Elastic Beanstalk** via CodePipeline (managed PaaS with automatic deployment on every push)
- ✅ **v2 — Docker + Amazon ECR + EC2** via GitHub Actions (containerized deployment using a self-hosted runner and end-to-end CI/CD)

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     data_ingestion.py                            │
│        Reads raw source data → train/test split → artifacts     │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                  data_transformation.py                          │
│      ColumnTransformer — imputation, scaling, OHE → pickled     │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     model_trainer.py                             │
│   Benchmarks multiple regressors (incl. CatBoost) → best model  │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                 src/pipeline/predict_pipeline.py                 │
│       Loads model.pkl + preprocessor.pkl → exposes predict()    │
└────────────────────────────┬──────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                         main.py [FastAPI]                        │
│            Thin serving layer — Jinja2 templates + routes       │
└─────────────────────────────────────────────────────────────────┘
```

The web layer has zero knowledge of preprocessing or model internals — `PredictPipeline.predict()` is the only contract between FastAPI and the ML logic underneath it.

---

## Model Selection — Generalization Over Raw Score

`model_trainer.py` benchmarks 7 regressors and selects on test R², not training R² — the metric that actually matters once a model leaves the training set behind.

| Model | Train R² | Test R² | Train–Test Gap |
|---|---|---|---|
| **Linear Regression** | 0.8737 | **0.8816** | **−0.008** |
| Decision Tree | 0.9997 | 0.7368 | 0.263 |
| Random Forest | 0.9768 | 0.8488 | 0.128 |
| AdaBoost | 0.8484 | 0.8464 | 0.002 |
| Gradient Boosting | 0.8950 | 0.8732 | 0.022 |
| CatBoost | 0.9026 | 0.8580 | 0.045 |
| XGBoost | 0.9287 | 0.8515 | 0.077 |

The selection logic deliberately rewards the model with the **smallest generalization gap**, not the most parameters. Decision Tree memorized the training set almost perfectly (0.9997) and then lost over a quarter of its R² on unseen data — the textbook overfitting signature. Every boosted ensemble landed within a respectable 2–5% gap, but none of them out-generalized a properly regularized linear model on this feature set.

Linear Regression came out on top for a structural reason, not a lack of competition: once `data_transformation.py` has already handled categorical encoding, scaling, and imputation, much of the non-linearity in the raw signal has been absorbed upstream. At that point, ensemble methods are spending their extra capacity chasing noise rather than real structure — which is exactly what the train–test gap exposes. The pipeline's job was never to force the most complex model into production; it was to find the one that generalizes best, and the logging shows its work.

---

## Deployment — v1 · AWS Elastic Beanstalk + CodePipeline

The simplest path to production: push to GitHub, CodePipeline picks it up automatically, deploys straight to Elastic Beanstalk. No manual `eb deploy`, no SSH.

```
GitHub (main branch)
        │
        ▼  auto-trigger on push
AWS CodePipeline
        │
        ├── Stage 1: Source   (GitHub via OAuth app)
        │
        └── Stage 2: Deploy   (AWS Elastic Beanstalk)
                    │
                    ▼
        EC2 Instance · Procfile-driven · Nginx → Uvicorn → FastAPI
```

### CodePipeline — Source + Deploy (Both Succeeded ✅)

<img width="1913" height="797" alt="Screenshot 2026-06-28 141306" src="https://github.com/user-attachments/assets/737825f2-dc19-4275-ac42-bfe73b0d9cbb" />

### Live on Elastic Beanstalk

App running on the public EBS URL — `projectmeridian.ap-south-1.elasticbeanstalk.com`.

<img width="1918" height="971" alt="Screenshot 2026-06-28 141441" src="https://github.com/user-attachments/assets/7f0c7ffc-b576-4256-aafe-47eadc91350d" />

---

## Deployment — v2 · Docker + Amazon ECR + EC2 (GitHub Actions CI/CD)

The more involved path: a fully scripted CI/CD pipeline that builds a Docker image, pushes it to a private ECR repository, and deploys it onto an EC2 instance using a **self-hosted GitHub Actions runner**.

```
GitHub (main branch)
        │
        ▼  push triggers .github/workflows/main.yml
┌───────────────────────────────────────────────────┐
│  JOB 1 — Continuous Integration                    │
│  Checkout → Lint → Unit Tests                      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│  JOB 2 — Build & Push Docker Image                 │
│  Configure AWS creds → Login to ECR →              │
│  docker build → docker push                        │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌───────────────────────────────────────────────────┐
│  JOB 3 — Continuous Deployment (self-hosted runner) │
│  Pull latest image → stop old container →           │
│  run new container → prune unused images            │
└───────────────────────────────────────────────────┘
```

### IAM Role — EC2 + ECR Permissions

A dedicated IAM user was created for the GitHub Actions runner with `AmazonEC2ContainerRegistryFullAccess` and `AmazonEC2FullAccess`, scoping exactly what the pipeline needs to push images and manage the deployment target.

<img width="1912" height="812" alt="Screenshot 2026-06-29 183039" src="https://github.com/user-attachments/assets/1767f5d2-c0ce-497b-9e06-f0bc7ecfc2d3" />

### Self-Hosted Runner — Registration on EC2

GitHub Actions doesn't deploy to private infrastructure by default — a self-hosted runner installed directly on the EC2 instance bridges that gap, letting the deployment job execute commands on the actual target machine.

<img width="1918" height="482" alt="Screenshot 2026-06-29 190030" src="https://github.com/user-attachments/assets/e969ee1d-15bd-4f41-8a7e-b2864f2abb9e" />

### Self-Hosted Runner — Active and Idle

Once registered, the runner shows up under the repo's Actions settings, ready to pick up deployment jobs.

<img width="1917" height="867" alt="Screenshot 2026-06-29 190432" src="https://github.com/user-attachments/assets/10e8ea01-2cf7-4c2e-9518-e1b1f4b626fa" />

### Repository Secrets — AWS Credentials

All AWS credentials and ECR configuration are stored as encrypted GitHub repository secrets, never hardcoded into the workflow file.

| Secret | Purpose |
|---|---|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | Target AWS region (ap-south-1) |
| `AWS_ECR_LOGIN_URI` | ECR registry login URI |
| `ECR_REPOSITORY_NAME` | Target ECR repository name |

<img width="1916" height="960" alt="Screenshot 2026-06-29 191641" src="https://github.com/user-attachments/assets/35b08594-9264-4ca7-8711-25dcd68e338d" />

### CI/CD Pipeline — All 3 Jobs Succeeded

The full pipeline — Continuous Integration → Build & Push Docker Image → Continuous Deployment — completing end to end in under 2 minutes.

![GitHub Actions — Full CI/CD Pipeline Success](https://github.com/user-attachments/assets/PLACEHOLDER_pipeline_success)

### EC2 Runner — Listening for Jobs, Deployment Succeeded

Terminal on the EC2 instance showing the self-hosted runner picking up the deployment job in real time. The first run failed, the second succeeded — a real debugging cycle, not a one-shot lucky deploy.

![EC2 Terminal — Runner Listening and Job Result](https://github.com/user-attachments/assets/PLACEHOLDER_ec2_terminal)

### Amazon ECR — Docker Image Pushed

The built image lands in a private ECR repository (`project-meridian`), tagged `latest`, ready to be pulled by the deployment job.

![Amazon ECR — Pushed Docker Image](https://github.com/user-attachments/assets/PLACEHOLDER_ecr_image)

### Live on EC2

The containerized app running directly on the EC2 public DNS, port 8000.

![Live on EC2](https://github.com/user-attachments/assets/PLACEHOLDER_ec2_live)

### FastAPI Swagger Docs — Live on EC2

Auto-generated OpenAPI docs (`/docs`) confirm the FastAPI app is fully operational inside the container, with both `GET /` and `GET /predict` routes registered.

![FastAPI Swagger UI on EC2](https://github.com/user-attachments/assets/PLACEHOLDER_swagger_ec2)

### EC2 Instance — Running and Healthy

The target instance (`project-meridian`, `t3.micro`) passing all 3/3 status checks.

![EC2 Instance — Running, 3/3 Checks Passed](https://github.com/user-attachments/assets/PLACEHOLDER_ec2_instance)

---

## Tech Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.13 | Core development |
| **Backend** | FastAPI | REST routing + Jinja2 template serving |
| **ML** | Scikit-Learn, CatBoost | Preprocessing pipeline, model benchmarking |
| **Serialization** | Pickle | `model.pkl` + `preprocessor.pkl` artifacts |
| **Containerization** | Docker | `python:3.13-slim` base image |
| **Registry** | Amazon ECR | Private Docker image registry |
| **Compute** | AWS EC2, Elastic Beanstalk | Two parallel deployment targets |
| **CI/CD (v1)** | AWS CodePipeline | Source → Deploy, triggered on push |
| **CI/CD (v2)** | GitHub Actions | CI → Build/Push → Deploy, self-hosted runner |

---

## Project Structure

```
Project-Meridian/
│
├── .github/
│   └── workflows/
│       └── main.yml              # CI/CD: lint → build/push ECR → deploy EC2
│
├── src/
│   ├── components/
│   │   ├── data_ingestion.py     # Train/test split, artifact writing
│   │   ├── data_transformation.py # ColumnTransformer, preprocessing pickle
│   │   └── model_trainer.py      # Multi-model benchmark, best model pickle
│   ├── pipeline/
│   │   ├── predict_pipeline.py   # Inference entrypoint
│   │   └── train_pipeline.py     # Training entrypoint
│   ├── exception.py               # Custom exception with file/line context
│   ├── logger.py                  # Timestamped logging per run
│   └── utils.py
│
├── artifacts/
│   ├── model.pkl                  # Trained model
│   ├── preprocessor.pkl           # Fitted ColumnTransformer
│   ├── train.csv / test.csv / data.csv
│
├── templates/
│   ├── index.html                 # Landing page
│   └── predict.html               # Inference request form + result
│
├── notebook/
│   ├── EDA.ipynb
│   └── ModelTrain.ipynb
│
├── main.py                        # FastAPI application entrypoint
├── Dockerfile                     # python:3.13-slim, exposes :8000
├── Procfile                       # Elastic Beanstalk process definition
├── requirements.txt
├── setup.py
└── README.md
```

---

## Getting Started — Local

### 1. Clone the repository
```bash
git clone https://github.com/ayushhh026/Project-Meridian.git
cd Project-Meridian
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 5. Open in browser
```
http://localhost:8000
```

---

## Getting Started — Docker (v2 path locally)

```bash
docker build -t project-meridian .
docker run -d -p 8000:8000 project-meridian
```

---

## Key Engineering Decisions

**Why a self-hosted runner instead of GitHub's hosted runners?**
GitHub-hosted runners can't reach private infrastructure inside a VPC or a specific EC2 instance directly. Installing a self-hosted runner on the target EC2 instance means the deployment job executes commands *on the actual machine* — `docker pull`, `docker run` — rather than trying to push to it externally.

**Why pickle the preprocessor separately from the model?**
At inference time, raw input must go through the exact same transformation the model was trained on. Pickling `preprocessor.pkl` independently guarantees train-serve consistency — there's no risk of preprocessing drifting out of sync with what the model expects.

**Why two deployment paths for the same app?**
Elastic Beanstalk abstracts away almost all infrastructure decisions — good for shipping fast. Docker + ECR + EC2 with a self-hosted runner is the opposite: full control over the container, the image registry, and the exact deployment commands. Building both demonstrates the pipeline holds up under two very different operational models without changing a line of ML code.

**Why decouple `predict_pipeline.py` from `main.py`?**
FastAPI's route handler only calls `PredictPipeline.predict()` — it has no knowledge of `ColumnTransformer`, model internals, or artifact paths. This means the serving framework is swappable (FastAPI today, could be a CLI script or batch job tomorrow) without touching any ML logic.

---

## Roadmap

- [ ] Add automated tests in the CI stage (currently placeholder echo statements)
- [ ] ECS/Fargate deployment as a v3 (no manual EC2 management)
- [ ] Model versioning via MLflow
- [ ] Swap in a different structured dataset to validate the "dataset-agnostic" claim end to end

---

## License

[MIT License](LICENSE) — free to use, modify, and distribute with attribution.

---

## Author

**Ayush Shetty**
AI & Data Science Engineering Student

[![GitHub](https://img.shields.io/badge/GitHub-ayushhh026-181717?style=flat-square&logo=github)](https://github.com/ayushhh026)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Ayush_Shetty-0A66C2?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/ayush-shetty-830a03281/)

---

<div align="center">

**⭐ Star this repo if it helped you — it keeps the project alive.**

*Same pipeline, two deployment paths, zero shortcuts on either.*

</div>
