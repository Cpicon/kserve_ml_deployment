# KServe ML Deployment Project

This repository provides a **batteries-included template** for serving machine-learning models on Kubernetes with the following stack:

* **Istio** for service mesh & ingress
* **Knative Serving** for serverless autoscaling
* **KServe** for model management & prediction endpoints

The goal is to give you an opinionated, yet extensible starting point that you can run locally on Kind _or_ promote to any managed Kubernetes service.

---
## Project Structure

```
.
├── environments/           # environment-specific assets
│   ├── dev/                # placeholders for dev-cluster overrides
│   ├── stage/              # placeholders for staging overrides
│   ├── prod/               # production overrides (Helm values, manifests, …)
│   └── local/              # scripts & manifests for spinning up a full stack on Kind
│       └── test/           # demo model, inference script & payload
├── model/                  # example pretrained artefacts (sklearn iris)
├── scripts/                # generic helper scripts (kept for backwards-compat)
└── src/                    # put your model code, custom predictors, transformers, … here
```

---
## Usage At A Glance

1. **Local experimentation** – follow the instructions in  `environments/local/README.md` to launch a one-node Kind cluster, install Istio/Knative/KServe and deploy the sample *sklearn-iris* model.
2. **CI/CD / Production** – adapt the manifests in `environments/prod` (and friends) to match your cloud provider, storage and security requirements, then apply them through your favourite GitOps tool.

---
## Developer Notes

* You will find the full step-by-step tutorial (including Swagger-UI tips) in:

  ```
  environments/local/README.md
  ```


* Contribute improvements or bug fixes via pull request. Make sure to run `bash environments/local/test/test_inference.sh …` before submitting so that the sample pipeline stays green.