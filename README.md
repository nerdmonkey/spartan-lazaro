
<p align="center"><img src="docs/ssf_banner.png" alt="Lazaro - Based on Spartan Serverless Framework"></p>

# Lazaro


## About
**Lazaro** is a modern, extensible, and opinionated serverless framework for building scalable cloud-native applications with Python. Lazaro is based on the Spartan Serverless Framework, inheriting its robust architecture and best practices, while introducing new features and branding.

**Note:** While Lazaro is based on Spartan Serverless Framework, it is designed specifically for use on Google Cloud Platform (GCP) only. Support for other cloud providers has been removed to focus on providing the best possible experience and integration with GCP services.

Lazaro streamlines your development process and ensures code consistency, allowing you to build scalable and efficient applications on GCP with ease.


#### Lazaro is versatile and can be used to efficiently develop:
- RESTful APIs
- Cloud Functions and Event-driven workflows
- Small or Medium-sized ETL Pipelines
- Containerized Microservices on Cloud Run
- Pub/Sub message processing
- Agentic AI (Coming Soon)



Lazaro is fully tested and optimized for Google Cloud Platform, making it the ideal choice for serverless applications on GCP.



## Installation & Usage


1. **Install Lazaro (based on Spartan):**
```bash
pip install python-spartan
```

2. **Try it out:**
```bash
spartan --help
```

3. **Set up your environment:**

<details>
<summary><strong>▶️ For Linux / macOS</strong></summary>

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

</details>

<details>
<summary><strong>🪟 For Windows PowerShell</strong></summary>

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```

</details>

<details>
<summary><strong>🪟 For Windows CMD / DOS</strong></summary>

```cmd
python -m venv .venv
.venv\Scripts\activate.bat
pip install -r requirements-dev.txt
```

</details>

4. **Copy and configure environment variables:**

```bash
cp .env.example .env  # Linux/macOS
```

```powershell
copy .env.example .env  # PowerShell
```

```cmd
copy .env.example .env  # CMD
```


5. **Run database migration:**
```bash
spartan migrate init -d sqlite
spartan migrate upgrade
```

6. **Insert seed data:**
```bash
spartan db seed
```

---

## Running the Application

### Option 1: Run directly with Python
```bash
python main.py
```

### Option 2: Run with Functions Framework
```bash
functions-framework --target=main
```

---

## Sending a Test CloudEvent

You can test the endpoint using `curl` once the app is running (default: `localhost:8080`):

```bash
curl -X POST localhost:8080 \
  -H "Content-Type: application/cloudevents+json" \
  -d '{
    "specversion" : "1.0",
    "type" : "google.cloud.pubsub.topic.v1.messagePublished",
    "source" : "//pubsub.googleapis.com/projects/my-project/topics/my-topic",
    "subject" : "123",
    "id" : "A234-1234-1234",
    "time" : "2018-04-05T17:31:00Z",
    "data" : "Hello Spartan Lazaro!"
}'
```


### Deploy to Google Cloud Functions

Deploy your function to GCP:

```bash
# Deploy as HTTP function
gcloud functions deploy lazaro-function \
  --runtime python311 \
  --trigger-http \
  --entry-point main \
  --allow-unauthenticated

# Deploy as Pub/Sub triggered function
gcloud functions deploy lazaro-function \
  --runtime python311 \
  --trigger-topic my-topic \
  --entry-point main
```

---

## Testing

Run the test suite using `pytest`:

```bash
pytest -vv
```

---


## Migration from Spartan

Lazaro is based on the Spartan Serverless Framework. If you are migrating from Spartan, most core concepts and APIs remain compatible. See the documentation for migration tips and new features.

## Changelog

Please see [CHANGELOG](CHANGELOG.md) for recent updates.

---


## Contributing

Please see [CONTRIBUTING](./docs/CONTRIBUTING.md) for details on contributing.

---

## Security Vulnerabilities

Please review [our security policy](../../security/policy) for how to report vulnerabilities.

---


## Credits

- [Sydel Palinlin](https://github.com/nerdmonkey)
- [All Contributors](../../contributors)

---


## License

The MIT License (MIT). Please see the [License File](LICENSE) for more information.
