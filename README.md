<p align="center"><img src="docs/ssf_banner.png" alt="Social Card of Spartan"></p>

<h1 align="center">Lazaro — Spartan for GCP</h1>

<p align="center">
  <a href="https://github.com/nerdmonkey/spartan-lazaro/actions/workflows/lint.yml"><img src="https://github.com/nerdmonkey/spartan-lazaro/actions/workflows/lint.yml/badge.svg" alt="Lint"></a>
  <a href="https://github.com/nerdmonkey/spartan-lazaro/actions/workflows/tests.yml"><img src="https://github.com/nerdmonkey/spartan-lazaro/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://github.com/nerdmonkey/spartan-lazaro/actions/workflows/security.yml"><img src="https://github.com/nerdmonkey/spartan-lazaro/actions/workflows/security.yml/badge.svg" alt="Security"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License: MIT"></a>
  <img src="https://img.shields.io/badge/python-3.11%2B-blue.svg" alt="Python 3.11+">
</p>

## About

Lazaro is the GCP variant of the Spartan Serverless Framework. It streamlines your development process and ensures code consistency, allowing you to build scalable and efficient applications on Google Cloud with ease.

Lazaro is versatile and can be used to efficiently develop:

- RESTful APIs (Cloud Functions HTTP triggers)
- Event-driven workloads (Pub/Sub, CloudEvents)
- Small or medium-sized ETL pipelines
- Agentic AI (coming soon)

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Changelog](#changelog)
- [Contributing](#contributing)
- [Security Vulnerabilities](#security-vulnerabilities)
- [Credits](#credits)
- [License](#license)

## Features

| **Feature Category**           | **Status**                   | **Details**                                                  |
| ------------------------------ | ---------------------------- | ------------------------------------------------------------ |
| **Google Functions Framework** | ✅ Excellent                  | GCP-native CloudEvent support, event-driven, typed functions |
| **Pydantic Integration**       | ✅ Full Support               | Validation, serialization, EmailStr, type safety             |
| **Architecture Patterns**      | ✅ Robust                     | Service pattern, clean separation of concerns                |
| **Testing Framework**          | ✅ Fully Integrated           | pytest, mocking, coverage tools                              |
| **Code Quality Tools**         | ✅ Complete                   | Black, isort, flake8, mypy, bandit, pre-commit               |
| **Development Workflow**       | ✅ Streamlined                | Poetry, Tox, environment support                             |
| **Cloud-Native Features**      | ✅ Advanced                   | Tasks, secrets, parameter manager, multi-cloud hooks         |
| **Observability & Monitoring** | ✅ Enterprise-Grade           | Structured logging, tracing, exception handling              |
| **Developer Experience**       | ✅ High                       | Docker, Serverless Framework, Terraform, .env support                              |
| **Security Best Practices**    | ✅ Strong                     | Hashing, input validation, secrets handling                  |
| **Scalability Features**       | ✅ Built-in                   | Pagination, filtering, bulk operations                       |
| **Logging Support**            | ✅ Advanced                   | Factory logger types (file, stream, GCP), structured output  |
| **GCP Cloud Logging**          | ✅ Fully Integrated           | Trace context, severity levels, resource detection           |
| **Structured Logs**            | ✅ JSON + Metadata            | PII redaction, function source, custom metadata              |
| **Observability Hooks**        | ✅ Extensible                 | Factory patterns for loggers/tracers, sampling               |
| **Reusability**                | ✅ High                       | Abstract base classes, reusable modules                      |
| **Modular Architecture**       | ✅ Excellent                  | Factory design, reusable services/utilities                  |
| **Configuration Management**   | ✅ Centralized                | Pydantic + .env + environment-detection                      |
| **Cross-Platform Support**     | ✅ Multi-Cloud Ready          | GCP, AWS, local support via abstraction layers                |
| **Code Consistency**           | ✅ Consistent with minor gaps | Naming conventions, model structures, unified patterns       |

## Requirements

- Python 3.11+
- pip (or [Poetry](https://python-poetry.org/), which the project's tox environments use)
- [`python-spartan`](https://pypi.org/project/python-spartan/) CLI (`pip install python-spartan`)
- [Google Cloud SDK](https://cloud.google.com/sdk) (`gcloud`) — only needed for deploying to Cloud Functions

## Installation

Clone the repo:

```bash
git clone https://github.com/nerdmonkey/spartan-lazaro.git
cd spartan-lazaro
```

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

## Usage

### Run directly with Python

```bash
python main.py
```

### Run with Functions Framework

```bash
functions-framework --target=main
```

### Sending a Test CloudEvent

Test the endpoint with `curl` once the app is running (default: `localhost:8080`):

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

```bash
# Deploy as HTTP function
gcloud functions deploy spartan-function \
  --runtime python311 \
  --trigger-http \
  --entry-point main \
  --allow-unauthenticated

# Deploy as Pub/Sub triggered function
gcloud functions deploy spartan-function \
  --runtime python311 \
  --trigger-topic my-topic \
  --entry-point main
```

## Testing

Run the unit test suite with coverage:

```bash
source .venv/bin/activate
python -m pytest tests/unit -q --cov=app --cov=config --cov-report=term-missing
```

Alternatively, via tox (installs dependencies through Poetry):

```bash
tox -e coverage
```

## Changelog

Please see [CHANGELOG](CHANGELOG.md) for more information on what has changed recently.

## Contributing

Please see [CONTRIBUTING](./docs/CONTRIBUTING.md) for details.

## Security Vulnerabilities

Please review [our security policy](../../security/policy) on how to report security vulnerabilities.

## Credits

- [Sydel Palinlin](https://github.com/nerdmonkey)
- [All Contributors](../../contributors)

## License

The MIT License (MIT). Please see [License File](LICENSE) for more information.
