# Unify-App
Designated Support App
## Features

- **LLM Coaching**: Provides personalized coaching using a large language model to assist users with their challenges.
- **Peer Connection**: Connects users with individuals who have experienced similar situations for support and guidance.

## Technology Stack

- **Backend**: Flask framework for building the application backend.
- **AI Integration**: Leveraging large language models for coaching functionality.
- **Database**: PostgreSQL

## Getting Started

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/Unify-App.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Unify-App
    ```
3. Create a `.env` file by copying the `.env.example` file and update the environment variables.
    ```bash
    cp .env.example .env
    ```
4. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Development Setup

This project uses Docker and `docker-compose` to run the PostgreSQL database.

1. **Start the database:**
    ```bash
    docker-compose up -d
    ```
2. **Create the database tables:**
    ```bash
    python create_db_tables.py
    ```
3. **Run the application:**
    ```bash
    flask run
    ```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.