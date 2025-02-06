# CCMT Counseling Assistant

Welcome to the CCMT Counseling Assistant repository! This project is a Generative AI Chatbot designed to assist users with information related to the Centralized Counseling for M.Tech./M.Arch./M.Plan./M.Des. (CCMT) process. The chatbot leverages advanced AI models to provide accurate and timely information to prospective students.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)

## Features

- **Generative AI Model**: Powered by LLAMA2 to generate human-like responses.
- **Robust Guardrail System**: Implemented to minimize hallucinations and ensure accurate information dissemination.
- **Optimized Inference**: Utilizes VLLM to enhance inference times, ensuring quick responses.
- **Resource Efficient**: Tested on a 24 GB GPU machine for optimal performance.
- **Retrieval-Augmented Generation (RAG)**: Built using the CCMT-2023 Information Brochure to provide precise and relevant information.

## Architecture

The chatbot employs a Retrieval-Augmented Generation (RAG) architecture, combining the strengths of information retrieval and generative modeling. This approach ensures that the chatbot provides accurate and contextually relevant responses by retrieving pertinent information from the CCMT-2023 Information Brochure and generating coherent answers.

## Installation

To set up the CCMT Counseling Assistant locally, follow these steps:

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/animesh1012/genai_chatbot.git
   cd genai_chatbot/MTech_Project

   python3 -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

   pip install -r requirement.txt

   python app.py



