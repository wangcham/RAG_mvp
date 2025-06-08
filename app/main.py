# main.py
from quart import Quart, request, jsonify
import asyncio
import logging
import os
from RAG_Manager import RAG_Manager
from services.embedding_models import EMBEDDING_MODEL_CONFIGS, DEFAULT_EMBEDDING_MODEL_TYPE, DEFAULT_EMBEDDING_MODEL_NAME

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Quart(__name__)

rag_manager: RAG_Manager = None

@app.before_serving
async def startup():
    global rag_manager
    logger.info("Quart app startup initiated.")
    try:
        model_type_to_use = os.getenv("EMBEDDING_MODEL_TYPE", DEFAULT_EMBEDDING_MODEL_TYPE)
        model_name_to_use = os.getenv("EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME)

        # Validate configuration
        if model_name_to_use not in EMBEDDING_MODEL_CONFIGS or \
           EMBEDDING_MODEL_CONFIGS[model_name_to_use]["type"] != model_type_to_use:
            logger.warning(
                f"Configured EMBEDDING_MODEL_TYPE/NAME ('{model_type_to_use}/{model_name_to_use}') "
                "is not a valid combination in EMBEDDING_MODEL_CONFIGS. Falling back to defaults."
            )
            model_type_to_use = DEFAULT_EMBEDDING_MODEL_TYPE
            model_name_to_use = DEFAULT_EMBEDDING_MODEL_NAME

        rag_manager = RAG_Manager(embedding_model_type=model_type_to_use,
                                 embedding_model_name=model_name_to_use)
        await rag_manager.initialize_system()
        logger.info(f"Quart app startup: RAG_Manager initialized successfully with model '{model_name_to_use}'.")
    except Exception as e:
        logger.critical(f"Failed to initialize RAG_Manager during startup: {e}", exc_info=True)
        # In a production environment, you might want to terminate the application if this fails
        # import sys
        # sys.exit(1)

@app.after_serving
async def shutdown():
    logger.info("Quart app shutdown initiated.")
    logger.info("Quart app shutdown complete.")

@app.route("/")
async def health_check():
    return jsonify({"status": "healthy", "message": "RAG system is up and running!"}), 200

@app.route("/store", methods=["POST"])
async def store_data_endpoint():
    try:
        data = await request.get_json()
        file_path = data.get("file_path")
        kb_name = data.get("kb_name")
        file_type = data.get("file_type", "txt")
        kb_description = data.get("kb_description", "Default knowledge base description.")

        if not file_path or not kb_name:
            logger.warning(f"Bad request for /store: Missing file_path or kb_name. Request data: {data}")
            return jsonify({"error": "Missing 'file_path' or 'kb_name' in request body."}), 400

        if not os.path.exists(file_path):
            logger.info(f"Creating dummy file for testing: {file_path}")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"This is a dummy content for {os.path.basename(file_path)}. "
                        f"It belongs to the '{kb_name}' knowledge base. "
                        f"Large Language Models are powerful. Retrieval Augmented Generation improves them."
                        f"\n\nAdditional text to ensure multiple chunks for testing different models.")

        logger.info(f"Received /store request for file: {file_path}, KB: {kb_name}")
        await rag_manager.store_data(
            file_path=file_path,
            kb_name=kb_name,
            file_type=file_type,
            kb_description=kb_description
        )
        logger.info(f"Successfully processed file: {file_path} for storage.")
        return jsonify({"message": f"Successfully processed file: {file_path}"}), 200
    except Exception as e:
        logger.error(f"API Error in /store: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred during storage: {str(e)}"}), 500

@app.route("/retrieve", methods=["POST"])
async def retrieve_data_endpoint():
    try:
        data = await request.get_json()
        query = data.get("query")
        if not query:
            logger.warning("Bad request for /retrieve: Missing query.")
            return jsonify({"error": "Missing 'query' in request body."}), 400

        logger.info(f"Received /retrieve request for query: '{query}'")
        results = await rag_manager.retrieve_data(query)
        logger.info(f"Successfully retrieved {len(results)} chunks for query: '{query}'.")

        return jsonify({"query": query, "retrieved_chunks": results}), 200
    except Exception as e:
        logger.error(f"API Error in /retrieve: {e}", exc_info=True)
        return jsonify({"error": f"An internal error occurred during retrieval: {str(e)}"}), 500

if __name__ == "__main__":
    logger.info("Starting Quart application...")
    app.run(host='0.0.0.0', port=5000)