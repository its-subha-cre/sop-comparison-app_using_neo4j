import os
import uuid
from flask import Flask, request, jsonify
from flask_cors import CORS
from config import config_instance
from api.config import config_bp
from agents.orchestrator import PipelineOrchestrator

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Register blueprints
app.register_blueprint(config_bp, url_prefix='/api/config')

# In-memory storage for pipeline runs and statuses
pipeline_jobs = {}

# Set upload folders
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
GLOBAL_UPLOAD_DIR = os.path.join(UPLOAD_FOLDER, 'global')
LOCAL_UPLOAD_DIR = os.path.join(UPLOAD_FOLDER, 'local')

os.makedirs(GLOBAL_UPLOAD_DIR, exist_ok=True)
os.makedirs(LOCAL_UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'.pdf', '.docx'}

def allowed_file(filename):
    ext = os.path.splitext(filename)[1].lower()
    return ext in ALLOWED_EXTENSIONS

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Handles multi-file uploads for both Global SOP and Local SOP slots."""
    slot = request.form.get('slot')  # 'global' or 'local'
    if slot not in ['global', 'local']:
        return jsonify({"success": False, "message": "Invalid upload slot specified. Must be 'global' or 'local'."}), 400

    if 'files' not in request.files:
        return jsonify({"success": False, "message": "No files found in request."}), 400

    files = request.files.getlist('files')
    saved_files = []
    
    target_dir = GLOBAL_UPLOAD_DIR if slot == 'global' else LOCAL_UPLOAD_DIR

    for file in files:
        if file and allowed_file(file.filename):
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(target_dir, filename)
            file.save(filepath)
            saved_files.append({
                "original_name": file.filename,
                "saved_path": filepath
            })
        else:
            return jsonify({
                "success": False, 
                "message": f"File type not allowed or empty file. Acceptable formats: {list(ALLOWED_EXTENSIONS)}"
            }), 400

    return jsonify({
        "success": True,
        "slot": slot,
        "files": saved_files
    })

@app.route('/api/compare/run', methods=['POST'])
def run_comparison():
    """Starts the comparison pipeline job asynchronously using in-memory threads."""
    data = request.json or {}
    global_file = data.get('globalFile')
    local_files = data.get('localFiles', [])
    
    if not global_file or not local_files:
        return jsonify({
            "success": False,
            "message": "Both globalFile path and at least one localFiles path are required."
        }), 400

    job_id = str(uuid.uuid4())
    
    # Initialize job tracking state
    pipeline_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "stage": "queued",
        "errors": [],
        "report": {},
        "progress": 0
    }

    # Internal worker function to run pipeline in thread
    def worker():
        orchestrator = PipelineOrchestrator()
        pipeline_jobs[job_id]["status"] = "in_progress"
        try:
            result = orchestrator.run_pipeline(
                global_path=global_file, 
                local_paths=local_files
            )
            pipeline_jobs[job_id].update(result)
        except Exception as e:
            pipeline_jobs[job_id]["status"] = "failed"
            pipeline_jobs[job_id]["errors"].append(str(e))

    # Trigger worker asynchronously (using threading for local environment reliability)
    import threading
    thread = threading.Thread(target=worker)
    thread.start()

    return jsonify({
        "success": True,
        "job_id": job_id,
        "message": "Comparison pipeline triggered successfully."
    })

@app.route('/api/compare/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Retrieves pipeline execution status and results."""
    job = pipeline_jobs.get(job_id)
    if not job:
        return jsonify({"success": False, "message": f"Job ID {job_id} not found."}), 404
        
    return jsonify({
        "success": True,
        "job": job
    })

@app.route('/api/chat', methods=['POST'])
def chat_assistant():
    """Provides LLM response about SOP comparison results and raw document context."""
    from llm.factory import LLMFactory
    
    data = request.json or {}
    message = data.get('message')
    job_id = data.get('jobId')
    
    if not message:
        return jsonify({"success": False, "message": "Message parameter is required."}), 400
        
    job_context = ""
    if job_id and job_id in pipeline_jobs:
        job = pipeline_jobs[job_id]
        if job.get("status") == "completed":
            report = job.get("report", {})
            best_match = report.get("best_match", {})
            
            # Smart Dynamic Filter: check if user asks about specific clauses or keywords
            user_query = message.lower()
            detailed_recs = []
            light_metadata = []
            
            for k, v in report.get("sop_results", {}).items():
                filename = v.get("name")
                for r in v.get("recommendations", []):
                    cls_num = r.get("clause_number", "").lower()
                    global_txt = r.get("global_text", "").lower()
                    local_txt = r.get("local_text", "").lower()
                    
                    # Detect relevance: user specifically asking about this clause, text, or file
                    is_relevant = False
                    for word in user_query.split():
                        if len(word) > 1 and (word in cls_num or word in global_txt or word in local_txt or word in filename.lower()):
                            is_relevant = True
                            break
                            
                    if is_relevant:
                        detailed_recs.append(r)
                    else:
                        light_metadata.append({
                            "file": filename,
                            "clause": r.get("clause_number"),
                            "action": r.get("action")
                        })
                        
            job_context = f"""
            Active SOP Comparison Results Context:
            - Summary: {report.get('summary')}
            - Best Match SOP: {best_match.get('name')} with score {best_match.get('similarity_score')}%
            - Summary list of other mismatches: {str(light_metadata[:15])}
            """
            
            if detailed_recs:
                job_context += f"""
                - USER-REQUESTED DETAILED COMPLIANCE MISMATCH CLAUSES (Use these exact texts to answer the user's specific query):
                  {str(detailed_recs[:5])}
                """
            
    system_prompt = f"""
    You are the 'SOP Compliance Assistant', an AI advisor integrated into a multi-agent SOP comparison platform.
    Your goal is to help users understand the results of scanning Global and Local SOPs, explain recommendations (e.g. keeping, retiring, or merging clauses), 
    and answer questions regarding standard operating procedures.
    
    {job_context}

    === AUDIENCE AND TOPIC GUARDRAILS ===
    - You must ONLY answer questions that are related to Standard Operating Procedures (SOPs), document compliance, regulatory guidelines, audits, section mismatches, or the dashboard results shown in the app.
    - If the user asks general, unrelated, or conversational questions (e.g., jokes, stories, trivia, coding problems, general life advice like "how to be a good boy"), you MUST politely decline to answer.
    - If a question is out-of-scope, respond EXACTLY with this message:
      "I am your SOP Compliance Assistant. I can only help you with questions related to Standard Operating Procedures, regulatory compliance, mismatches, or dashboard metrics. Please ask an SOP-related question."
    ======================================
    
    Answer clearly and professionally. Keep answers concise since they display in a sidebar.
    """
    
    try:
        chat_model = LLMFactory.get_chat_model()
        from langchain_core.messages import SystemMessage, HumanMessage
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=message)
        ]
        res = chat_model.invoke(messages)
        return jsonify({
            "success": True,
            "response": res.content.strip()
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        # If the LLM key is configured, return the actual error so the user can debug (e.g. invalid key, model mismatch)
        if config_instance.is_configured():
            return jsonify({
                "success": True,
                "response": f"AI Assistant Error: {str(e)}. Please check your API key or model configuration in the settings."
            })
        # Fallback response for unconfigured local testing
        return jsonify({
            "success": True,
            "response": f"[Local Assistant Demo] I received your question: '{message}'. To generate live compliance answers, please configure your LLM Engine API key in the settings wizard."
        })

@app.route('/api/graph/data', methods=['GET'])
def get_graph_data():
    from graph.writer import GraphWriter
    
    nodes = []
    links = []
    node_ids = set()
    
    try:
        writer = GraphWriter()
        writer.connect()
        driver = writer._driver
        
        with driver.session() as session:
            result = session.run("MATCH (n) OPTIONAL MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 150")
            for record in result:
                n = record.get("n")
                r = record.get("r")
                m = record.get("m")
                
                if n:
                    # Retrieve node ID safely
                    n_id = str(n.element_id) if hasattr(n, 'element_id') else str(n.id)
                    if n_id not in node_ids:
                        node_ids.add(n_id)
                        labels = list(n.labels)
                        label = labels[0] if labels else "Node"
                        props = dict(n)
                        name = props.get("name") or props.get("title") or props.get("number") or n_id
                        nodes.append({
                            "id": n_id,
                            "label": label,
                            "name": name,
                            "properties": props
                        })
                        
                if m:
                    m_id = str(m.element_id) if hasattr(m, 'element_id') else str(m.id)
                    if m_id not in node_ids:
                        node_ids.add(m_id)
                        labels = list(m.labels)
                        label = labels[0] if labels else "Node"
                        props = dict(m)
                        name = props.get("name") or props.get("title") or props.get("number") or m_id
                        nodes.append({
                            "id": m_id,
                            "label": label,
                            "name": name,
                            "properties": props
                        })
                        
                if r and n and m:
                    n_id = str(n.element_id) if hasattr(n, 'element_id') else str(n.id)
                    m_id = str(m.element_id) if hasattr(m, 'element_id') else str(m.id)
                    links.append({
                        "source": n_id,
                        "target": m_id,
                        "type": r.type
                    })
                    
        writer.close()
        return jsonify({
            "success": True,
            "nodes": nodes,
            "links": links,
            "source": "neo4j"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Neo4j offline: {str(e)}",
            "nodes": [],
            "links": [],
            "source": "fallback"
        })


if __name__ == '__main__':
    port = config_instance.PORT
    print(f"Starting Multi-Agent SOP Comparison API on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=(config_instance.FLASK_ENV == 'development'))
