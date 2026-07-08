from typing import List, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm.factory import LLMFactory

class LLMChunk(BaseModel):
    id: str = Field(description="A unique alphanumeric identifier for this chunk (e.g., chunk_1).")
    label: str = Field(description="The dynamic node label/type decided by the LLM (e.g., Section, Clause, Requirement, Control, SafetyRule, Role, Constraint).")
    content: str = Field(description="The actual text content segment of this chunk.")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Key-value properties to store (e.g., title, number, site, equipment).")

class LLMChunkingResult(BaseModel):
    chunks: List[LLMChunk] = Field(description="The complete list of logical chunks identified in the text.")

class LLMChunkerAgent:
    """LLM Ingestion Agent that reads raw text and dynamically decides how to partition it into nodes."""

    def __init__(self):
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert Document Decomposition Agent. 
            Read the raw document text and partition it into logical chunks/nodes.
            
            You must dynamically decide:
            1. The boundaries of each chunk.
            2. The appropriate Node Label/Type (e.g., Section, Clause, Requirement, SafetyControl, Role, Constraint, etc.).
            3. The metadata attributes (title, index, scope, etc.) for each chunk.
            
            Make sure to capture the entire document without omitting details.
            """),
            ("user", "Document Title: {doc_title}\n\nRaw Text:\n{raw_text}")
        ])

    def chunk_document_with_llm(self, doc_title: str, raw_text: str) -> LLMChunkingResult:
        """Invokes LLM with structured output mapping to dynamically define nodes/chunks."""
        model = LLMFactory.get_chat_model()
        structured_llm = model.with_structured_output(LLMChunkingResult)
        chain = self.prompt | structured_llm
        
        result = chain.invoke({
            "doc_title": doc_title,
            "raw_text": raw_text[:8000]  # Limit context window to prevent truncation errors in basic runs
        })
        return result