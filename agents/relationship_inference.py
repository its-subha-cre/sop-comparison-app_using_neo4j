from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from llm.factory import LLMFactory

class RelationshipDecision(BaseModel):
    target_chunk_id: str = Field(
        description="The ID of the target chunk to connect to."
    )
    relationship_type: str = Field(
        description="Dynamic relationship type decided by the LLM (e.g. HAS_SECTION, DIFFERS_FROM, REQUIRES, MITIGATES, COMPLIES_WITH, CONTRADICTS)."
    )
    confidence: float = Field(
        description="Similarity or linking confidence score between 0.0 (low) and 1.0 (high)."
    )
    diff_type: Optional[str] = Field(
        None,
        description="Classification of differences: exact_match, semantic_match, cosmetic, addition, deletion, modification, role_change, control_change."
    )
    rationale: str = Field(
        description="Short natural-language rationale explaining why this relationship is justified."
    )

class SemanticNodeMapping(BaseModel):
    chunk_id: str = Field(description="The ID of the current chunk.")
    node_type: str = Field(
        description="The dynamic Neo4j node label (e.g. Section, Clause, Requirement, SafetyCheck, Role, Equipment constraint)."
    )
    properties: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Cleaned properties for the node, e.g. text, title, or clause number."
    )
    relationships: List[RelationshipDecision] = Field(
        default_factory=list,
        description="List of outbound semantic relationships decided by the LLM."
    )

class RelationshipInferenceAgent:
    """LLM Agent responsible for semantic parsing, node labeling, and dynamic edge determination."""
    
    def __init__(self):
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert Regulatory Compliance Graph Architect and LLM Agent.
            Your job is to read a document chunk alongside its surrounding context and paired candidate chunks, 
            determine its dynamic semantic node classification, and decide which Neo4j relationships to wire.

            You are free to assign any logical Node Labels (e.g., Section, Clause, Requirement, SafetyCheck, Role, Equipment) 
            and define any arbitrary, expressive Relationship Types (e.g., HAS_CLAUSE, DIFFERS_FROM, COMPLIES_WITH, CONTRADICTS, MITIGATES, REQUIRES, SUPPORTED_BY) 
            that represent the true semantic links.
            
            Use 'DIFFERS_FROM' when mapping local discrepancies against the global standard.
            """),
            ("user", """
            Analyze the following document chunk:
            - Chunk ID: {chunk_id}
            - Content: "{chunk_content}"
            
            Contextual details:
            - Parent Metadata: {parent_context}
            - Sibling Chunks: {sibling_context}
            
            Candidate Paired SOP Chunks (for difference comparisons):
            {candidate_pairs}

            Determine the appropriate dynamic Node Type and any semantic relationships to generate.
            Ensure you compare this chunk against candidate pairs to decide if a DIFFERS_FROM or similar relation applies.
            Provide detailed, audit-traceable rationales.
            """)
        ])

    def infer_relationships(self, chunk_id: str, chunk_content: str, 
                            parent_context: dict, sibling_context: list, 
                            candidate_pairs: list) -> SemanticNodeMapping:
        """Invokes LLM with structured output parsing to obtain semantic nodes and edges mapping."""
        # Retrieve client from dynamic provider config factory
        model = LLMFactory.get_chat_model()
        
        # Equip client with structured output schemas
        structured_llm = model.with_structured_output(SemanticNodeMapping)
        
        # Construct and call prompt chain
        chain = self.prompt_template | structured_llm
        result = chain.invoke({
            "chunk_id": chunk_id,
            "chunk_content": chunk_content,
            "parent_context": str(parent_context),
            "sibling_context": str(sibling_context),
            "candidate_pairs": str(candidate_pairs)
        })
        
        return result